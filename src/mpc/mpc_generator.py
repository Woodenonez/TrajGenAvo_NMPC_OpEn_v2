import os, sys

import traceback
import numpy as np

import opengen as og
import casadi.casadi as cs
from collections import Iterable

from utils.config import SolverParams


MAX_SOLVER_TIME_MICROS = 8_000_000 #500_000
MAX_OUTER_ITERATIONS = 15

def get_length(iterable):
    """Casadi can't handle len.
    Args:
        iterable ([type]): [description]
    Returns:
        int: length of iterable
    """
    try:
        return len(iterable)
    except:
        pass
    try:
        return iterable.shape[0]
    except:
        raise ValueError(f"Couldn't calculate length of {iterable}")

def rec_positive_check(costs):
    if isinstance(costs, Iterable) and not(type(costs).__module__ == cs.__name__ and  costs.is_constant() and costs.is_scalar()):
        for i in range(get_length(costs)):
            cost = costs[i]
            if not rec_positive_check(cost):
                return False
        return True

    positive = costs >= 0
    return positive

def cost_positive(func):
    """Decorator function that checks if the cost returned is positive.

    Args:
        func ([type]): [description]
    """
    def wrapper(*args, **kwargs):
        # Calculate the cost
        cost = func(*args, **kwargs)

        # Check if the cost is a casadi symbolic expression or an iterable of casadi symbolic expressions. 
        if type(cost).__module__ == cs.__name__ and not cost.is_constant():
            return cost
        elif isinstance(cost, Iterable) and type(cost[0]).__module__ == cs.__name__ and not cost[0].is_constant():
            return cost 

        # Check if the cost is negative or an iterable of negative costs.
        positive = rec_positive_check(cost)
        if not positive:
            raise ValueError(f"The cost cannot be negative. It is: {cost}")
        return cost
    return wrapper

def check_shape(variable_name, requested_shape, var):
    try:
        var.shape
    except:
        traceback.print_exc()
        raise Exception(f"{variable_name} must have .shape property.")

    # Special case for 1-D arrays
    if len(var.shape) == 1:
        return

    assert var.shape == requested_shape, f"Wrong shape. {variable_name} must have shape: {requested_shape}. It had shape: {var.shape}."


class MpcModule:
    def __init__(self, solver_param:SolverParams):
        self.print_name = "[MPC-Generator]"
        self.obstacles = []                     # list of obstacles
        self.solver_param = solver_param
        self.ts = self.solver_param.base.ts
    
    # motion model (discretized via Runge-Kutta)
    def dynamics_ct(self, x, u): # δ(state) per ts
        dx = self.ts * (u[0]*cs.cos(x[2]))
        dy = self.ts * (u[0]*cs.sin(x[2]))
        dtheta = self.ts * u[1]
        return cs.vertcat(dx, dy, dtheta)
    def dynamics_rk1(self, x, u): # discretized via Runge-Kutta 1 (Euler method)
        return x + self.dynamics_ct(x, u)
    def dynamics_rk4(self, x, u): # discretized via Runge-Kutta 4
        k1 = self.dynamics_ct(x, u)
        k2 = self.dynamics_ct(x + 0.5*k1, u)
        k3 = self.dynamics_ct(x + 0.5*k2, u)
        k4 = self.dynamics_ct(x + k3, u)
        x_next = x + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        return x_next
    def dynamics_rk4_dual(self, x, u):
        x_master = x[0:3]
        x_slave  = x[3:]
        u_master = u[0:2]
        u_slave  = u[2:] 

        x_next_master = self.dynamics_rk4(x_master, u_master)
        x_next_slave  = self.dynamics_rk4(x_slave, u_slave)
        return cs.vertcat(x_next_master, x_next_slave)

    # obstacle scan
    def inside_dyn_ellipse2(self, x, y, a0_t, a1_t, a2_t, a3_t, c0_t, c1_t, active_obs):
        #1 - (a1*(c0-x) + a3*(c1-y)) * (c1-y) - (a0*(c0-x) + a2*(c1-y)) * (c0-x)
        crash = 0 
        for obs in range(0, self.solver_param.base.n_dyn_obs):
            ellips_eq = 1 - (a1_t[obs]*(c0_t[obs] - x) + a3_t[obs]*(c1_t[obs] - y))*(c1_t[obs] - y) - (a0_t[obs]*(c0_t[obs] - x) + a2_t[obs]*(c1_t[obs] - y))*(c0_t[obs] - x)
            crash += cs.fmax(0.0, active_obs[obs]*ellips_eq)**2
        return crash

    # cost term: cross-track-error
    def dist_to_line(self, pos, l1, l2):
        # line segment
        line = l2-l1
        # t_hat
        t_hat = cs.dot(pos-l1, line)/(line[0]**2+line[1]**2+1e-16)
        # limit t
        t_star = cs.fmin(cs.fmax(t_hat, 0.0), 1.0)
        # vector pointing from us to closest point
        temp_vec = l1 + t_star*line - pos
        # append distance (is actually squared distance)
        squared_distance = temp_vec[0]**2+temp_vec[1]**2
        return squared_distance

    def get_closest_line(self, pos, lines_x, lines_y):
        try:
            lines_x.shape
        except:
            raise Exception("lines_x must have attribute shape (lines_x.shape).")

        distances = cs.SX.ones(1)
        s2 = cs.vertcat(lines_x[0], lines_y[0])
        for i in range(1, self.solver_param.base.n_hor):
            # set start point as previous end point
            s1 = s2
            # new end point
            s2 = cs.vertcat(lines_x[i], lines_y[i])

            squared_distance = self.dist_to_line(pos, s1, s2)
            distances = cs.horzcat(distances, squared_distance)
        return cs.mmin(distances[1:])

    @cost_positive
    def cost_dist2ref_line(self, x_all, y_all, th_all, ref_x, ref_y , q_dist, q_th, individual_costs=False):
        if not individual_costs:
            tot_cost = 0
        else:
            tot_cost = []

        for t in range(0, self.solver_param.base.n_hor):
            x = x_all[t]
            y = y_all[t]
            p = cs.vertcat(x, y)
            closest_line_dist = self.get_closest_line(p, ref_x, ref_y)
            
            # Add cost for angle error_
            if not individual_costs:
                tot_cost += (self.theta_ref_master - th_all[t])**2*q_th
            else:
                tot_cost.append((self.theta_ref_master - th_all[t])**2*q_th)
            # Add cost for deviating from line
            if not individual_costs:
                tot_cost += closest_line_dist*q_dist
            else:
                tot_cost[-1] += closest_line_dist*q_dist
        return tot_cost
    
    @cost_positive
    def cost_distance_between_atrs(self, all_x_master, all_y_master, all_x_slave, all_y_slave, q_distance, individual_costs=False):
        if not individual_costs:
            cost = 0
        else:
            cost = [] 
        for t in range(0, self.solver_param.base.n_hor):
            x_master = all_x_master[t]
            y_master = all_y_master[t]
            x_slave = all_x_slave[t]
            y_slave = all_y_slave[t]
            if not individual_costs:
                cost += cs.fmax(0, ((x_master-x_slave)**2 + (y_master-y_slave)**2 -(self.d)**2))*q_distance #Penalize deviations from distance
                cost += cs.fmax(0, (-(x_master-x_slave)**2 + -(y_master-y_slave)**2 +(self.d)**2))*q_distance
            else:
                cost.append(cs.fmax(0, ((x_master-x_slave)**2 + (y_master-y_slave)**2 -(self.d)**2))*q_distance)
                cost[-1] += cs.fmax(0, (-(x_master-x_slave)**2 + -(y_master-y_slave)**2 +(self.d)**2))*q_distance
                
        return cost
    
    @cost_positive
    def cost_distance_atr_soft_c(self, all_x_master, all_y_master, all_x_slave, all_y_slave, q_distance_c, individual_costs=False):
        if not individual_costs:
            cost = 0
        else:
            cost = [] 
        for t in range(0, self.solver_param.base.n_hor):
            x_master = all_x_master[t]
            y_master = all_y_master[t]
            x_slave = all_x_slave[t]
            y_slave = all_y_slave[t]
            if not individual_costs:
                cost += cs.fmax(0, ((x_master-x_slave)**2 + (y_master-y_slave)**2 -(self.d_ub)**2))*q_distance_c #Penalize deviations from distance
                cost += cs.fmax(0, (-(x_master-x_slave)**2 - (y_master-y_slave)**2 +(self.d_lb)**2))*q_distance_c
            else:
                cost.append(cs.fmax(0, ((x_master-x_slave)**2 + (y_master-y_slave)**2 -(self.d_ub)**2))*q_distance_c)
                cost[-1] += cs.fmax(0, (-(x_master-x_slave)**2 - (y_master-y_slave)**2 +(self.d_lb)**2))*q_distance_c

        return cost

    @cost_positive
    def cost_dist2ref_points(self, x_all, y_all, th_all, x_refs, y_refs, th_refs, q_pos, q_theta, individual_costs=False):
        check_shape("all_x", (self.solver_param.base.n_hor,1), x_all)
        check_shape("x_refs", (self.solver_param.base.n_hor,1), x_refs)
        
        if not individual_costs:
            tot_cost = 0
        else:
            tot_cost = []

        for t in range(0, self.solver_param.base.n_hor):
            x = x_all[t]
            y = y_all[t]
            theta = th_all[t]

            if not individual_costs:
                tot_cost += q_pos*((x-x_refs[t])**2 + (y-y_refs[t])**2) + q_theta*((theta-th_refs[t])**2)
            else:
                tot_cost.append(q_pos*((x-x_refs[t])**2 + (y-y_refs[t])**2) + q_theta*((theta-th_refs[t])**2))
        return tot_cost

    # cost term: control input
    @cost_positive
    def cost_control(self, u, q_lin_v, q_ang, individual_costs=False):
        lin_v = u[0::2]
        ang_v = u[1::2]
        if not individual_costs:
            lin_cost = cs.mtimes(lin_v.T, lin_v)*q_lin_v
            ang_cost = cs.mtimes(ang_v.T, ang_v)*q_ang
        else:
            lin_cost = lin_v**2*q_lin_v
            ang_cost = ang_v**2*q_ang
        return lin_cost, ang_cost

    def calc_accelerations(self, u, lin_vel_init, ang_vel_init):
        lin_vel = u[0::2]
        ang_vel = u[1::2]
        # Accelerations
        lin_acc= (lin_vel - cs.vertcat(lin_vel_init, lin_vel[0:-1]))/self.solver_param.base.ts
        ang_acc = (ang_vel - cs.vertcat(ang_vel_init, ang_vel[0:-1]))/self.solver_param.base.ts
        return lin_acc, ang_acc
    
    @cost_positive
    def cost_angular_acc(self, acc_list, q_acc, individual_costs=False):      
        if not individual_costs:
            cost = cs.mtimes(acc_list.T, acc_list)*q_acc
        else:
            cost = acc_list**2*q_acc
        return cost

    @cost_positive
    def cost_linear_acc(self, acc_list, q_acc, q_ret, individual_costs=False):   
        # forward acceleration and retarding acceleration are differently weighted (allow emergency stop)         
        acc_up = cs.fmax(0, acc_list)
        acc_down = cs.fmin(0,acc_list)

        if not individual_costs:
            cost = cs.mtimes(acc_up.T, acc_up)*q_acc
            cost += cs.mtimes(acc_down.T, acc_down)*q_ret
        else:
            cost = acc_up**2*q_acc
            cost += acc_down**2*q_ret
            
        return cost

    @cost_positive
    def cost_inside_static_object(self, x_all, y_all, q, individual_costs=False):         
        if not individual_costs:
            crash_in_trajectory = 0 
        else:
            crash_in_trajectory = [] 
        
        for t in range(0, self.solver_param.base.n_hor):
            crash = 0
            x = x_all[t]
            y = y_all[t]
            base_vert_obj = 0 
            for vert in range(self.solver_param.base.min_vertices, self.solver_param.base.max_vertices + 1):                
                for obs in range(0, self.solver_param.base.n_obs_of_each_vertices):
                    inside = 1
                    for line in range(vert):
                        b  = self.bs_static[base_vert_obj + line + obs * vert] 
                        a0 = self.a0s_static[base_vert_obj + line + obs * vert]
                        a1 = self.a1s_static[base_vert_obj + line + obs * vert]
                        obs_eq = b - a0*x - a1*y
                        #h = (cs.fmax(0.0, obs_eq )/obs_eq)**2.0
                        h = cs.fmax(0.0, obs_eq )**2.0
                        inside = inside *h
                    # If all equations for each object is > 0, then inside is > 0 
                    crash += inside
                #Set new index for next object shape
                base_vert_obj += vert * self.solver_param.base.n_obs_of_each_vertices

            if not individual_costs:
                crash_in_trajectory +=crash*q
            else:
                crash_in_trajectory.append(crash*q)

        return crash_in_trajectory

    @cost_positive
    def cost_inside_dyn_ellipse2(self, x_all, y_all, q, individual_costs=False):
        #1 - (a1*(c0-x) + a3*(c1-y)) * (c1-y) - (a0*(c0-x) + a2*(c1-y)) * (c0-x)
        if not individual_costs:
            crash_in_trajectory = 0 
        else:
            crash_in_trajectory = [] 
        
            # Dynamic ellipse parameters for timestep t
        for t in range(0, self.solver_param.base.n_hor):
            crash = 0 
            x = x_all[t]
            y = y_all[t]
            a0_t = self.a0_dyn[t :: self.solver_param.base.n_hor]
            a1_t = self.a1_dyn[t :: self.solver_param.base.n_hor]
            a2_t = self.a2_dyn[t :: self.solver_param.base.n_hor]
            a3_t = self.a3_dyn[t :: self.solver_param.base.n_hor]
            c0_t = self.c0_dyn[t :: self.solver_param.base.n_hor]
            c1_t = self.c1_dyn[t :: self.solver_param.base.n_hor]
            #for obs in range(0, self.solver_param.base.n_dyn_obs):
            #    ellips_eq = 1 - (a1_t[obs]*(c0_t[obs] - x) + a3_t[obs]*(c1_t[obs] - y))*(c1_t[obs] - y) - (a0_t[obs]*(c0_t[obs] - x) + a2_t[obs]*(c1_t[obs] - y))*(c0_t[obs] - x)
            #    crash += cs.fmax(0.0, self.active_dyn_obs[obs]*ellips_eq)**2
            crash = self.inside_dyn_ellipse2(x,y,a0_t,a1_t, a2_t, a3_t, c0_t, c1_t, self.active_dyn_obs) 
            
            if not individual_costs:
                crash_in_trajectory +=crash*q
            else:
                crash_in_trajectory.append(crash*q)

        return crash_in_trajectory

    @cost_positive
    def cost_inside_future_dyn_ellipse(self, x_all, y_all, q):
        print("This should never be used") #TODO: Delete this if it's never used

        crash_in_trajectory = 0 
            # Dynamic ellipse parameters for timestep t
        for t in range(0, self.solver_param.base.n_hor):
            #crash = 0 
            x = x_all[t]
            y = y_all[t]

            for t_future in range(t+1, self.solver_param.base.n_hor):
                a0_t = self.a0_dyn[t_future :: self.solver_param.base.n_hor]
                a1_t = self.a1_dyn[t_future :: self.solver_param.base.n_hor]
                a2_t = self.a2_dyn[t_future :: self.solver_param.base.n_hor]
                a3_t = self.a3_dyn[t_future :: self.solver_param.base.n_hor]
                c0_t = self.c0_dyn[t_future :: self.solver_param.base.n_hor]
                c1_t = self.c1_dyn[t_future :: self.solver_param.base.n_hor]
                crash = self.inside_dyn_ellipse2(x,y,a0_t,a1_t, a2_t, a3_t, c0_t, c1_t, self.active_dyn_obs) 
                crash_in_trajectory +=crash#* ((1) / (t_future - t + 1)**2)
                #crash_in_trajectory +=crash * 0.9**(t_future-t) 

        return crash_in_trajectory*q
    
    def cost_v_ref_difference(self, all_u, v_ref, q_upper, q_lower, individual_costs=False):
        check_shape("all_u", (2*self.solver_param.base.n_hor, 1), all_u)
        check_shape("v_ref", (self.solver_param.base.n_hor, 1), v_ref)

        all_v = all_u[0::2]
        diff = all_v-v_ref

        v_upper = cs.fmax(0, diff)
        v_lower = cs.fmin(0,diff)
        if not individual_costs: 
            cost = cs.mtimes(v_upper.T, v_upper)*q_upper 
            cost += cs.mtimes(v_lower.T, v_lower)*q_lower 
        else:
            cost = v_upper**2*q_upper 
            cost += v_lower**2*q_lower #TODO Make sure they're added and not appended

        return cost

    def cost_ang_vel_ref_difference(self, all_u, ang_vel_ref, q, individual_costs=False):
        all_ang_vel = all_u[1::2]
        diff = all_ang_vel-ang_vel_ref

        if not individual_costs:
            cost = cs.sum1(diff**2)*q
        else:
            cost = diff**2*q

        return cost

    def angle_between_2_lines(self, l1, l2, normalized=False):
        """Calculates the angle between two lines

        Args:
            l1 (np.array): array of the 2 positions which make up the first line. [[x0 x1], [y0 y1]]
            l2 ([type]): [description]
        Returns:
            [type]: [description]
        """
        check_shape("l1", (2,2), l1)
        check_shape("l2", (2,2), l2)
        epsilon = 1e-10

        l1_vec = l1[:,1] - l1[:,0]
        l2_vec = l2[:,1] - l2[:,0]

        if not normalized:
            cos_angle = cs.dot(l1_vec, l2_vec) / (cs.norm_2(l1_vec)*cs.norm_2(l2_vec) + epsilon)
        else:
            cos_angle = cs.dot(l1_vec, l2_vec)
            cos_angle = cs.fmin(cos_angle, 1.0-epsilon)
            cos_angle = cs.fmax(cos_angle, -1.0+epsilon)
        angle = cs.acos(cos_angle)

        sign = cs.sign(l2_vec[0]*l1_vec[1] - l2_vec[1]*l1_vec[0])
        angle *= sign

        return angle

    def calc_formation_angle_error(self, x_m, y_m, x_s, y_s, angle_ref):
        p_m = cs.vertcat(x_m,y_m)
        p_s = cs.vertcat(x_s,y_s)

        # Calculate the angle error
        master_slave_line = cs.horzcat(p_m, p_s)
        wanted_line = cs.horzcat(cs.vertcat(0.0,0.0), cs.vertcat(cs.cos(angle_ref), cs.sin(angle_ref))) #TODO: Maybe pass this as a parameter instead of angle_ref
        
        angle_error = self.angle_between_2_lines(wanted_line, master_slave_line, normalized=False)
        return angle_error 

    @cost_positive
    def cost_formation(self, x_m, y_m, x_s, y_s, ref_angle, q, individual_costs=False):
        check_shape("X_m", cs.vertcat(-1,1), x_m)
        check_shape("Y_m", cs.vertcat(-1,1), y_m)
        
        if not individual_costs:
            cost = 0
        else:
            cost = []

        for t in range(self.solver_param.base.n_hor):
            if not individual_costs:
                cost += q * self.calc_formation_angle_error(x_m[t], y_m[t], x_s[t], y_s[t], ref_angle)**2
            else:
                cost.append(q * self.calc_formation_angle_error(x_m[t], y_m[t], x_s[t], y_s[t], ref_angle)**2)
        return cost
    
    def outside_bounds(self, x, y, a0 , a1, b):
        outside = 0 
        for i in range(self.solver_param.base.n_bounds_vertices):
            a0_i = a0[i]
            a1_i = a1[i]
            b_i  = b[i]

            outside_this_line = cs.fmin(0.0, b_i - a0_i*x - a1_i*y)**2.0
            outside += outside_this_line
            #print(outside_this_line)
        
        return outside

    @cost_positive
    def cost_outside_bounds(self, x_all, y_all, q, individual_costs=False):
        if not individual_costs:
            cost = 0
        else:
            cost = []

        for t in range(self.solver_param.base.n_hor):
            if not individual_costs:
                cost += self.outside_bounds(x_all[t], y_all[t], self.a0_bounds, self.a1_bounds, self.b0_bounds)* q
            else:
                cost.append(self.outside_bounds(x_all[t], y_all[t], self.a0_bounds, self.a1_bounds, self.b0_bounds)* q)
            
        return cost 

    def calc_formation_angle_ref(self, x, vertix0, vertix1, vertix2):
        cur_line_ang = cs.arctan2(vertix1[1]-vertix0[1], vertix1[0]-vertix0[0])
        next_line_ang = cs.arctan2(vertix2[1]-vertix1[1], vertix2[0]-vertix1[0])
        if abs(next_line_ang-cur_line_ang) > np.pi:
            cur_line_ang += 2*np.pi if cur_line_ang < 0 else -2*np.pi
        cur_line_ang_wanted = cur_line_ang - np.pi
        next_line_ang_wanted = next_line_ang - np.pi
        # Calculate angle ref
        dist_to_next_line = self.dist_to_line(x[:2], vertix1, vertix2).__float__()
        cur_line_length = self.solver_param.base.dist_to_start_turn 
        positive_factor = min(1, dist_to_next_line / cur_line_length)
        angle_ref = cur_line_ang_wanted*(positive_factor) + next_line_ang_wanted*(1-positive_factor)
        if abs(angle_ref) > np.pi:
            angle_ref += 2*np.pi if angle_ref < 0 else -2*np.pi
        return angle_ref

    def calc_master_ref_angle(self, x, vertix0, vertix1, vertix2):
        dist_to_next_line = self.dist_to_line(x[:2], vertix1, vertix2).__float__()
        cur_line_length = self.solver_param.base.dist_to_start_turn 
        positive_factor = min(1, dist_to_next_line / cur_line_length)

        # This isn't a good solution either. But putting the ref angle master should keep on the end of line_vertices
        cur_line_ang = cs.arctan2(vertix1[1]-vertix0[1], vertix1[0]-vertix0[0])
        next_line_ang = cs.arctan2(vertix2[1]-vertix1[1], vertix2[0]-vertix1[0])
        if abs(next_line_ang-cur_line_ang) > np.pi:
            cur_line_ang += 2*np.pi if cur_line_ang < 0 else -2*np.pi

        master_theta_ref = cur_line_ang*(positive_factor) + next_line_ang*(1-positive_factor)
        if master_theta_ref > np.pi:
            master_theta_ref -= 2*np.pi

        if abs(master_theta_ref - x[2]) > np.pi:
            if x[2]<=0 : master_theta_ref-=2*np.pi
            elif x[2]>0 : master_theta_ref+=2*np.pi
            else: traceback.print_exc(); raise Exception("This shouldn't happen")
        
        return master_theta_ref

    # constraints
    def acc_lagrangian_constraints(self, lin_acc_master, ang_acc_master, lin_acc_slave, ang_acc_slave):
        f = cs.vertcat(lin_acc_master, ang_acc_master, lin_acc_slave, ang_acc_slave)
        amin = [self.solver_param.base.lin_acc_min, -self.solver_param.base.ang_acc_max]*self.solver_param.base.n_hor * 2
        amax = [self.solver_param.base.lin_acc_max, self.solver_param.base.ang_acc_max]*self.solver_param.base.n_hor * 2
        c = og.constraints.Rectangle(amin, amax)
        
        return f, c

    def dist_lagrangian_constraint(self, x_master, y_master, x_slave, y_slave):
        f = cs.SX.ones(0)
        for t in range(0, self.solver_param.base.n_hor):
            x_m = x_master[t]
            y_m = y_master[t]
            x_s = x_slave[t]
            y_s = y_slave[t]
            dist_lb = (-((x_m-x_s)**2 + (y_m-y_s)**2) + self.d_lb**2 ) *self.enable_distance_constraint
            dist_ub = (((x_m-x_s)**2 + (y_m-y_s)**2) - self.d_ub**2 ) *self.enable_distance_constraint #TODO Rename enable_distance_constraint
            f = cs.vertcat(f, dist_lb, dist_ub)

        dist_min = [-cs.inf] * 2 *self.solver_param.base.n_hor # Casadi cannot use parameters as Rectangle constraints / bounds
        dist_max = [0] * 2 * self.solver_param.base.n_hor
        c = og.constraints.Rectangle(dist_min, dist_max)
        return f, c

    def store_params(self, parameters=None):
        """Stores all the parameters required for building the solver and calculating all 
        the costs, states, velocites etc. Z0 can either be None or a list of all the required 
        params. If it's None Z0 is set to a casadi symbolic lists, for building the solver.
        Args:
            self.parameters ([type], optional): [description]. 
        """

        # Build parametric optimizer
        # ------------------------------------
        # Create index numbers
        self.n_static_obs_parameters = sum(range(self.solver_param.base.min_vertices, self.solver_param.base.max_vertices+1)) * self.solver_param.base.n_param_line * self.solver_param.base.n_obs_of_each_vertices
        self.n_dyn_obs_parameters = self.solver_param.base.n_param_dyn_obs * self.solver_param.base.n_hor * self.solver_param.base.n_dyn_obs
        self.n_distance_parameters = 5 # Excluding slave offset
        self.base_ref_u =  2 * self.solver_param.base.nx + 2 * self.solver_param.base.nu + self.n_distance_parameters + self.solver_param.pos_goal_weights.n_weights
        # Index where reference points start
        self.base_ref_points = self.base_ref_u + self.solver_param.base.n_hor*self.solver_param.base.nu
        self.base_static_obs_param = self.base_ref_points + self.solver_param.base.nx * self.solver_param.base.n_hor
        self.base_dyn_obs_param = self.base_static_obs_param + self.n_static_obs_parameters
        self.n_z0 = 2*self.solver_param.base.nx + 2 * self.solver_param.base.nu + self.solver_param.base.n_hor*self.solver_param.base.nu + self.n_distance_parameters + self.solver_param.pos_goal_weights.n_weights + self.solver_param.base.nx * self.solver_param.base.n_hor + self.n_static_obs_parameters + self.n_dyn_obs_parameters + self.solver_param.base.n_dyn_obs + self.solver_param.base.n_param_line * self.solver_param.base.n_bounds_vertices

        self.parameters = parameters
        # Define variables for solver
        if self.parameters is None:
            self.parameters = cs.SX.sym('parameters', self.n_z0) 
        else:
            if not len(self.parameters) == self.n_z0:
                raise Exception(f"{self.print_name} The number of params sent to build_params has to be: {self.n_z0}. It was: {len(self.parameters)}.")
        
        # Position Parameters
        self.x_master, self.y_master, self.theta_master = self.parameters[0], self.parameters[1], self.parameters[2]
        self.x_slave, self.y_slave, self.theta_slave=  self.parameters[3], self.parameters[4], self.parameters[5]
        self.x_ref_master, self.y_ref_master, self.theta_ref_master = self.parameters[6], self.parameters[7], self.parameters[8]
        self.x_ref_slave, self.y_ref_slave, self.theta_ref_slave = self.parameters[9], self.parameters[10], self.parameters[11]
        # Init parameters
        self.vel_init, self.ang_init, self.vel_init_slave, self.ang_init_slave = self.parameters[12], self.parameters[13], self.parameters[14], self.parameters[15]
        self.lin_acc_init_master, self.ang_acc_init_master, self.lin_acc_init_slave, self.ang_acc_init_slave =  self.parameters[16], self.parameters[17], self.parameters[18], self.parameters[19]
        # Formation parameters
        self.d, self.d_lb, self.d_ub, self.formation_angle, self.formation_angle_tol = self.parameters[20], self.parameters[21], self.parameters[22], self.parameters[23], self.parameters[24]
        # Weight parameters
        self.q_lin_v, self.q_lin_acc, self.q_lin_ret, self.q_lin_jerk, self.q_ang, self.q_ang_acc, self.q_ang_jerk, self.q_cte = self.parameters[25], self.parameters[26], self.parameters[27], self.parameters[28], self.parameters[29], self.parameters[30], self.parameters[31], self.parameters[32]
        self.q_pos_master, self.q_pos_slave, self.q_d_lin_vel_upper, self.q_d_lin_vel_lower, self.q_reverse, self.q_d_ang_vel, self.q_theta_master, self.q_theta_slave, self.q_pos_N, self.q_theta_N, self.q_distance, self.q_theta_diff = self.parameters[33], self.parameters[34], self.parameters[35], self.parameters[36], self.parameters[37], self.parameters[38], self.parameters[39], self.parameters[40], self.parameters[41], self.parameters[42], self.parameters[43], self.parameters[44]
        self.q_distance_c, self.q_obs_c, self.q_dyn_obs_c, self.q_future_dyn_obs, self.q_acc_c = self.parameters[45], self.parameters[46], self.parameters[47], self.parameters[48], self.parameters[49]
        self.enable_distance_constraint = self.parameters[50]
        self.q_formation_ang = self.parameters[51]
        self.q_line_theta = self.parameters[52]

        # u_ref  
        self.v_ref_master        = self.parameters[self.base_ref_u    : self.base_ref_points : self.solver_param.base.nu]
        self.ang_vel_ref_master    = self.parameters[self.base_ref_u + 1: self.base_ref_points : self.solver_param.base.nu]
        self.v_ref_slave         = self.parameters[self.base_ref_u + 2: self.base_ref_points : self.solver_param.base.nu]
        self.ang_vel_ref_slave     = self.parameters[self.base_ref_u + 3: self.base_ref_points : self.solver_param.base.nu]

        # SX list of statics objects coords, self.length = Maximum num of objects 
        self.ref_points_master_x     = self.parameters[self.base_ref_points : self.base_static_obs_param : self.solver_param.base.nx]
        self.ref_points_master_y     = self.parameters[self.base_ref_points +1: self.base_static_obs_param : self.solver_param.base.nx]
        self.ref_points_master_th    = self.parameters[self.base_ref_points +2: self.base_static_obs_param : self.solver_param.base.nx]
        self.ref_points_slave_x      = self.parameters[self.base_ref_points +3: self.base_static_obs_param : self.solver_param.base.nx]
        self.ref_points_slave_y      = self.parameters[self.base_ref_points +4: self.base_static_obs_param : self.solver_param.base.nx]
        self.ref_points_slave_th     = self.parameters[self.base_ref_points +5: self.base_static_obs_param : self.solver_param.base.nx]

        self.bs_static = self.parameters[self.base_static_obs_param     :self.base_dyn_obs_param :self.solver_param.base.n_param_line]
        self.a0s_static = self.parameters[self.base_static_obs_param + 1:self.base_dyn_obs_param :self.solver_param.base.n_param_line]
        self.a1s_static = self.parameters[self.base_static_obs_param + 2:self.base_dyn_obs_param :self.solver_param.base.n_param_line]

        # ordering is x_master, y_master, r for obstacle 0 for n_hor timesteps, self.then x_master, y_master, r for obstalce 1 for n_hor timesteps etc.
        # Get dynamic obstacle parameters for each timestep 
        self.end_of_dynamic_obs_idx = self.base_dyn_obs_param + self.solver_param.base.n_param_dyn_obs * self.solver_param.base.n_dyn_obs * self.solver_param.base.n_hor
        self.a0_dyn = self.parameters[self.base_dyn_obs_param    : self.end_of_dynamic_obs_idx: self.solver_param.base.n_param_dyn_obs]
        self.a1_dyn = self.parameters[self.base_dyn_obs_param + 1: self.end_of_dynamic_obs_idx: self.solver_param.base.n_param_dyn_obs]
        self.a2_dyn = self.parameters[self.base_dyn_obs_param + 2: self.end_of_dynamic_obs_idx: self.solver_param.base.n_param_dyn_obs]
        self.a3_dyn = self.parameters[self.base_dyn_obs_param + 3: self.end_of_dynamic_obs_idx: self.solver_param.base.n_param_dyn_obs]
        self.c0_dyn = self.parameters[self.base_dyn_obs_param + 4: self.end_of_dynamic_obs_idx: self.solver_param.base.n_param_dyn_obs]
        self.c1_dyn = self.parameters[self.base_dyn_obs_param + 5: self.end_of_dynamic_obs_idx: self.solver_param.base.n_param_dyn_obs]
        self.active_dyn_obs = self.parameters[self.end_of_dynamic_obs_idx : self.end_of_dynamic_obs_idx + self.solver_param.base.n_dyn_obs]

        self.base_bounds_param = self.end_of_dynamic_obs_idx + self.solver_param.base.n_dyn_obs
        # set up bounds ineqs into lists
        self.b0_bounds = self.parameters[self.base_bounds_param    : self.base_bounds_param + self.solver_param.base.n_param_line * self.solver_param.base.n_bounds_vertices : self.solver_param.base.n_param_line]
        self.a0_bounds = self.parameters[self.base_bounds_param + 1: self.base_bounds_param + self.solver_param.base.n_param_line * self.solver_param.base.n_bounds_vertices : self.solver_param.base.n_param_line]
        self.a1_bounds = self.parameters[self.base_bounds_param + 2: self.base_bounds_param + self.solver_param.base.n_param_line * self.solver_param.base.n_bounds_vertices : self.solver_param.base.n_param_line]

    def build(self):
        # Init costs
        penalty_c = 0.0 
        cost = 0        
        self.store_params()

        state_vec = cs.vertcat(self.x_master, self.y_master, self.theta_master, self.x_slave, self.y_slave, self.theta_slave)
        u = cs.SX.sym('u', self.solver_param.base.nu*self.solver_param.base.n_hor)
        all_states = cs.SX(self.solver_param.base.n_hor+1, self.solver_param.base.nx)
        all_states[0,:] = state_vec

        for t in range(0, self.solver_param.base.n_hor): # LOOP OVER TIME STEPS
            u_t = u[t*self.solver_param.base.nu: (t+1)*self.solver_param.base.nu]
            state_vec = self.dynamics_rk4_dual(state_vec, u_t)
            all_states[t+1, :] = state_vec
            

        all_positions_master = all_states[:, 0:2]
        all_positions_slave = all_states[:, 3:5]
        all_x_master = all_states[:, 0]
        all_y_master = all_states[:, 1]
        all_th_master = all_states[:, 2]
        all_x_slave = all_states[:, 3]
        all_y_slave = all_states[:, 4]
        all_th_slave = all_states[:, 5]

        # Cost for constraints
        cost += self.cost_distance_between_atrs(all_x_master[1:], all_y_master[1:], all_x_slave[1:], all_y_slave[1:], self.q_distance)
        cost += self.cost_distance_atr_soft_c(all_x_master[1:], all_y_master[1:], all_x_slave[1:], all_y_slave[1:], self.q_distance_c)
        # f1, c1 = self.dist_lagrangian_constraint(all_x_master[1:], all_y_master[1:], all_x_slave[1:], all_y_slave[1:])

        # Cost for obstacles
        cost += self.cost_inside_static_object(all_x_master[1:], all_y_master[1:], self.q_obs_c)
        cost += self.cost_inside_static_object(all_x_slave[1:], all_y_slave[1:], self.q_obs_c)
        cost += self.cost_inside_dyn_ellipse2(all_x_master[1:], all_y_master[1:], self.q_dyn_obs_c)
        cost += self.cost_inside_dyn_ellipse2(all_x_slave[1:], all_y_slave[1:], self.q_dyn_obs_c)
        #cost += self.cost_inside_future_dyn_ellipse(all_x_master[1:], all_y_master[1:], self.q_future_dyn_obs)
        #cost += self.cost_inside_future_dyn_ellipse(all_x_slave[1:], all_y_slave[1:], self.q_future_dyn_obs)

        # Cost for outside bounds
        cost += self.cost_outside_bounds(all_x_master[1:], all_y_master[1:], self.q_obs_c)
        cost += self.cost_outside_bounds(all_x_slave[1:], all_y_slave[1:], self.q_obs_c)
        # Cost for ideal states
        cost += self.cost_dist2ref_points(all_x_master[1:], all_y_master[1:], all_th_master[1:],  self.ref_points_master_x, self.ref_points_master_y, self.ref_points_master_th, self.q_pos_master, self.q_theta_master)
        cost += self.cost_dist2ref_points(all_x_slave[1:] , all_y_slave[1:] , all_th_slave[1:] ,  self.ref_points_slave_x , self.ref_points_slave_y , self.ref_points_slave_th , self.q_pos_slave , self.q_theta_slave)
        # cost += self.cost_formation(all_x_master, all_y_master, all_x_slave, all_y_slave, self.formation_angle, self.q_formation_ang)
        # Last point weigt (ignored for now)
        ##################
        cost += self.cost_dist2ref_line(all_x_master[1:], all_y_master[1:], all_th_master[1:], self.ref_points_master_x, self.ref_points_master_y, self.q_cte, self.q_line_theta)
        # cost += self.cost_dist2ref_line(all_x_slave, all_y_slave, all_th_slave, self.ref_points_slave_x, self.ref_points_slave_y, self.q_cte*self.enable_distance_constraint, self.q_theta*self.enable_distance_constraint*0)
        
        # Max speeds 
        umin = [self.solver_param.base.lin_vel_min, -self.solver_param.base.ang_vel_max, self.solver_param.base.lin_vel_min, -self.solver_param.base.ang_vel_max] * self.solver_param.base.n_hor
        umax = [self.solver_param.base.lin_vel_max, self.solver_param.base.ang_vel_max, self.solver_param.base.lin_vel_max, self.solver_param.base.ang_vel_max] * self.solver_param.base.n_hor
        bounds = og.constraints.Rectangle(umin, umax)

        # Control/velocity costs
        master_u_index =  []
        slave_u_index = []
        for t in range(0, self.solver_param.base.n_hor):
           master_u_index.append(0 + t*self.solver_param.base.nu)
           master_u_index.append(1 + t*self.solver_param.base.nu)
           slave_u_index.append(2 + t*self.solver_param.base.nu)
           slave_u_index.append(3 + t*self.solver_param.base.nu)
        
        master_u = u[master_u_index]
        slave_u = u[slave_u_index]
        lin_cost, ang_cost = self.cost_control(master_u, self.q_lin_v, self.q_ang) 
        cost += lin_cost + ang_cost


        lin_cost, ang_cost = self.cost_control(slave_u, self.q_lin_v, self.q_ang) 
        cost += lin_cost + ang_cost

        # Cost for reference velocity
        cost += self.cost_v_ref_difference(master_u, self.v_ref_master, self.q_d_lin_vel_upper, self.q_d_lin_vel_lower )
        cost += self.cost_ang_vel_ref_difference(master_u, self.ang_vel_ref_master, self.q_d_ang_vel)
        # Accelerations cost
        lin_acc_master, ang_acc_master = self.calc_accelerations(master_u, self.vel_init, self.ang_init)
        cost += self.cost_linear_acc(lin_acc_master, self.q_lin_acc, self.q_lin_ret)
        cost += self.cost_angular_acc(ang_acc_master, self.q_ang_acc)

        lin_acc_slave, ang_acc_slave = self.calc_accelerations(slave_u, self.vel_init_slave, self.ang_init_slave)
        cost += self.cost_linear_acc(lin_acc_slave, self.q_lin_acc, self.q_lin_ret)
        cost += self.cost_angular_acc(ang_acc_slave, self.q_ang_acc)

        problem = og.builder.Problem(u, self.parameters, cost)
        problem.with_constraints(bounds)
        # problem.with_aug_lagrangian_constraints(f1, c1)
        # problem.with_penalty_constraints(penalty_c)
        build_config = og.config.BuildConfiguration()\
            .with_build_directory(self.solver_param.base.build_directory)\
            .with_build_mode(self.solver_param.base.build_type)
        build_config.with_build_python_bindings()

        meta = og.config.OptimizerMeta()\
            .with_optimizer_name(self.solver_param.base.optimizer_name)

        solver_config = og.config.SolverConfiguration()
        solver_config.with_tolerance(1e-4)
        solver_config.with_max_duration_micros(MAX_SOLVER_TIME_MICROS)
        solver_config.with_max_outer_iterations(MAX_OUTER_ITERATIONS)
        # solver_config.with_initial_penalty(1000)
        #solver_config.with_penalty_weight_update_factor(2)
        # solver_config.with_delta_tolerance(1e-3)
        # solver_config.with_lbfgs_memory(50)

        builder = og.builder.OpEnOptimizerBuilder(problem, 
                                                meta, 
                                                build_config, 
                                                solver_config).with_verbosity_level(1)
        builder.build()



if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.__str__()+'/') # This allows solver_paramurator to be imported
    from utils.solver_param.base import solver_paramurator

    solver_param_fn = 'default.yaml'
    yaml_fp = os.path.join(str(Path(__file__).parent.parent.parent), 'solver_params', solver_param_fn)
    solver_paramurator = solver_paramurator(yaml_fp)
    solver_param.base = solver_paramurator.solver_paramurate()
    mpc_generator = MpcModule(solver_param.base)
    x_m = np.array([0]*solver_param.base.n_hor )
    y_m = np.array([1]*solver_param.base.n_hor )
    th_m = np.array([cs.pi/2] *solver_param.base.n_hor)
    x_s = np.array([0]*solver_param.base.n_hor )
    y_s = np.array([1]*solver_param.base.n_hor )
    rx =  np.array([0]*solver_param.base.n_hor )
    ry =  np.array([0] + [2]*(solver_param.base.n_hor -1))
    ans1 = mpc_generator.cost_dist2ref_line(x_m,y_m,th_m,rx, ry,1,1)
    print(ans1)

    # Check if finding closest line works:
    line_x = np.array([0, 1, 0, 1])
    line_y = np.array([0, 0, 1, 1])
    pos = np.array([0.5, 0])
    closest_line, closest_line_distance = mpc_generator.get_closest_line(pos, line_x, line_y)
    closest_line = [float(closest_line[i]) for i in range(closest_line.shape[0])]
    expected_line = np.array([0, 0, 1, 0])
    expected_distance = 0
    assert np.all(closest_line == expected_line), f"The closest line should be: {expected_line}, but is: {closest_line}"
    assert closest_line_distance == expected_distance, f"The closest line  distanceshould be: {expected_distance}, but is: {closest_line}"

    # Test calc master_line_ang
    # Test 1
    xm = np.array([2,2])
    xs = np.array([1,1])
    s0 = np.array([2,0])
    s1 = np.array([2,2])
    expected_angle = -np.pi/4
    angle = mpc_generator.calc_master_slave_ang(xm, xs, cs.arctan2(s1[1] - s0[1], s1[0] - s0[0]))
    assert expected_angle - angle < 1e-3 , f"The returned angle should be {expected_angle} but is: {angle}"
    # Test 2
    xm = np.array([0,1])
    xs = np.array([0,0])
    s0 = np.array([0,0])
    s1 = np.array([0,1])
    expected_angle = 0
    angle = mpc_generator.calc_master_slave_ang(xm, xs, cs.arctan2(s1[1] - s0[1], s1[0] - s0[0]))
    assert expected_angle - angle < 1e-3 , f"The returned angle should be {expected_angle} but is: {angle}"
    # Test 3
    xm = np.array([0,0])
    xs = np.array([0,1])
    s0 = np.array([0,0])
    s1 = np.array([0,1])
    expected_angle = np.pi
    angle = mpc_generator.calc_master_slave_ang(xm, xs, cs.arctan2(s1[1] - s0[1], s1[0] - s0[0]))
    assert expected_angle - np.abs(angle) < 1e-3 , f"The returned angle should be {expected_angle} but is: {angle}"
    # Test 4
    xm = np.array([-1,-2])
    xs = np.array([5,1])
    s0 = np.array([-1,-1])
    s1 = np.array([-1,-2])
    angle = mpc_generator.calc_master_slave_ang(xm, xs, cs.arctan2(s1[1] - s0[1], s1[0] - s0[0]))
    assert angle < 0 and angle > -np.pi/2

    # Line right, slave south 
    # -m->
    #  s
    x_m = np.array([0])
    y_m = np.array([0])
    x_s = np.array([0])
    y_s = np.array([-1])
    s0 = np.array([0,0])
    s1 = np.array([1,0])
    s2 = np.array([2,0])
    angle = mpc_generator.formation_angle(x_m, y_m, x_s, y_s, s0, s1, s2)
    expected_angle = -cs.pi/2
    assert angle == expected_angle , f"The returned angle should be {expected_angle} but is: {angle}"

    # Line up slave notheast
    #   ^
    #   | s
    #   m
    x_m = np.array([0])
    y_m = np.array([0])
    x_s = np.array([1])
    y_s = np.array([1])
    s0 = np.array([0,0])
    s1 = np.array([0,1])
    s2 = np.array([0,2])
    angle = mpc_generator.formation_angle(x_m, y_m, x_s, y_s, s0, s1, s2)
    expected_angle = -3*cs.pi/4
    assert angle == expected_angle , f"The returned angle should be {expected_angle} but is: {angle}"
    
    # Line up slave up
    #   s
    #   ^
    #   |
    #   m
    x_m = np.array([0])
    y_m = np.array([0])
    x_s = np.array([0])
    y_s = np.array([1])
    s0 = np.array([0,0])
    s1 = np.array([0,1])
    s2 = np.array([0,2])
    angle = mpc_generator.formation_angle(x_m, y_m, x_s, y_s, s0, s1, s2)
    expected_angle = cs.pi
    assert abs(float(angle)) == expected_angle , f"The returned angle should be {expected_angle} but is: {angle}"

    # Line up slave northwest
    #   ^
    #  s| 
    #   m
    x_m = np.array([0])
    y_m = np.array([0])
    x_s = np.array([-1])
    y_s = np.array([1])
    s0 = np.array([0,0])
    s1 = np.array([0,1])
    s2 = np.array([0,2])
    angle = mpc_generator.formation_angle(x_m, y_m, x_s, y_s, s0, s1, s2)
    expected_angle = 3*cs.pi/4
    assert angle == expected_angle , f"The returned angle should be {expected_angle} but is: {angle}"

    # Line up slave northwest
    #   ^
    #   | 
    #   m
    #  s
    x_m = np.array([0])
    y_m = np.array([0])
    x_s = np.array([-1])
    y_s = np.array([-1])
    s0 = np.array([0,0])
    s1 = np.array([0,1])
    s2 = np.array([0,2])
    angle = mpc_generator.formation_angle(x_m, y_m, x_s, y_s, s0, s1, s2)
    expected_angle = cs.pi/4
    assert np.abs(angle - expected_angle) < 1e-5  , f"The returned angle should be {expected_angle} but is: {angle}"
    
    # Line up slave northwest
    #   ^
    #   | 
    #   m
    #     s
    x_m = np.array([0])
    y_m = np.array([0])
    x_s = np.array([1])
    y_s = np.array([-1])
    s0 = np.array([0,0])
    s1 = np.array([0,1])
    s2 = np.array([0,2])
    angle = mpc_generator.formation_angle(x_m, y_m, x_s, y_s, s0, s1, s2)
    expected_angle = -cs.pi/4
    assert np.abs(angle - expected_angle) < 1e-5, f"The returned angle should be {expected_angle} but is: {angle}"
    
    # Check if angle punishment works
    mpc_generator.solver_param.base.n_hor = 2
    cost = mpc_generator.cost_dist2ref_line(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 1]), 0, 1)
    expected_cost = (cs.pi/2)**2*2
    assert cost == expected_cost, f"The returned cost should be {expected_cost} but is: {cost}"
    # line up, slave behind master
    # |
    # m
    # s
    x_m = np.array([0,0])
    y_m = np.array([0,0])
    th_m = np.array([0,0])
    x_s = np.array([0,0])
    y_s = np.array([-1,-1])
    ref_x = np.array([0,0,0])
    ref_y = np.array([0,1,1])
    cost = mpc_generator.cost_formation(x_m,y_m,x_s,y_s,ref_x,ref_y, 1)
    expected_cost = (0)**2*2
    assert cs.fabs(cost) <= expected_cost + 0.0001, f"The returned cost should be {expected_cost} but is: {cost}"
    
    # line up slave to right 
    #   ^
    #   |
    # <-m-s
    x_m = np.array([0,0]).reshape(-1,1)
    y_m = np.array([0,0]).reshape(-1,1)
    th_m = np.array([0,0]).reshape(-1,1)
    x_s = np.array([1,1]).reshape(-1,1)
    y_s = np.array([0,0]).reshape(-1,1)
    ref_x = np.array([0,0,0])
    ref_y = np.array([0,1,1])
    cost = mpc_generator.cost_formation(x_m,y_m,x_s,y_s,ref_x,ref_y, 1)
    expected_cost = (cs.pi/2)**2*2
    assert cost == expected_cost, f"The returned cost should be {expected_cost} but is: {cost}"

    # line up slave to right 
    #   ^
    #   |
    #   s
    #   m
    #   v
    x_m = np.array([0,0]).reshape(-1,1)
    y_m = np.array([-1,-1]).reshape(-1,1)
    th_m = np.array([0,0]).reshape(-1,1)
    x_s = np.array([0,0]).reshape(-1,1)
    y_s = np.array([0,0]).reshape(-1,1)
    ref_x = np.array([0,0,0])
    ref_y = np.array([0,1,1])
    cost = mpc_generator.cost_formation(x_m,y_m,x_s,y_s,ref_x,ref_y, 1)
    expected_cost = (cs.pi)**2*2
    assert cs.fabs(cost) <= expected_cost + 0.0001, f"The returned cost should be {expected_cost} but is: {cost}"    
    
    # distance soft constraint
    all_x_master = np.array([0]*solver_param.base.n_hor) 
    all_y_master = np.array([0,3]*solver_param.base.n_hor) 
    all_x_slave = np.array([0]*solver_param.base.n_hor) 
    all_y_slave = np.array([1,2]*solver_param.base.n_hor) 
    q_distance_c = 1
    mpc_generator.d = 1
    mpc_generator.d_tol_lb = -0.05
    mpc_generator.d_tol_ub = 0.05
    cost = mpc_generator.cost_distance_atr_soft_c(all_x_master, all_y_master, all_x_slave, all_y_slave, q_distance_c)
    expected_cost = 0 
    assert cs.fabs(cost) <= expected_cost + 0.00001 and cs.fabs(cost) >= expected_cost -0.00001, f"The returned cost should be {expected_cost} but is: {cost}"
    
    print("mpc_generator passed all tests!")