from queue import Full
import time
from path_planner.path_planner import PathPlanner
from mpc.mpc_generator import MpcModule, get_length
import numpy as np
import math
from ATRS import ATRS
from time import perf_counter_ns
import traceback
import shapely
from enum import Enum
from utils.config import SolverParams
import sys
from pathlib import Path



class States(Enum):
    COUPLING = 0
    FORMATION = 1
    DECOUPLING = 2
    DONE = 3

    def transition(self, x, x_ref, atol=0.05, rtol=0.00001):
        return np.allclose(x, x_ref, atol=atol, rtol=rtol) 

    def next(self):
        if self == States.COUPLING:
            return States.FORMATION
        elif self == States.FORMATION:
            return States.DECOUPLING
        elif self == States.DECOUPLING:
            return States.DONE
        elif self == States.DONE:
            return None
        else:
            traceback.print_exc()
            raise(Exception("This state doesn't exist!"))

class RollingAvg:

    def __init__(self, steps = 10) :
        from collections import deque
        self.steps = steps
        self.past_values = deque(maxlen=int(steps))

    def get_avg(self):
        rolling_avg = np.sum(self.past_values)/self.steps
        return rolling_avg

    def update(self, x):
        self.past_values.append(np.sum(x))

class TrajectoryGenerator:
    """Class that generates control inputs. Uses solver_paramuration file in solver_params folder
    """
    def __init__(self, plot_config, plot_queues, plot_process, solver_param:SolverParams, map_data, robot_data, verbose=False, name="[TrajGen]", self_destruct=True):
        self.plot_config = plot_config
        self.verbose = verbose 
        self.plot_queues = plot_queues
        self.plot_process = plot_process
        self.solver_param = solver_param
        self.robot_data = robot_data
        self.self_destruct = self_destruct

        self.path_planner = PathPlanner(solver_param.base, plot_config, plot_queues, map_data) #TODO: This is really slow. Why?
        self.bounds_equations = None # These are set in the map_data setter
        self.map_data = map_data
        self.bounds_equations = self.convert_bounds_to_eqs()

        self.mpc_generator = MpcModule(self.solver_param)
        self.times = {'loop_time':[], 'solver':[], 'static_to_eq':[], 'dyn_to_eq':[], 'plot':[]}
        self.solver_times = []
        self.overhead_times = []
        self.print_name = name
        self.costs = {key : [] for key in self.plot_queues if 'cost' in key and not 'future' in key}
        self.costs_future = {key : [] for key in self.plot_queues if 'cost' in key and 'future' in key}
        self.initial_guess_master = None
        self.initial_guess_dual = None
        self.initial_guess_pos = None
        self.u_previous = None
        self.state = States.COUPLING

        sys.path.append(Path(__file__).parent.parent.__str__() + '/mpc_build/' + self.solver_param.base.optimizer_name+'/') # This allows solver to be imported
        import trajectory_generator_solver
        self.solver = trajectory_generator_solver.solver()

        # Get the obstacles for the obstacle handler
        self.path_planner.update_obstacles()
        self.ra = RollingAvg(1e2) # TODO : fulfix 
        # Generate a global map
        self.path_planner.global_path_plan(self.map_data.positions_master.coupling_end[:2], self.map_data.positions_master.formation_end[:2])

        self.master_end = list(self.map_data.positions_master.next())
        self.slave_end = list(self.map_data.positions_slave.next())


        
    def generate_control_input(self, parameters, guess=None):
        """Calls on the nmpc solver mng with parameters to find the next control inputs

        Args:
            mng (npmc solver): nmpc solver fit for the current task
            parameters (list): list of parameters. Specified in default.yaml

        Returns:
            list: control inputs for the atrs, for how many time steps is specififed by num_steps_taken in default.yaml. [v_master, ang_slave, v_slave, oemga_slave ....]
        """
        start_time = perf_counter_ns()
        if guess is None:
            solution = self.solver.run(parameters)
        else:
            solution = self.solver.run(parameters, initial_guess=guess)
        self.times['solver'].append(perf_counter_ns() - start_time)

        exit_status = solution.exit_status
        u = solution.solution
        
        assert solution.cost >= 0, f"The cost cannot be negative. It was {solution.cost}"
        return u, solution.cost
     
    def calculate_dist_between_nodes(self, nodes, cumsum=True):
        assert(nodes.shape[1]==2), f"The nodes array sent to {self.print_name}.calculate_dist_between_nodes must be of shape: (-1, 2). They were in shape: {nodes.shape}."
        block_distance = np.diff(nodes, axis=0)
        euclidian_distance = np.linalg.norm(block_distance, axis=1)
        
        if cumsum: return np.cumsum(euclidian_distance)
        else: return euclidian_distance

    def generate_refs(self, x_cur, ref_points):
        """[summary]

        Args:
            x_cur ([type]): [description]
            ref_points ([type]): [description]

        Returns:
            [best_ref_point]: [Reachable ref point]
            [local_ref_points]: [n_hor points close to current]
        """        
        # Find the best ref point
        dist_to_nodes = np.linalg.norm(np.array(ref_points) - np.array(x_cur[:2]), axis=1)
        closest_idx = np.argmin(dist_to_nodes)

        nodes_forward = ref_points[closest_idx:]
        dist_to_nodes_forward = self.calculate_dist_between_nodes(np.array(nodes_forward))
        dist_to_nodes_forward -= 1 #self.solver_param.base.lin_vel_max*self.solver_param.base.ts*self.solver_param.base.n_hor
        
        ref_point_within_reach = False
        # if all of the nodes are within reach:
        if not np.all(dist_to_nodes_forward[dist_to_nodes_forward > 1e-3].shape):
            best_reference_point_idx = -1
            best_reference_point = list(ref_points[best_reference_point_idx])
            ref_point_within_reach = True

        else:
            best_dist = np.min(dist_to_nodes_forward[dist_to_nodes_forward > 1e-3]) #1e-3 is just an epsilon
            best_reference_point_idx = np.where(dist_to_nodes_forward == best_dist)[0][0] + closest_idx

            best_reference_point = ref_points[best_reference_point_idx]

        best_reference_point = list(best_reference_point) + [0]
            


        # Find the line vertices
        # Find the first vertix on the current line
        cur_node = ref_points[0]
        node_to_check_idx = 1
        calc_ang_between_nodes = lambda p1, p2: np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
        line_vertices = [cur_node]

        # Find the following vertices
        while node_to_check_idx < len(ref_points):
            cur_node = line_vertices[-1]
            ang_between_nodes = calc_ang_between_nodes(cur_node, ref_points[node_to_check_idx])
            
            for node in ref_points[node_to_check_idx:]:
                if ang_between_nodes == calc_ang_between_nodes(cur_node, node):
                    node_to_check_idx += 1
                else:
                    line_vertices.append(ref_points[node_to_check_idx-1])
                    break

        # Add last vertix
        line_vertices.append(ref_points[-1])


        # Calculate closest lines
        lines = [[np.array(line_vertices[i], dtype=float), np.array(line_vertices[i+1], dtype=float)] for i in range(len(line_vertices)-1)]
        dist_to_lines = np.array([self.mpc_generator.dist_to_line(x_cur[:2], *line).__float__() for line in lines])**0.5
        closest_line_idx = np.argmin(dist_to_lines)

        # Find the first vertix on the current line
        line_vertices = [lines[closest_line_idx][0]]
        dist_lines_ahead = np.cumsum(dist_to_lines[closest_line_idx:]) - self.solver_param.base.dist_to_start_turn*100 
        for i in range(len(lines[closest_line_idx:])): 
            if dist_lines_ahead[i] < 0: 
                line_vertices.append(lines[i+closest_line_idx][1]) 


        # Make sure the list of vertices is long enough
        line_vertices += [line_vertices[-1]]*(self.solver_param.base.n_hor - len(line_vertices))

        assert(len(line_vertices) == self.solver_param.base.n_hor), f"There are too many ref points. There can only be {self.solver_param.base.n_hor} and there were {len(line_vertices)}."
                

        return best_reference_point, line_vertices, ref_point_within_reach

    def generate_vel_ref(self, x_cur):
        # TODO : implement this
        vel_ref = 0
        return vel_ref

    def convert_static_obs_to_eqs(self, x_cur):
        """[summary]

        Args:
            x_cur ([type]): [description]
        
        Returns:
            obs_equations (numpy array): 1-d numpy array where the static and unexpected obstacles as [b0 a0^T ... bn an^T]
        
        """
        start_time = perf_counter_ns()

        # Get the closest static and dynamic obstacles
        closest_obs = self.path_planner.obs_handler.get_closest_static_obstacles(x_cur)
        
        # Convert the closest static and dynamic obstacles into inequalities 
        obs_equations = []
        for key in closest_obs:
            obs = closest_obs[key]
            key_obs_equations = np.array(self.path_planner.obs_handler.obstacles_as_inequalities(obs)).reshape(-1)

            # If there aren't enough obstacles, fill with 0s
            n_obs = len(obs)
            if n_obs < self.solver_param.base.n_obs_of_each_vertices: #TODO: Add special case for triangles
                obs_array = np.zeros((self.solver_param.base.n_obs_of_each_vertices-n_obs)*3*key) # CHange 3 to the variable in self.solver_param
                key_obs_equations = np.hstack((key_obs_equations, obs_array))
            obs_equations.append(key_obs_equations)
        
        obs_equations = np.hstack(obs_equations)

        self.times['static_to_eq'].append(perf_counter_ns() - start_time)
        
        # Make sure the correct number of params are sent
        assert( sum(range(self.solver_param.base.min_vertices, self.solver_param.base.max_vertices+1)) *3 * self.solver_param.base.n_obs_of_each_vertices == sum(obs_equations.shape)), "An invalid amount of static obstacles parameters were sent."

        return obs_equations, closest_obs

    def convert_bounds_to_eqs(self):
        bounds_equations = []
        bounds = self.path_planner.obs_handler.boundry
        A, B = self.path_planner.obs_handler.obstacle_as_inequality(np.mat(bounds))
        bounds_array = np.array(np.concatenate((B, A), axis = 1)).reshape(-1)

        #convert to list
        bounds_array = list(bounds_array)
        n_params = len(bounds_array)
        req_param = self.solver_param.base.n_bounds_vertices*self.solver_param.base.n_param_line
        if n_params < req_param:
            extra_params = np.zeros(req_param-n_params)
            bounds_array = bounds_array + list(extra_params)
        return bounds_array

    def convert_dynamic_obs_to_eqs(self, x_cur):
        start_time = perf_counter_ns()
        
        # Get the closest dynamic obst
        closest_obs = self.path_planner.obs_handler.get_closest_dynamic_obstacles(x_cur)
        
        # Convert them into ellipses
        obs_ellipse = []
        for ob in closest_obs:
            obs_ellipse.append(self.path_planner.obs_handler.obstacles_as_ellipses([np.array(ob_.exterior.xy).T[:-1] for ob_ in ob]))
        ellipse_to_plot = obs_ellipse.copy()


        # If there aren't enough obstacles, append 0s
        n_obs = len(obs_ellipse)
        obs_ellipse = np.array(obs_ellipse).reshape(-1)
        if n_obs < self.solver_param.base.n_dyn_obs:
            obs_array = np.zeros((self.solver_param.base.n_dyn_obs - n_obs) * self.solver_param.base.n_param_dyn_obs * self.solver_param.base.n_hor)
            obs_ellipse = np.hstack((obs_ellipse, obs_array))
        

        
        self.times['dyn_to_eq'].append(perf_counter_ns() - start_time)

        # Make sure the correct amount of parameters are returned
        assert(self.solver_param.base.n_param_dyn_obs *self.solver_param.base.n_hor*self.solver_param.base.n_dyn_obs == sum(obs_ellipse.shape)), "An invalid amount of dynamic obstacles parameters were sent."

        return obs_ellipse, closest_obs, ellipse_to_plot

    def get_active_dyn_obs(self, dyn_constraints):
        dyn_constraints_len = len(dyn_constraints)
        already_set = False
        active_obs = [1]*self.solver_param.base.n_dyn_obs
        a0_dyn = dyn_constraints[0: dyn_constraints_len: self.solver_param.base.n_param_dyn_obs]
        a1_dyn = dyn_constraints[1: dyn_constraints_len: self.solver_param.base.n_param_dyn_obs]
        a2_dyn = dyn_constraints[2: dyn_constraints_len: self.solver_param.base.n_param_dyn_obs]
        a3_dyn = dyn_constraints[3: dyn_constraints_len: self.solver_param.base.n_param_dyn_obs]
        c0_dyn = dyn_constraints[4: dyn_constraints_len: self.solver_param.base.n_param_dyn_obs]
        c1_dyn = dyn_constraints[5: dyn_constraints_len: self.solver_param.base.n_param_dyn_obs]
        for t in range(0, self.solver_param.base.n_hor):
            a0_t = a0_dyn[t :: self.solver_param.base.n_hor]
            a1_t = a1_dyn[t :: self.solver_param.base.n_hor]
            a2_t = a2_dyn[t :: self.solver_param.base.n_hor]
            a3_t = a3_dyn[t :: self.solver_param.base.n_hor]
            c0_t = c0_dyn[t :: self.solver_param.base.n_hor]
            c1_t = c1_dyn[t :: self.solver_param.base.n_hor]
            for obs in range(0, self.solver_param.base.n_dyn_obs):

                if a0_t[obs] == 0 and a1_t[obs] == 0 and a2_t[obs] == 0 and a3_t[obs] == 0 and c0_t[obs] == 0 and c1_t[obs] == 0:
                    active_obs[obs] = 0
        
        return active_obs

    def create_circle_sector(self, center, start_angle, end_angle, radius, steps=200):
        """Taken from https://stackoverflow.com/questions/54284984/sectors-representing-and-intersections-in-shapely

        Args:
            center ([type]): [description]
            start_angle ([type]): [description]
            end_angle ([type]): [description]
            radius ([type]): [description]
            steps (int, optional): [description]. Defaults to 200.
        """
        def polar_point(origin_point, angle,  distance):
            return [origin_point.x + math.sin(angle) * distance, origin_point.y + math.cos(angle) * distance]

        if start_angle > end_angle:
            start_angle = start_angle - 2*math.pi
        else:
            pass
        step_angle_width = (end_angle-start_angle) / steps
        sector_width = (end_angle-start_angle) 
        segment_vertices = []

        segment_vertices.append(polar_point(center, 0,0))
        segment_vertices.append(polar_point(center, start_angle,radius))

        for z in range(1, steps):
            segment_vertices.append((polar_point(center, start_angle + z * step_angle_width,radius)))
        segment_vertices.append(polar_point(center, start_angle+sector_width, radius))
        segment_vertices.append(polar_point(center, 0,0))
        return shapely.geometry.Polygon(segment_vertices)

    def check_for_dynamic_obstacles_ahead(self, x):
        center = shapely.geometry.Point(x[1], x[0])
        ang_master = x[2] + 2*np.pi if x[2] < 0 else 0
        ang_start = ang_master + self.solver_param.base.ang_detect_dyn_obs
        ang_end = ang_master - self.solver_param.base.ang_detect_dyn_obs
        circle_sector = self.create_circle_sector(center, ang_start, ang_end, self.solver_param.base.dist_detect_dyn_obs)

        # TODO: Actually check for obstacles ahead 

        return circle_sector

    def gen_line_following_traj(self, x_cur, dyn_constraints, closest_dynamic_obs_poly, line_vertices_master, line_vertices_slave, formation_angle_ref, constraints, active_dyn_obs, distance_lb, distance_ub, aggressive_factor):
        master_angle_ref = self.mpc_generator.calc_master_ref_angle(x_cur[:3], *line_vertices_master[:3])

        # Build parameter list
        acc_init = [self.robot_data.get_prev_acc()[0], self.robot_data.get_prev_acc()[1], 0, 0] # We don't care about the slaves accelerations
        u_ref = [1,0,0,0] * self.solver_param.base.n_hor# * self.solver_param.base.nu #TODO: 1 shouldn't be hardcoded

        x_finish = list(line_vertices_master[-3]) + [master_angle_ref] + list(line_vertices_slave[-3]) + [0]
        # Modify the reference values so that theta_ref = 0 for both master and slave
        refs = []# This is a bad name. What it containts is the line vertices
        for i in range(0, self.solver_param.base.n_hor):
            refs += list(line_vertices_master[i]) + [0] 
            refs += list(line_vertices_slave[i]) + [0]

        # Check if there are dynamic obstacles which might need to be avoided. If so reduce q_cte, q_theta, q_formation
        search_sector = self.check_for_dynamic_obstacles_ahead(x_cur[:3])
        
        # Master only trajectory generation weights
        weight_list_soft = [self.solver_param.line_follow_weights.q_lin_v, 
                    self.solver_param.line_follow_weights.q_lin_acc, 
                    self.solver_param.line_follow_weights.q_lin_ret, 
                    self.solver_param.line_follow_weights.q_lin_jerk, 
                    self.solver_param.line_follow_weights.q_ang, 
                    self.solver_param.line_follow_weights.q_ang_acc, 
                    self.solver_param.line_follow_weights.q_ang_jerk, 
                    self.solver_param.line_follow_weights.q_cte, 
                    self.solver_param.line_follow_weights.q_pos_master, 
                    self.solver_param.line_follow_weights.q_pos_slave, 
                    self.solver_param.line_follow_weights.q_d_lin_vel_upper, 
                    self.solver_param.line_follow_weights.q_d_lin_vel_lower, 
                    self.solver_param.line_follow_weights.q_reverse, 
                    self.solver_param.line_follow_weights.q_d_ang_vel, 
                    self.solver_param.line_follow_weights.q_theta_master, 
                    self.solver_param.line_follow_weights.q_theta_slave, 
                    self.solver_param.line_follow_weights.q_pos_N, 
                    self.solver_param.line_follow_weights.q_theta_N, 
                    self.solver_param.line_follow_weights.q_distance, 
                    self.solver_param.line_follow_weights.q_theta_diff, 
                    self.solver_param.line_follow_weights.q_distance_c, 
                    self.solver_param.line_follow_weights.q_obs_c, 
                    self.solver_param.line_follow_weights.q_dyn_obs_c, 
                    self.solver_param.line_follow_weights.q_future_dyn_obs, 
                    self.solver_param.line_follow_weights.q_acc_c, 
                    self.solver_param.line_follow_weights.enable_distance_constraint, 
                    self.solver_param.line_follow_weights.q_formation_ang,
                    self.solver_param.line_follow_weights.q_line_theta,
                    ]
        # Put aggressive weights into a list, 
        # Same outcome as code above, 
        weight_list_aggressive = []
        for weight in self.solver_param.line_follow_weights_aggressive:
            if not weight == 'n_weights': # dont include this
                weight_list_aggressive.append(self.solver_param.line_follow_weights_aggressive[weight])

        weight_list = []
        # Merge the two lists, if aggressive_factor is 0 => new list is equal to weight_list_soft
        # if aggresive_factor = 1 => new list is equal to weight_list_aggressive
        # between 0 and 1 weight list is an interpolation of the two.
        assert aggressive_factor <= 1 and aggressive_factor >= 0, " aggressive_factor must be a number between 0 and 1" 
        for i in range(len(weight_list_aggressive)):
            mixed_weight = weight_list_soft[i]*(1-aggressive_factor) + weight_list_aggressive[i] * aggressive_factor
            weight_list.append( mixed_weight)

        if self.state == States.FORMATION:
            u_previous = self.u_previous[:2] + [0,0]
            parameters = x_cur + x_finish + u_previous + acc_init + [self.solver_param.base.constraint['d']] + [distance_lb] + [distance_ub] + [formation_angle_ref] + [self.solver_param.base.constraint['formation_angle_error_margin']] + weight_list + u_ref + refs +  list(constraints) + list(dyn_constraints) + active_dyn_obs + self.bounds_equations 
        else: 
            raise NotImplementedError()

        parameters = [float(param) for param in parameters]

        #store input parameters for simulator plots
        if self.initial_guess_master is not None:
            # self.initial_guess_master = self.initial_guess_master[self.solver_param.base.nu:] + self.initial_guess_master[-self.solver_param.base.nu:]
            self.initial_guess_master = self.initial_guess_master[self.solver_param.base.nu:] + [0]*self.solver_param.base.nu
        solution, ref_cost = self.generate_control_input(parameters, guess=self.initial_guess_master)
                                
        self.initial_guess_master = solution
        return ref_cost, parameters, solution, search_sector

    def gen_traj_following_traj(self, trajectory_ref, u_ref, x_cur, formation_angle_ref, x_finish, constraints, dyn_constraints, active_dyn_obs, distance_lb, distance_ub, aggressive_factor):
        acc_init = self.robot_data.get_prev_acc()
        
        x_ref_master = trajectory_ref[0::6]
        y_ref_master = trajectory_ref[1::6]
        block_dist_master = np.vstack((np.array(x_ref_master)-x_cur[0], np.array(y_ref_master)-x_cur[1]))
        dist_to_refs_master = np.linalg.norm(block_dist_master,axis=0)
        avg_dist_to_ref_master = np.mean(dist_to_refs_master)

        x_ref_slave = trajectory_ref[3::6]
        y_ref_slave = trajectory_ref[4::6]
        block_dist_slave = np.vstack((np.array(x_ref_slave)-x_cur[0], np.array(y_ref_slave)-x_cur[1]))
        dist_to_refs_slave = np.linalg.norm(block_dist_slave,axis=0)
        avg_dist_to_ref_slave = np.mean(dist_to_refs_slave)

        weight_list = []
        weight_list_soft = [self.solver_param.traj_follow_weights.q_lin_v, 
                self.solver_param.traj_follow_weights.q_lin_acc, 
                self.solver_param.traj_follow_weights.q_lin_ret, 
                self.solver_param.traj_follow_weights.q_lin_jerk, 
                self.solver_param.traj_follow_weights.q_ang, 
                self.solver_param.traj_follow_weights.q_ang_acc, 
                self.solver_param.traj_follow_weights.q_ang_jerk, 
                self.solver_param.traj_follow_weights.q_cte, 
                self.solver_param.traj_follow_weights.q_pos_master/avg_dist_to_ref_master, 
                self.solver_param.traj_follow_weights.q_pos_slave/avg_dist_to_ref_slave,
                self.solver_param.traj_follow_weights.q_d_lin_vel_upper, 
                self.solver_param.traj_follow_weights.q_d_lin_vel_lower, 
                self.solver_param.traj_follow_weights.q_reverse, 
                self.solver_param.traj_follow_weights.q_d_ang_vel, 
                self.solver_param.traj_follow_weights.q_theta_master, 
                self.solver_param.traj_follow_weights.q_theta_slave * (1 if np.abs(x_cur[-1]-trajectory_ref[-1]) < np.pi else 50), #The increase in cost should probably be changed to a smooth function of some kind
                self.solver_param.traj_follow_weights.q_pos_N, 
                self.solver_param.traj_follow_weights.q_theta_N, 
                self.solver_param.traj_follow_weights.q_distance,
                self.solver_param.traj_follow_weights.q_theta_diff, 
                self.solver_param.traj_follow_weights.q_distance_c,
                self.solver_param.traj_follow_weights.q_obs_c, 
                self.solver_param.traj_follow_weights.q_dyn_obs_c, 
                self.solver_param.traj_follow_weights.q_future_dyn_obs, 
                self.solver_param.traj_follow_weights.q_acc_c, 
                self.solver_param.traj_follow_weights.enable_distance_constraint, 
                self.solver_param.traj_follow_weights.q_formation_ang,
                self.solver_param.traj_follow_weights.q_line_theta,
                ]
     
        weight_list_aggressive = [self.solver_param.traj_follow_weights_aggressive.q_lin_v, 
                self.solver_param.traj_follow_weights_aggressive.q_lin_acc, 
                self.solver_param.traj_follow_weights_aggressive.q_lin_ret, 
                self.solver_param.traj_follow_weights_aggressive.q_lin_jerk, 
                self.solver_param.traj_follow_weights_aggressive.q_ang, 
                self.solver_param.traj_follow_weights_aggressive.q_ang_acc, 
                self.solver_param.traj_follow_weights_aggressive.q_ang_jerk, 
                self.solver_param.traj_follow_weights_aggressive.q_cte, 
                self.solver_param.traj_follow_weights_aggressive.q_pos_master/avg_dist_to_ref_master, 
                self.solver_param.traj_follow_weights_aggressive.q_pos_slave/avg_dist_to_ref_slave,
                self.solver_param.traj_follow_weights_aggressive.q_d_lin_vel_upper, 
                self.solver_param.traj_follow_weights_aggressive.q_d_lin_vel_lower, 
                self.solver_param.traj_follow_weights_aggressive.q_reverse, 
                self.solver_param.traj_follow_weights_aggressive.q_d_ang_vel, 
                self.solver_param.traj_follow_weights_aggressive.q_theta_master, 
                self.solver_param.traj_follow_weights_aggressive.q_theta_slave * (1 if np.abs(x_cur[-1]-trajectory_ref[-1]) < np.pi else 50), #The increase in cost should probably be changed to a smooth function of some kind
                self.solver_param.traj_follow_weights_aggressive.q_pos_N, 
                self.solver_param.traj_follow_weights_aggressive.q_theta_N, 
                self.solver_param.traj_follow_weights_aggressive.q_distance,
                self.solver_param.traj_follow_weights_aggressive.q_theta_diff, 
                self.solver_param.traj_follow_weights_aggressive.q_distance_c,
                self.solver_param.traj_follow_weights_aggressive.q_obs_c, 
                self.solver_param.traj_follow_weights_aggressive.q_dyn_obs_c, 
                self.solver_param.traj_follow_weights_aggressive.q_future_dyn_obs, 
                self.solver_param.traj_follow_weights_aggressive.q_acc_c, 
                self.solver_param.traj_follow_weights_aggressive.enable_distance_constraint, 
                self.solver_param.traj_follow_weights_aggressive.q_formation_ang,
                self.solver_param.traj_follow_weights_aggressive.q_line_theta,
                ]
        
        # Merge the two lists, if aggressive_factor is 0 => new list is equal to weight_list_soft
        # if aggresive_factor = 1 => new list is equal to weight_list_aggressive
        # between 0 and 1 weight list is an interpolation of the two.
        assert aggressive_factor <= 1 and aggressive_factor >= 0, " aggressive_factor must be a number between 0 and 1" 
        for i in range(len(weight_list_aggressive)):
            mixed_weight = weight_list_soft[i]*(1-aggressive_factor) + weight_list_aggressive[i] * aggressive_factor
            weight_list.append( mixed_weight)

        parameters = x_cur + x_finish + self.u_previous + acc_init + [self.solver_param.base.constraint['d']] + [distance_lb] + [distance_ub] + [formation_angle_ref] + [self.solver_param.base.constraint['formation_angle_error_margin']] + weight_list + u_ref + trajectory_ref +  list(constraints) + list(dyn_constraints) + active_dyn_obs  + self.bounds_equations 
        parameters = [float(param) for param in parameters]

        
        if self.initial_guess_dual is not None:
            # self.initial_guess_dual = self.initial_guess_dual[self.solver_param.base.nu:] + self.initial_guess_dual[-self.solver_param.base.nu:]
            self.initial_guess_dual = self.initial_guess_dual[self.solver_param.base.nu:] + [0]*self.solver_param.base.nu
            
        solution, ref_cost = self.generate_control_input(parameters, guess=self.initial_guess_dual)
        
        #Store solution to be passed as initial guess in next iteration
        self.initial_guess_dual = solution

        return ref_cost, parameters, solution

    def gen_pos_traj(self, x_cur, x_finish, constraints, dyn_constraints, active_dyn_obs, distance_lb, distance_ub):
        d_near = 0.00001
        acc_init = self.robot_data.get_prev_acc()

        refs = x_finish * self.solver_param.base.n_hor
        u_ref = [0]*2*2*self.solver_param.base.n_hor

        if ((x_cur[0]-x_finish[0])**2 + (x_cur[1]-x_finish[1])**2) > d_near:
            q_master_denominator = ((x_cur[0]-x_finish[0])**2 + (x_cur[1]-x_finish[1])**2)
        else:
            q_master_denominator = 1
        if ((x_cur[3]-x_finish[3])**2 + (x_cur[4]-x_finish[4])**2) > d_near:
            q_slave_denominator = ((x_cur[3]-x_finish[3])**2 + (x_cur[4]-x_finish[4])**2)
        else:
            q_slave_denominator = 1

        weight_list = [self.solver_param.pos_goal_weights.q_lin_v, 
                self.solver_param.pos_goal_weights.q_lin_acc, 
                self.solver_param.pos_goal_weights.q_lin_ret, 
                self.solver_param.pos_goal_weights.q_lin_jerk, 
                self.solver_param.pos_goal_weights.q_ang, 
                self.solver_param.pos_goal_weights.q_ang_acc, 
                self.solver_param.pos_goal_weights.q_ang_jerk, 
                self.solver_param.pos_goal_weights.q_cte, 
                self.solver_param.pos_goal_weights.q_pos_master/q_master_denominator, 
                self.solver_param.pos_goal_weights.q_pos_slave*self.solver_param.base.enable_slave/q_slave_denominator,
                self.solver_param.pos_goal_weights.q_d_lin_vel_upper, 
                self.solver_param.pos_goal_weights.q_d_lin_vel_lower, 
                self.solver_param.pos_goal_weights.q_reverse, 
                self.solver_param.pos_goal_weights.q_d_ang_vel, 
                self.solver_param.pos_goal_weights.q_theta_master*np.allclose(np.array(x_cur)[[0,1]], np.array(x_finish)[[0,1]], atol=2, rtol=0.00001), 
                self.solver_param.pos_goal_weights.q_theta_slave*self.solver_param.base.enable_slave* np.allclose(np.array(x_cur)[[3,4]], np.array(x_finish)[[3,4]], atol=2, rtol=0.00001),
                self.solver_param.pos_goal_weights.q_pos_N/(min(q_master_denominator, q_slave_denominator)), 
                self.solver_param.pos_goal_weights.q_theta_N*np.allclose(np.array(x_cur)[[0,1,3,4]], np.array(x_finish)[[0,1,3,4]], atol=2, rtol=0.00001), 
                self.solver_param.pos_goal_weights.q_distance,
                self.solver_param.pos_goal_weights.q_theta_diff, 
                self.solver_param.pos_goal_weights.q_distance_c,
                self.solver_param.pos_goal_weights.q_obs_c, 
                self.solver_param.pos_goal_weights.q_dyn_obs_c, 
                self.solver_param.pos_goal_weights.q_future_dyn_obs, 
                self.solver_param.pos_goal_weights.q_acc_c, 
                self.solver_param.pos_goal_weights.enable_distance_constraint, 
                self.solver_param.pos_goal_weights.q_formation_ang,
                self.solver_param.pos_goal_weights.q_line_theta,
                ]
        
        
        parameters = x_cur + x_finish + self.u_previous + acc_init + [self.solver_param.base.constraint['d']] + [distance_lb] + [distance_ub] + [0] + [self.solver_param.base.constraint['formation_angle_error_margin']] + weight_list + u_ref + refs +  list(constraints) + list(dyn_constraints) + active_dyn_obs + self.bounds_equations 
        parameters = [float(param) for param in parameters]

        
        if self.initial_guess_pos is not None:
            # self.initial_guess_pos = self.initial_guess_pos[self.solver_param.base.nu:] + self.initial_guess_pos[-self.solver_param.base.nu:]
            self.initial_guess_pos = self.initial_guess_pos[self.solver_param.base.nu:] + [0]*self.solver_param.base.nu
            
        solution, ref_cost = self.generate_control_input(parameters, guess=self.initial_guess_pos)
        
        #Store solution to be passed as initial guess in next iteration
        self.initial_guess_pos = solution

        return ref_cost, parameters, solution

    def offset_trajectory(self, base_trajectory, offset_angle):
        offset_traj = []
        for t in range(self.solver_param.base.n_hor):
            master_ref = base_trajectory[t, :3]
            slave_ref = master_ref[:2] + np.array([np.cos(offset_angle)*self.solver_param.base.constraint['d'], np.sin(offset_angle)*self.solver_param.base.constraint['d']])

            # Calculate the angle the slave should be in
            slave_prev = base_trajectory[0, :3] + np.array([np.cos(offset_angle)*self.solver_param.base.constraint['d'], np.sin(offset_angle)*self.solver_param.base.constraint['d'], 0]) if t == 0 else base_trajectory[t-1, :3] + np.array([np.cos(offset_angle)*self.solver_param.base.constraint['d'], np.sin(offset_angle)*self.solver_param.base.constraint['d'], 0])
            slave_next = base_trajectory[1, :3] + np.array([np.cos(offset_angle)*self.solver_param.base.constraint['d'], np.sin(offset_angle)*self.solver_param.base.constraint['d'], 0]) if t == 0 else base_trajectory[-1, :3] + np.array([np.cos(offset_angle)*self.solver_param.base.constraint['d'], np.sin(offset_angle)*self.solver_param.base.constraint['d'], 0]) if t == self.solver_param.base.n_hor-1 else base_trajectory[t, :3] + np.array([np.cos(offset_angle)*self.solver_param.base.constraint['d'], np.sin(offset_angle)*self.solver_param.base.constraint['d'], 0])
            theta_slave = np.arctan2(slave_next[1]-slave_prev[1], slave_next[0]-slave_prev[0])  
            
            if theta_slave < 0 and slave_prev[-1] > np.pi/2:
                theta_slave += 2*np.pi
            elif theta_slave > 0 and slave_prev[-1] < -np.pi/2:
                theta_slave -= 2*np.pi

            slave_ref = np.hstack((slave_ref, theta_slave))
            offset_traj += list(master_ref) + list(slave_ref)

        #Make sure all distances diff is d
        assert  1e-5 > self.solver_param.base.constraint['d'] - np.sum([(np.array(offset_traj[i::6]) - np.array(offset_traj[3+i::6]))**2 for i in range(2)])/self.solver_param.base.n_hor

        return offset_traj

    def generate_trajectory(self, master_end, slave_end, plot): 
        """[summary]
        This should take as input x cur and then simulate the complete trajectory to end. 
        It should store it's own ATRs to keep track of where the simulated position is.
        Should also not request updates on the map but instead work with the current data.
        """
        generated_trajectory = [[],[]]       
        t = 0
        search_sector = None # TODO:Maybe remove this

        # Initialize lists
        refs = [0.0] * (self.solver_param.base.n_hor * self.solver_param.base.nx)
        #This is filled later anyway
        constraints = [] 
        dyn_constraints = []
        if self.u_previous is None:
            self.u_previous = [0]*self.solver_param.base.nu

        #u_previous = self.initial_guess[:self.config.nu]
        done = False
        # This is the main loop which will run until the goal is hit
        while not done:
            start_time = perf_counter_ns()
            x_cur = list(self.robot_data.get_states()) # I think this means that we cannot plan all the way to goal since we get the current position all the time

            # While simulating this calculates the next step for the obstacles. While running via ros it requests updates positions for them
            self.path_planner.update_obstacles()  

            # Generate obstacle constraints
            constraints, closest_static_unexpected_obs = self.convert_static_obs_to_eqs(x_cur[:2])
            dyn_constraints, closest_dynamic_obs_poly, closest_dynamic_obs_ellipse = list(self.convert_dynamic_obs_to_eqs(x_cur[:2]))
            active_dyn_obs = self.get_active_dyn_obs(dyn_constraints) # TODO: try to do this in solver as well, 

            # Generate reference values
            v_ref = self.generate_vel_ref(x_cur[:3])
            x_finish_master, line_vertices_master, goal_in_reach = self.generate_refs(x_cur, self.path_planner.path)
            slave_path = self.path_planner.path[0:-1] + [(slave_end[0], slave_end[1])]
            x_finish_slave, line_vertices_slave, _ = self.generate_refs(x_cur[3:], slave_path)
            
            # Calculate reference angles
            formation_angle_ref = self.mpc_generator.calc_formation_angle_ref(x_cur[:3], *line_vertices_master[:3])

            ref_trajectory = None

            # Calcutae trajectories
            if self.state == States.COUPLING or self.state == States.DECOUPLING:
                ref_cost, parameters, solution = self.gen_pos_traj(x_cur, master_end + slave_end, constraints, dyn_constraints, active_dyn_obs,  self.solver_param.base.vehicle_margin, 1e100)

            elif self.state == States.FORMATION: 
                # If goal is not in reach, follow the lines
                if not goal_in_reach:
                    # Generate one trajectory for master
                    ref_cost, parameters, solution, search_sector = self.gen_line_following_traj(x_cur, dyn_constraints, closest_dynamic_obs_poly, line_vertices_master, line_vertices_slave, formation_angle_ref, constraints, active_dyn_obs, self.solver_param.base.constraint['distance_lb'], self.solver_param.base.constraint['distance_ub'], self.solver_param.base.aggressive_factor)

                    if self.solver_param.base.enable_slave:
                        # Offset master trajectory to receive slave trajectory
                        trajectory = self.calculate_trajectory(solution[:2*2*self.solver_param.base.trajectory_length], x_cur[:3], x_cur[3:])
                        ref_trajectory = self.offset_trajectory(trajectory, formation_angle_ref)
                        
                        ref_cost, parameters, solution = self.gen_traj_following_traj(ref_trajectory, solution, x_cur, formation_angle_ref, x_finish_master + x_finish_slave, constraints, dyn_constraints, active_dyn_obs, self.solver_param.base.constraint['distance_lb'], self.solver_param.base.constraint['distance_ub'], self.solver_param.base.aggressive_factor)
                
                # If goal is in reach, go to the goal
                elif goal_in_reach:
                    if self.solver_param.base.enable_slave:
                        ref_cost, parameters, solution = self.gen_pos_traj(x_cur, master_end + slave_end, constraints, dyn_constraints, active_dyn_obs, self.solver_param.base.constraint['distance_lb'], self.solver_param.base.constraint['distance_ub'])
                    else:
                        ref_cost, parameters, solution = self.gen_pos_traj(x_cur, master_end + slave_end, constraints, dyn_constraints, active_dyn_obs, self.solver_param.base.vehicle_margin, 1e100)
                    

            else:
                raise ValueError(f"Invalid state to be in: {self.state}!")

            
            self.u_previous = solution[:self.solver_param.base.nu]
            #store input parameters for simulator plots
            self.mpc_generator.store_params(parameters)
            
                
            # Pass the control inputs to something 
            self.robot_data.update_pos(self.u_previous, np.array(line_vertices_master)[:,0], np.array(line_vertices_master)[:,1], adjust_theta=True)

            trajectory = self.calculate_trajectory(solution[:2*2*self.solver_param.base.trajectory_length], x_cur[:3], x_cur[3:])

            # Update the generated trajectory with the recently generated trajectory
            generated_trajectory[0].append(trajectory[0,0:3])
            if self.solver_param.base.enable_slave:
                generated_trajectory[1].append(trajectory[0,3:6])

            # Plot
            if plot:
                self.plot(trajectory, closest_static_unexpected_obs, closest_dynamic_obs_poly, closest_dynamic_obs_ellipse, line_vertices_master, line_vertices_slave, x_finish_master, x_finish_slave, search_sector, ref_cost, master_end, slave_end, ref_trajectory)

            
            t += self.solver_param.base.num_steps_taken
            self.times['loop_time'].append(perf_counter_ns() - start_time)

            break # TODO: This is temporary. It is so that not all of the trajectory has to be generated each loop

        return generated_trajectory

    def run(self, plot=False):
        """The main function for moving from start to end position
        """
        done = False

        # Generate trajectory
        # Check if there are unexpected obstacles in the way. 
        # If there are create a visibility graph and plan around it 
        x = self.robot_data.get_states()
        self.path_planner.local_path_plan(x[:3])

        # If the slave is disabled we shouldn't try to trajectory plan for it
        if not self.solver_param.base.enable_slave:
            self.slave_end = list(x[3:])

        # Generate one full trajectory
        trajectory = self.generate_trajectory(self.master_end, self.slave_end, plot)
        

        # Check if done with current state
        done_with_state = self.state.transition(self.robot_data.get_states(), self.master_end + self.slave_end)
        if done_with_state:
            # TODO: This should maybe be published and we'll receive a command to move to the next state
            if self.verbose:
                print(f"{self.print_name} Finished with state: {self.state.name}, moving on to state: {self.state.next().name}")
            self.state = self.state.next() 


            # Add to ATRs the time step when state transition
            self.robot_data.state_transitions_t[self.state.name.lower()] = len(self.robot_data.past_u[0])//2

            # Check if done
            if self.state == States.DONE:
                done = True
            else:
                # Update current end positions
                self.master_end = list(self.map_data.positions_master.next())
                self.slave_end = list(self.map_data.positions_slave.next())


        # Fulfix
        self.ra.update(x)
        if np.abs(self.ra.get_avg() - np.sum(x)) < 1e-2:
            print(f"{self.print_name} Stuck!")
            done = True
        
        # Print how much the distance constraint was max violated
        if done:
            print(f"Graph nr: violated distance constraint as max of {max([self.solver_param.base.constraint['d'] - d for d in self.robot_data.calc_constraint_violation(state='formation')])}") 


        # If this doesn't happen then someone else has to stop this process and it's associated processes
        if done and self.self_destruct:
            self.kill()

        
        return done
        

    def calculate_trajectory(self, u, master_state, slave_state):
        #TODO: do this in a more efficient way
        atr = ATRS(master_state, slave_state, self.solver_param, self.plot_config, False)
        [atr.update_pos(u[i*4:i*4+4], None, None) for i in range(len(u)//4)]
        x_master, y_master, theta_master, x_slave, y_slave, theta_slave = atr.get_prev_states()
        v_master, ang_master, v_slave, ang_slave = atr.get_prev_control_signals()

        x_master = np.array(x_master[1:], ndmin=2).transpose()
        y_master = np.array(y_master[1:], ndmin=2).transpose()
        theta_master = np.array(theta_master[1:], ndmin=2).transpose()

        x_slave = np.array(x_slave[1:], ndmin=2).transpose()
        y_slave = np.array(y_slave[1:], ndmin=2).transpose()
        theta_slave = np.array(theta_slave[1:], ndmin=2).transpose()

        v_master = np.array(v_master, ndmin=2).transpose()
        ang_master = np.array(ang_master, ndmin=2).transpose()

        v_slave = np.array(v_slave, ndmin=2).transpose()
        ang_slave = np.array(ang_slave, ndmin=2).transpose()

        trajectory = np.concatenate((x_master, y_master, theta_master, x_slave, y_slave, theta_slave, v_master, ang_master, v_slave, ang_slave), axis=1)

        return trajectory
        

    
    def plot(self, trajectory, closest_static_unexpected_obs, closest_dynamic_obs_poly, closest_dynamic_obs_ellipse, line_vertices_master, line_vertices_slave, ref_master, ref_slave, search_sector, ref_cost, x_finish_master, x_finish_slave, planned_trajectory):
        
        self.path_planner.plot()
        goal_positions = [self.map_data.positions_master.decoupling_end[:2], self.map_data.positions_slave.decoupling_end[:2]]
        self.robot_data.plot(goal_positions)

        # Plot the closest static and unexpected obstacles
        start_time = perf_counter_ns()
        
        # Dynamic obstacles
        data = []
        if len(closest_dynamic_obs_poly):
            if not self.plot_config['enable_ellipses']:
                data = [np.array(ob.exterior.xy).T for ob in np.array(closest_dynamic_obs_poly)[:, 0]] # Plot the polygon representation of the obstacle

            if self.plot_config['enable_ellipses']:
                # Plot the ellipse representation of the obstacle
                for ob in closest_dynamic_obs_ellipse:
                    data.append(self.path_planner.obs_handler.polygon_to_coord_plotable(ob[0]))
            

        # Static obstacles
        for obs in list(closest_static_unexpected_obs.values()):
            obs = [np.array(ob.exterior.xy).T for ob in obs]
            data += obs

        

        # Plot planned trajectory
        try:
            if planned_trajectory is not None:
                    master_trajectory_x = planned_trajectory[0::6]
                    master_trajectory_y = planned_trajectory[1::6]
                    self.plot_queues['planned_trajectory_master'].put_nowait(np.vstack((master_trajectory_x, master_trajectory_y)))
                    if self.solver_param.base.enable_slave and self.plot_config['plot_slave']:
                        slave_trajectory_x = planned_trajectory[3::6]
                        slave_trajectory_y = planned_trajectory[4::6]
                        self.plot_queues['planned_trajectory_slave'].put_nowait(np.vstack((slave_trajectory_x, slave_trajectory_y)))
            else:
                self.plot_queues['planned_trajectory_master'].put_nowait([[],[]])
                if self.plot_config['plot_slave']:
                    self.plot_queues['planned_trajectory_slave'].put_nowait([[],[]])
        except Full:
            pass


        # Plot search sector
        try:
            pass
            # y,x = search_sector.exterior.xy
            # self.plot_queues['search_sector'].put_nowait([x,y])
        except Full:
            pass

        
        # Plot costs 
        all_positions_master = trajectory[:,0:2].T
        all_x_master = all_positions_master[0,:]
        all_y_master = all_positions_master[1,:]
        all_th_master = trajectory[:,2].T
        all_positions_slave = trajectory[:,3:5].T
        all_x_slave = all_positions_slave[0,:]
        all_y_slave = all_positions_slave[1,:]
        all_th_slave = trajectory[:,5].T
        self.mpc_generator.ref_points_master_x = np.array(self.mpc_generator.ref_points_master_x)
        self.mpc_generator.ref_points_master_y = np.array(self.mpc_generator.ref_points_master_y)
        self.mpc_generator.ref_points_master_th = np.array(self.mpc_generator.ref_points_master_th)
        self.mpc_generator.ref_points_slave_x = np.array(self.mpc_generator.ref_points_slave_x)
        self.mpc_generator.ref_points_slave_y = np.array(self.mpc_generator.ref_points_slave_y)
        self.mpc_generator.ref_points_slave_th = np.array(self.mpc_generator.ref_points_slave_th)
            
        master_ref_point_cost = self.mpc_generator.cost_dist2ref_points(all_x_master, all_y_master, all_th_master, self.mpc_generator.ref_points_master_x, self.mpc_generator.ref_points_master_y,  self.mpc_generator.ref_points_master_th, self.mpc_generator.q_pos_master, self.mpc_generator.q_theta_master)
        slave_ref_point_cost = self.mpc_generator.cost_dist2ref_points( all_x_slave,  all_y_slave,  all_th_slave,  self.mpc_generator.ref_points_slave_x,  self.mpc_generator.ref_points_slave_y,   self.mpc_generator.ref_points_slave_th,  self.mpc_generator.q_pos_slave,  self.mpc_generator.q_theta_slave)
        master_ref_point_N_cost = self.mpc_generator.cost_dist2ref_point_N(all_x_master[-1], all_y_master[-1], all_th_master[-1], self.mpc_generator.ref_points_master_x[-1], self.mpc_generator.ref_points_master_y[-1], self.mpc_generator.ref_points_master_th[-1], self.mpc_generator.q_pos_N, self.mpc_generator.q_theta_N)
        slave_ref_point_N_cost =  self.mpc_generator.cost_dist2ref_point_N(all_x_slave[-1], all_y_slave[-1], all_th_slave[-1], self.mpc_generator.ref_points_slave_x[-1], self.mpc_generator.ref_points_slave_y[-1], self.mpc_generator.ref_points_slave_th[-1], self.mpc_generator.q_pos_N, self.mpc_generator.q_theta_N)
        self.costs['cost_master_ref_points'].append(master_ref_point_cost + master_ref_point_N_cost)
        self.costs['cost_slave_ref_points'].append(slave_ref_point_cost + slave_ref_point_N_cost)
        distance_cost = self.mpc_generator.cost_distance_between_atrs(all_x_master, all_y_master, all_x_slave, all_y_slave, self.mpc_generator.q_distance)
        distance_cost += self.mpc_generator.cost_distance_atr_soft_c( all_x_master, all_y_master, all_x_slave, all_y_slave, self.mpc_generator.q_distance_c)
        self.costs['cost_constraint'].append(distance_cost)
        self.costs_future['cost_future_constraint'] = np.sum((self.mpc_generator.cost_distance_between_atrs(all_x_master, all_y_master, all_x_slave, all_y_slave, self.mpc_generator.q_distance, individual_costs=True) , self.mpc_generator.cost_distance_atr_soft_c( all_x_master, all_y_master, all_x_slave, all_y_slave, self.mpc_generator.q_distance_c, individual_costs=True)), axis=0)
        
        self.costs['cost_master_line_deviation'].append(float(self.mpc_generator.cost_dist2ref_line(all_x_master, all_y_master, all_th_master, self.mpc_generator.ref_points_master_x, self.mpc_generator.ref_points_master_y, self.mpc_generator.q_cte, self.mpc_generator.q_line_theta)))
        self.costs_future['cost_future_master_line_deviation'] = self.mpc_generator.cost_dist2ref_line(all_x_master, all_y_master, all_th_master, self.mpc_generator.ref_points_master_x, self.mpc_generator.ref_points_master_y, self.mpc_generator.q_cte, self.mpc_generator.q_line_theta, individual_costs=True)
        # self.costs['cost_slave_line_deviation'].append(float(self.mpc_generator.cost_dist2ref_line(all_x_slave,all_y_slave, all_th_slave, self.mpc_generator.ref_points_slave_x, self.mpc_generator.ref_points_slave_y, self.mpc_generator.q_cte*self.mpc_generator.enable_distance_constraint,self.mpc_generator.q_line_theta*self.mpc_generator.enable_distance_constraint*0 )))

        self.costs['cost_master_dynamic_obs'].append(float(self.mpc_generator.cost_inside_dyn_ellipse2(all_x_master, all_y_master, self.mpc_generator.q_dyn_obs_c)))
        self.costs_future['cost_future_master_dynamic_obs'] = self.mpc_generator.cost_inside_dyn_ellipse2(all_x_master, all_y_master, self.mpc_generator.q_dyn_obs_c, individual_costs=True)
        self.costs['cost_slave_dynamic_obs'].append(float(self.mpc_generator.cost_inside_dyn_ellipse2(all_x_slave, all_y_slave, self.mpc_generator.q_dyn_obs_c)))
        self.costs_future['cost_future_slave_dynamic_obs'] = self.mpc_generator.cost_inside_dyn_ellipse2(all_x_slave, all_y_slave, self.mpc_generator.q_dyn_obs_c, individual_costs=True)

        cost_static_master = float(self.mpc_generator.cost_inside_static_object(all_x_master, all_y_master, self.mpc_generator.q_obs_c))
        cost_static_slave = float(self.mpc_generator.cost_inside_static_object(all_x_slave, all_y_slave, self.mpc_generator.q_obs_c))
        cost_bounds_master = float(self.mpc_generator.cost_outside_bounds(all_x_master, all_y_master, self.mpc_generator.q_obs_c))
        cost_bounds_slave = float(self.mpc_generator.cost_outside_bounds(all_x_slave, all_y_slave, self.mpc_generator.q_obs_c))
        self.costs['cost_master_static_obs'].append(float(cost_static_master + cost_bounds_master)) #TODO: Split boudns cost into seperate plot function
        self.costs_future['cost_future_master_static_obs'] = self.mpc_generator.cost_inside_static_object(all_x_master, all_y_master, self.mpc_generator.q_obs_c, individual_costs=True)
        self.costs['cost_slave_static_obs'].append(float(cost_static_slave + cost_bounds_slave)) #TODO: Split boudns cost into seperate plot function
        self.costs_future['cost_future_slave_static_obs'] = self.mpc_generator.cost_inside_static_object(all_x_slave, all_y_slave, self.mpc_generator.q_obs_c, individual_costs=True)
        #######  
        # Master control signal costs
        master_u = trajectory[:, [6, 7]].reshape(-1, 1)
        master_lin_vel_cost, master_ang_vel_cost = self.mpc_generator.cost_control(master_u, self.mpc_generator.q_lin_v, self.mpc_generator.q_ang)
        self.costs['cost_master_lin_vel'].append(float(master_lin_vel_cost))
        self.costs_future['cost_future_master_lin_vel'] = self.mpc_generator.cost_control(master_u, self.mpc_generator.q_lin_v, self.mpc_generator.q_ang, individual_costs=True)[0]
        self.costs['cost_master_ang_vel'].append(float(master_ang_vel_cost))
        self.costs_future['cost_future_master_ang_vel'] = self.mpc_generator.cost_control(master_u, self.mpc_generator.q_lin_v, self.mpc_generator.q_ang, individual_costs=True)[1]

        # Master accelerations cost
        master_lin_vel_init = 0 if len(self.robot_data.past_u[0]) < 4 else self.robot_data.past_u[0][-4]
        master_ang_vel_init = 0 if len(self.robot_data.past_u[0]) < 4 else self.robot_data.past_u[0][-3]
        master_lin_acc, master_ang_acc = self.mpc_generator.calc_accelerations(master_u, master_lin_vel_init, master_ang_vel_init)

        master_lin_acc_cost = self.mpc_generator.cost_acc_and_retardation(master_lin_acc, self.mpc_generator.q_lin_acc, self.mpc_generator.q_lin_acc)        
        # master_lin_acc_cost += self.mpc_generator.cost_acc_constraint(master_lin_acc, self.solver_param.base.lin_acc_min, self.solver_param.base.lin_acc_max, self.mpc_generator.q_acc_c)
        self.costs['cost_master_lin_acc'].append(float(master_lin_acc_cost))
        self.costs_future['cost_future_master_lin_acc'] = self.mpc_generator.cost_acc_and_retardation(master_lin_acc, self.mpc_generator.q_lin_acc, self.mpc_generator.q_lin_acc, individual_costs=True)    

        master_ang_acc_cost = self.mpc_generator.cost_acc(master_ang_acc, self.mpc_generator.q_ang_acc)
        # master_ang_acc_cost += self.mpc_generator.cost_acc_constraint(master_ang_acc, self.solver_param.base.lin_acc_min, self.solver_param.base.lin_acc_max, self.mpc_generator.q_acc_c)
        self.costs['cost_master_ang_acc'].append(float(master_ang_acc_cost))
        self.costs_future['cost_future_master_ang_acc'] = self.mpc_generator.cost_acc(master_ang_acc, self.mpc_generator.q_ang_acc, individual_costs=True)

        # Master jerk cost
        if len(self.robot_data.past_u[0]) >= 4:
            master_lin_acc_init = (self.robot_data.past_u[0][-4]-self.robot_data.past_u[0][-2])/self.solver_param.base.ts
            master_ang_acc_init = (self.robot_data.past_u[0][-3]-self.robot_data.past_u[0][-1])/self.solver_param.base.ts
            master_lin_jerk, master_ang_jerk = self.mpc_generator.calc_jerk(master_lin_acc, master_ang_acc, master_lin_acc_init, master_ang_acc_init)

            master_lin_jerk_cost = self.mpc_generator.cost_jerk(master_lin_jerk, self.mpc_generator.q_lin_jerk)
            self.costs['cost_master_lin_jerk'].append(float(master_lin_jerk_cost))
            self.costs_future['cost_future_master_lin_jerk'] = self.mpc_generator.cost_jerk(master_lin_jerk, self.mpc_generator.q_lin_jerk, individual_costs=True)

            master_ang_jerk_cost = self.mpc_generator.cost_jerk(master_ang_jerk, self.mpc_generator.q_ang_jerk)
            self.costs['cost_master_ang_jerk'].append(float(master_ang_jerk_cost))
            self.costs_future['cost_future_master_ang_jerk'] = self.mpc_generator.cost_jerk(master_ang_jerk, self.mpc_generator.q_ang_jerk, individual_costs=True)

        else:
            self.costs['cost_master_lin_jerk'].append(0)
            self.costs['cost_master_ang_jerk'].append(0)

            #Future jerk costs should be plotted here but w/e

        
        # Master line deviation cost
        

        # slave control signal costs
        slave_u = trajectory[:, [8, 9]].reshape(-1, 1)
        slave_lin_vel_cost, slave_ang_vel_cost = self.mpc_generator.cost_control(slave_u, self.mpc_generator.q_lin_v, self.mpc_generator.q_ang)
        self.costs['cost_slave_lin_vel'].append(float(slave_lin_vel_cost))
        self.costs_future['cost_future_slave_lin_vel'] = self.mpc_generator.cost_control(slave_u, self.mpc_generator.q_lin_v, self.mpc_generator.q_ang, individual_costs=True)[0]
        self.costs['cost_slave_ang_vel'].append(float(slave_ang_vel_cost))
        self.costs_future['cost_future_slave_ang_vel'] = self.mpc_generator.cost_control(slave_u, self.mpc_generator.q_lin_v, self.mpc_generator.q_ang, individual_costs=True)[1]

        # Slave accelerations cost
        slave_lin_vel_init = 0 if len(self.robot_data.past_u[1]) < 4 else self.robot_data.past_u[1][-4]
        slave_ang_vel_init = 0 if len(self.robot_data.past_u[1]) < 4 else self.robot_data.past_u[1][-3]
        slave_lin_acc, slave_ang_acc = self.mpc_generator.calc_accelerations(slave_u, slave_lin_vel_init, slave_ang_vel_init)

        slave_lin_acc_cost = self.mpc_generator.cost_acc_and_retardation(slave_lin_acc, self.mpc_generator.q_lin_acc, self.mpc_generator.q_lin_ret)        
        slave_lin_acc_cost += self.mpc_generator.cost_acc_constraint(slave_lin_acc, self.solver_param.base.lin_acc_min, self.solver_param.base.lin_acc_max, self.mpc_generator.q_acc_c)
        self.costs['cost_slave_lin_acc'].append(float(slave_lin_acc_cost))
        self.costs_future['cost_future_slave_lin_acc'] = self.mpc_generator.cost_acc_and_retardation(slave_lin_acc, self.mpc_generator.q_lin_acc, self.mpc_generator.q_lin_ret, individual_costs=True)   + self.mpc_generator.cost_acc_constraint(slave_lin_acc, self.solver_param.base.lin_acc_min, self.solver_param.base.lin_acc_max, self.mpc_generator.q_acc_c, individual_costs=True) #TODO: Somehow add these up

        slave_ang_acc_cost = self.mpc_generator.cost_acc(slave_ang_acc, self.mpc_generator.q_ang_acc)
        slave_ang_acc_cost += self.mpc_generator.cost_acc_constraint(slave_ang_acc, self.solver_param.base.lin_acc_min, self.solver_param.base.lin_acc_max, self.mpc_generator.q_acc_c)
        self.costs['cost_slave_ang_acc'].append(float(slave_ang_acc_cost))
        self.costs_future['cost_future_slave_ang_acc'] = self.mpc_generator.cost_acc(slave_ang_acc, self.mpc_generator.q_ang_acc, individual_costs=True) + self.mpc_generator.cost_acc_constraint(slave_ang_acc, self.solver_param.base.lin_acc_min, self.solver_param.base.lin_acc_max, self.mpc_generator.q_acc_c, individual_costs=True)

        # Slave jerk cost
        if len(self.robot_data.past_u[0]) >= 4:
            slave_lin_acc_init = (self.robot_data.past_u[1][-4]-self.robot_data.past_u[1][-2])/self.solver_param.base.ts
            slave_ang_acc_init = (self.robot_data.past_u[1][-3]-self.robot_data.past_u[1][-1])/self.solver_param.base.ts
            slave_lin_jerk, slave_ang_jerk = self.mpc_generator.calc_jerk(slave_lin_acc, slave_ang_acc, slave_lin_acc_init, slave_ang_acc_init)

            slave_lin_jerk_cost = self.mpc_generator.cost_jerk(slave_lin_jerk, self.mpc_generator.q_lin_jerk)
            self.costs['cost_slave_lin_jerk'].append(float(slave_lin_jerk_cost))
            self.costs_future['cost_future_slave_lin_jerk'] = self.mpc_generator.cost_jerk(slave_lin_jerk, self.mpc_generator.q_lin_jerk, individual_costs=True)


            slave_ang_jerk_cost = self.mpc_generator.cost_jerk(slave_ang_jerk, self.mpc_generator.q_ang_jerk)
            self.costs['cost_slave_ang_jerk'].append(float(slave_ang_jerk_cost))
            self.costs_future['cost_future_slave_ang_jerk'] = self.mpc_generator.cost_jerk(slave_ang_jerk, self.mpc_generator.q_ang_jerk, individual_costs=True)

        else:
            self.costs['cost_slave_lin_jerk'].append(0)
            self.costs['cost_slave_ang_jerk'].append(0)

        # Slave line deviation cost



        # Constraints cost

        # Vel ref costs
        vel_ref_cost = float(self.mpc_generator.cost_v_ref_difference(master_u.flatten(), np.array(self.mpc_generator.v_ref_master), self.mpc_generator.q_d_lin_vel_upper, self.mpc_generator.q_d_lin_vel_lower))
        vel_ref_cost += float(self.mpc_generator.cost_ang_vel_ref_difference(master_u.flatten(), np.array(self.mpc_generator.ang_vel_ref_master), self.mpc_generator.q_d_ang_vel))
        self.costs['cost_master_vel_ref'].append(vel_ref_cost)
        self.costs_future['cost_future_master_vel_ref'] = self.mpc_generator.cost_v_ref_difference(master_u.flatten(), np.array(self.mpc_generator.v_ref_master), self.mpc_generator.q_d_lin_vel_upper, self.mpc_generator.q_d_lin_vel_lower, individual_costs=True) + self.mpc_generator.cost_ang_vel_ref_difference(master_u.flatten(), np.array(self.mpc_generator.ang_vel_ref_master), self.mpc_generator.q_d_ang_vel, individual_costs=True) 
        vel_ref_cost = float(self.mpc_generator.cost_reverse(slave_u[0::2], self.mpc_generator.q_reverse))
        self.costs['cost_slave_vel_ref'].append(vel_ref_cost)
        self.costs_future['cost_future_slave_vel_ref'] = self.mpc_generator.cost_reverse(slave_u[0::2], self.mpc_generator.q_reverse, individual_costs=True)
        


        past_time = np.linspace(0, self.solver_param.base.ts*(len(self.costs['cost_master_line_deviation'])) - self.solver_param.base.ts, len(self.costs['cost_master_line_deviation']))
        future_time = np.linspace(0, self.solver_param.base.ts*(len(self.costs_future['cost_future_master_line_deviation'])) - self.solver_param.base.ts, len(self.costs_future['cost_future_master_line_deviation'])) + past_time[-1]

        try: # Try statement needed in case the queue is full
            for key in self.costs:
                try:
                    if self.costs[key] != []:
                        self.plot_queues[key].put_nowait([past_time, self.costs[key]])
                except KeyError:
                    print(f"Key not found:  {key}")

        except Full:
            pass
        try: # Try statement needed in case the queue is full
            for key in self.costs_future:
                try:
                    if get_length(self.costs_future[key]) > 0:
                        future_costs = [float(self.costs_future[key][i]) for i in range(get_length(self.costs_future[key]))]
                        self.plot_queues[key].put_nowait([future_time, future_costs])
                except KeyError:
                    print(f"Key not found:  {key}")

        except Full:
            pass


        self.times['plot'].append(perf_counter_ns() - start_time)

    def kill(self):
        if self.verbose:
            print(f"{self.print_name} is killing itself.")
        self.plot_process.kill()
        
        