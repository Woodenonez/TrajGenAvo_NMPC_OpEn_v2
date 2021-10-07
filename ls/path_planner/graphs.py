import numpy as np
from path_planner.pos_flow import PositionsFlow



class Graph:
    def __init__(self, boundary_coordinates, obstacle_list, nodes, positions_master:PositionsFlow, positions_slave:PositionsFlow=None, unexpected_obstacles=[], dyn_obs_list = []):
        self.boundary_coordinates = boundary_coordinates
        self.processed_boundary_coordinates = []
        self.obstacle_list = obstacle_list
        self.dyn_obs_list = dyn_obs_list 
        self.positions_master = positions_master
        self.positions_slave = positions_slave
        self.nodes = nodes
        self.unexpected_obstacles = unexpected_obstacles
        self.replot = True # If the obstacles should be replotted


        # Check if there are any duplicate nodes
        for node in nodes:
            for node_ in nodes:
                if node == node_:
                    continue
                elif node.coord == node_.coord:
                    raise Exception(str(node.coord) + " is duplicate")

    
    def get_unexpected_obstacles(self):
        self.update_unexpected_obstacles()
        return [ob.coords for ob in self.unexpected_obstacles if ob.appear]

    def get_dyn_obs_list(self): 
        self.update_dynamic_obstacles()
        return self.dyn_obs_list

    def update_unexpected_obstacles(self):
        [ob.update() for ob in self.unexpected_obstacles]

    def update_dynamic_obstacles(self):
        [ob.update() for ob in self.dyn_obs_list]

    def unexpected_obs_update(self):
        return self.get_unexpected_obstacles()

    def dyn_obs_update(self):
        return self.get_dyn_obs_list()

class Node:
    def __init__(self, coord, neighbours):
        self.coord: tuple = coord
        self.neighbours: list = neighbours

class UnexpectedObstacle:
    def __init__(self, coords, appear_after=0, stay_for=float('infinity')):
        self.coords = coords
        self.time_until_appear = appear_after
        self.time_until_dissapear = stay_for + appear_after
        self.appear = False
        # self.triangle_representation = None
        # self.convert_to_triangles()
    
    def update(self):
        self.time_until_appear -= 1
        self.time_until_dissapear -= 1

        if self.time_until_appear <= 0:
            self.appear = True
        if self.time_until_dissapear <= 0:
            self.appear = False

class DynamicObstacle:
    def __init__(self, polygon, p1, p2, vel, n_steps_ahead):
        """Creates a dynamic obstacle class

        Args:
            polygon (list of tuples ): The tuples describes the extremities of the polygon
            p1 (tuple): start position
            p2 (tuple): end position
            dp (tuple): (dx, dy)
        """
        self.polygon = np.array(polygon, dtype='float')
        self.center_pos = np.mean(self.polygon, axis=0)
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.vel = np.array(vel)

        self.future_polygons = []
        self.n_steps_ahead = n_steps_ahead

    def update(self):
        """Updates the current state to one step forward and updates future polygons with future polygons
        """

        def step(polygon, center_pos):
            if any(center_pos > self.p2):
                self.vel = -self.vel

            elif any(center_pos < self.p1):
                self.vel = -self.vel
            
            polygon += self.vel
            center_pos = np.mean(polygon, axis=0)

            return polygon, center_pos


        if len(self.future_polygons) == 0:
            for i in range(self.n_steps_ahead):
                self.polygon, self.center_pos = step(self.polygon, self.center_pos)
                self.future_polygons.append(self.polygon.copy())


        elif len(self.future_polygons) > self.n_steps_ahead:
            print("This code is untested. It's highly experimental. DynamicObstacle.step.elif len(self.future_polygons) > (self.n_steps_ahead:")
            self.polygon = self.future_polygons[1]
            self.center_pos = np.mean(self.polygon, axis=0)
            self.future_polygons = self.future_polygons[1:self.n_steps_ahead+1]
        
        elif len(self.future_polygons) <= self.n_steps_ahead:
            self.polygon = self.future_polygons[1]
            self.center_pos = np.mean(self.polygon, axis=0)
            self.future_polygons = self.future_polygons[1:self.n_steps_ahead+1]

            for i in range(self.n_steps_ahead - (len(self.future_polygons))):
                polygon, _ = step(self.future_polygons[-1].copy(), np.mean(self.future_polygons[-1], axis=0))
                self.future_polygons.append(polygon)
            



        
            



class Graphs:
    """For formation control thesis
    """
    def __init__(self, config):
        self.graphs = []

        ############### First Graph ############################# 
        # Start till mål med statiskt hinder
        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(3.0, 0.0), (6.0, 0.0), (6.0, 10.0), (3, 10.0)] 

        # To be specified in clock-wise ordering
        obstacle_list = [
            [(6.0, 3.0), (6.0, 5.0), (5.0, 5.0), (5.5, 4), (5.0, 3.0)], # Concave
        ]

        nodes = []
        for i in range(8, 1, -1):
            if nodes == []:
                nodes.append(Node((5, i), []))
            else:
                nodes.append(Node((5, i), [nodes[-1]]))
        

        start = (5, 2, 0)
        start_master = start
        end_coupling_master = (5, 2, 1.57)
        end_formation_master = (5, 8, 0)
        end_decoupling_master = (5, 8, 0)
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,1,0)
        end_coupling_slave = (5, 1, 1.57)
        end_formation_slave = (6, 8, 0)
        end_decoupling_slave = (2, 8, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave))


        ############### Second Graph ############################# 
        # Start till mål med unexpected obstacle
        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(3.0, -2), (6.0, -2), (6.0, 10.0), (3, 10.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = []

        unexpected_obstacle = [
            UnexpectedObstacle([(6.0, 3.0), (6.0, 5.0), (5.0, 5.0), (5.0, 3.0)], appear_after=10)
        ]

        nodes = []
        for i in range(9, 0, -1):
            if nodes == []:
                nodes.append(Node((5, i), []))
            else:
                nodes.append(Node((5, i), [nodes[-1]]))
        

        start = (5, 1, 1.57)
        start_master = start
        end_coupling_master = start
        end_formation_master = (5, 9, 0)
        end_decoupling_master = (5, 9, 0)
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,0,1.57)
        end_coupling_slave = (5, 0, 1.57)
        end_formation_slave = (5, 8, 0)
        end_decoupling_slave = (5, 8, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle))
        # self.graphs.append(Graph(boundary_coordinates, obstacle_list, start, end, nodes, unexpected_obstacles=unexpected_obstacle))

        ############### Third Graph ############################# 
        # Start till mål med unexpected obstacle som försvinner efter ett tag
        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(3.0, -2.0), (6.0, -2.0), (6.0, 10.0), (3, 10.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = []

        unexpected_obstacle = [
            UnexpectedObstacle([(6.0, 3.0), (6.0, 5.0), (5.0, 5.0), (5.0, 3.0)], appear_after=10, stay_for=10)
        ]

        nodes = []
        for i in range(9, 0, -1):
            if nodes == []:
                nodes.append(Node((5, i), []))
            else:
                nodes.append(Node((5, i), [nodes[-1]]))
        

        start = (5, 1, 1.57)
        start_master = start
        end_coupling_master = start
        end_formation_master = (5, 9, 0)
        end_decoupling_master = (5, 9, 0)
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,0,1.57)
        end_coupling_slave = (5, 0, 1.57)
        end_formation_slave = (5, 8, 0)
        end_decoupling_slave = (5, 8, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle))
        # self.graphs.append(Graph(boundary_coordinates, obstacle_list, start, end, nodes, unexpected_obstacles=unexpected_obstacle))


        ############### Fourth Graph ############################# 
        # Start till mål med dynamic obstacle
        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(3.0, -2.0), (6.0, -2.0), (6.0, 10.0), (3, 10.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = []

        unexpected_obstacle = []

        dynamic_obstacles = [
            DynamicObstacle([(6.0, 3.0), (6.0, 5.0), (5.0, 5.0), (5.0, 3.0)], (0, 0), (55, 15), (0, 0.05), config.n_hor), 
        ]

        nodes = []
        for i in range(9, 0, -1):
            if nodes == []:
                nodes.append(Node((5, i), []))
            else:
                nodes.append(Node((5, i), [nodes[-1]]))
        

        end = (5, 9, 0)
        start = (5, 1, 1.57)
        start_master = start
        end_coupling_master = start
        end_formation_master = (5, 9, 0)
        end_decoupling_master = (5, 9, 0)
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,0,1.57)
        end_coupling_slave = (5, 0, 1.57)
        end_formation_slave = (5, 8, 0)
        end_decoupling_slave = (5, 8, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, dyn_obs_list=dynamic_obstacles))
        # self.graphs.append(Graph(boundary_coordinates, obstacle_list, start, end, nodes, unexpected_obstacles=unexpected_obstacle, dyn_obs_list = dynamic_obstacles))



        ############### Fifth Graph ############################# 
        # Start till mål med två vägar där den ena är kortare men inte går att ta plankan igenom
        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(0, 0), (0, 5), (5, 5.5), (0, 6), (0, 11), (20, 11), (20, 0), ] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = [[(2, 2), (2, 3), (7, 3), (7, 8), (2, 8), (2, 9), (18, 9), (18, 2)]]

        unexpected_obstacle = []

        dynamic_obstacles = dyn_obs_list = [] # weird


        nodes = []
        nodes.extend([Node((x, 1), []) for x in range(2, 19)])
        nodes.extend([Node((x, 10), []) for x in range(2, 19)])
        nodes.extend([Node((x, 4), []) for x in range(2, 6)])
        nodes.extend([Node((x, 7), []) for x in range(2, 6)])
        nodes.extend([Node((6, y), []) for y in range(4, 8)])
        nodes.extend([Node((1, y), []) for y in range(1, 5)])
        nodes.extend([Node((1, y), []) for y in range(7, 11)])
        nodes.extend([Node((19, y), []) for y in range(1, 11)])



        for i, node  in enumerate(nodes):
            x, y = node.coord
            for j in range(i+1, len(nodes)):
                node_ = nodes[j]
                x_, y_ = node_.coord
                if (x==x_ or y==y_) and abs(x-x_ + y-y_)<2:
                    node.neighbours.append(node_)
                    node_.neighbours.append(node)

        start = (1, 1, 1.57)
        start_master = start
        end_coupling_master = start
        end_formation_master = (1, 10, 0)
        end_decoupling_master = (1, 10, 0)
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (2,1,1.57)
        end_coupling_slave = (2, 1, 1.57)
        end_formation_slave = (2, 10, 0)
        end_decoupling_slave = (2, 10, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, dyn_obs_list=dynamic_obstacles))
        #self.graphs.append(Graph(boundary_coordinates, obstacle_list, start, end, nodes))


        ############### Sixth Graph ############################# 
        # Volvo tuve
        # To be specified in counter-clockwise ordering
        x_min = 0
        x_max = 149
        y_min = 0
        y_max = 16
        safety_dist = 1
        prod_line_start = 11
        
        boundary_coordinates = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)] # AKA drivable area
        boundary_coordinates.reverse()

        # To be specified in clock-wise ordering
        obstacle_list = [
                         [(4, 3.5), (4, 8.5), (83, 8.5), (83, 3.5)], 
                         [(103.5, 7), (103.5, 8.5), (112.5, 8.5), (112.5, 7)], 
                         [(87.5, 3.5), (87.5, 8.5), (100, 8.5), (100, 7), (89, 7), (89, 5), (142, 5), (142, 7), (116, 7), (116, 8.5), (144, 8.5), (144, 3.5), ]
                        ]

        dynamic_obstacles = []
        # Add motor transporter
        motor_from_prod_line = 1.9
        dist_between_motors = 1.5
        dy = motor_from_prod_line+prod_line_start
        dx = 0
        for i in range(int(x_max//(2.3+dist_between_motors))):
            # motor = DynamicObstacle([(dx, dy+0.5), (dx, dy+1), (dx+1.5, dy+1), (dx+1.5, dy+1.3), (dx+2.3, dy+1.3), (dx+2.3, dy), (dx+1.5, dy), (dx+1.5, dy+0.5)], (dx, 0), (dx+2.3+dist_between_motors, 100000), (0.1, 0))
            # dynamic_obstacles.append(motor)
            obstacle_list.append([(dx, dy+0.5), (dx, dy+1), (dx+1.5, dy+1), (dx+1.5, dy+1.3), (dx+2.3, dy+1.3), (dx+2.3, dy), (dx+1.5, dy), (dx+1.5, dy+0.5)])
            dx += dist_between_motors+2.3


        unexpected_obstacle = []

        # dynamic_obstacles = [] 

        # start = (safety_dist, 1.1+safety_dist, 0)
        # end = (safety_dist, prod_line_start-safety_dist, 0)
        start  = (114.25, 6, 3.14)
        end = (24, 11, 0)

        def one_dir_nodes(start, end):
            if start[0] == end[0]:
                nodes = [Node((start[0], y), []) for y in np.linspace(start[1], end[1], int(abs(start[1] - end[1]) + 1))]
            elif start[1] == end[1]:
                nodes = [Node((x, start[1]), []) for x in np.linspace(start[0], end[0], int(abs(start[0] - end[0]) + 1))]
            else:
                print(f"Either x or y must be same in: {start} and {end}")
                exit()

            for i in range(len(nodes) - 1):
                nodes[i].neighbours.append(nodes[i+1])

            nodes = {x.coord: x for x in nodes}
            return nodes


        nodes = one_dir_nodes((x_min+safety_dist, 1.1+safety_dist), (x_max-safety_dist, 1.1+safety_dist))
        
        nodes =  {**nodes, **one_dir_nodes((x_max-safety_dist, 1.1+safety_dist+1), (x_max-safety_dist, prod_line_start-safety_dist))}
        nodes[(x_max-safety_dist, 1.1+safety_dist)].neighbours.append(nodes[(x_max-safety_dist, 1.1+safety_dist+1)])

        nodes =  {**nodes, **one_dir_nodes((x_max-1-safety_dist, prod_line_start-safety_dist), (safety_dist, prod_line_start-safety_dist))}
        nodes[(x_max-safety_dist, prod_line_start-safety_dist)].neighbours.append(nodes[(x_max-1-safety_dist, prod_line_start-safety_dist)])

        nodes =  {**nodes, **one_dir_nodes((safety_dist, prod_line_start-1-safety_dist), (safety_dist, 1.1+safety_dist+1))}
        nodes[(safety_dist, prod_line_start-safety_dist)].neighbours.append(nodes[(safety_dist, prod_line_start-1-safety_dist)])
        nodes[(safety_dist, 1.1+safety_dist+1)].neighbours.append(nodes[(safety_dist, 1.1+safety_dist)])

        # Add kiting area nodes
        nodes =  {**nodes, **one_dir_nodes((114.25, prod_line_start-safety_dist), (114.25, 6))}
        nodes[(114, prod_line_start-1)].neighbours.append(nodes[(114.25, prod_line_start-safety_dist)])

        nodes =  {**nodes, **one_dir_nodes((113.25, 6), (101.75, 6))}
        nodes[(114.25, 6)].neighbours.append(nodes[(113.25, 6)])

        nodes =  {**nodes, **one_dir_nodes((101.75, 7), (101.75, prod_line_start-safety_dist))}
        nodes[(101.75, 6)].neighbours.append(nodes[(101.75, 7)])
        nodes[(101.75, prod_line_start-safety_dist)].neighbours.append(nodes[(101, prod_line_start-safety_dist)])

        # For demo
        nodes[(24, 11)] = Node((24, 11), [])
        nodes[(24, 10)].neighbours.append(nodes[(24, 11)])
        ####

        nodes = list(nodes.values())


        # for i, node  in enumerate(nodes):
        #     x, y = node.coord
        #     for j in range(i+1, len(nodes)):
        #         node_ = nodes[j]
        #         x_, y_ = node_.coord
        #         if (x==x_ or y==y_) and abs(x-x_ + y-y_)<2:
        #             node.neighbours.append(node_)
        #             node_.neighbours.append(node)

        #start  = (114.25, 6, 3.14)
        #end = (24, 11, 0)
        start_master = start
        end_coupling_master = start
        end_formation_master = end
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (115.25,6,3.14)
        end_coupling_slave = start_slave
        end_formation_slave = (24,12,0)
        end_decoupling_slave = end_formation_slave
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))
        #self.graphs.append(Graph(boundary_coordinates, obstacle_list, start, end, nodes, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))


        ############### Seventh Graph ############################# 
        # Dynamic obs
        # Start till mål med dynamic obstacle
        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(-1.0, -1), (10.0, -1), (10.0, 50.0), (-1, 50.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = []

        unexpected_obstacle = []

        dynamic_obstacles = [
            DynamicObstacle([(5, 8.0), (5, 10), (5.51, 10), (5.51, 8.0)], (0, 0), (20, 15), (0, -0.1), config.n_hor), 
            DynamicObstacle([(4, 18.0), (4, 19), (6, 19), (6, 18)], (0, 0), (20, 30), (0, -0.05), config.n_hor), 
        ]

        nodes = []
        for i in range(40, -1, -1):
            if nodes == []:
                nodes.append(Node((5, i), []))
            else:
                nodes.append(Node((5, i), [nodes[-1]]))
        

        start = (5, 1, 1.57)
        end = (5, 40, 0)
        start_master = start
        end_coupling_master = start
        end_formation_master = end
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,0,1.57)
        end_coupling_slave = start_slave
        end_formation_slave = (6,40,0)
        end_decoupling_slave = end_formation_slave
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))
        #self.graphs.append(Graph(boundary_coordinates, obstacle_list, start, end, nodes, unexpected_obstacles=unexpected_obstacle, dyn_obs_list = dynamic_obstacles))

        ############### Eight Graph ############################# 
        # Start till mål med directed graph
        # To be specified in counter-clockwise ordering
        # Rectangle map, includes 180 deg turn

        # Make a connecting rectangle of nodes
        nodes = []
        #nodes.append(Node((5, 9), []))

        for i in range(1, 10):
            if nodes == []:
                nodes.append(Node((6, i), [Node((5, 1), [])]))
            else:
                nodes.append(Node((6, i), [nodes[-1]]))
        for i in range(9, 0, -1):
            if nodes == []:
                nodes.append(Node((5, i), []))
            else:
                nodes.append(Node((5, i), [nodes[-1]]))


        #nodes.append(Node((6, 1), [nodes[-1]]))

        #nodes.append(Node((5, 9), [nodes[-1]]))

        boundary_coordinates = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0, 10.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = []

        unexpected_obstacle = []

        dynamic_obstacles = []

 

        start = (5, 2, 1.57)
        end = (6, 3, 0)
        start_master = start
        end_coupling_master = start
        end_formation_master = end
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,1,1.57)
        end_coupling_slave = start_slave
        end_formation_slave = (6,2,0)
        end_decoupling_slave = end_formation_slave
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))
        #self.graphs.append(Graph(boundary_coordinates, obstacle_list, start, end, nodes, unexpected_obstacles=unexpected_obstacle, dyn_obs_list = dynamic_obstacles))


        ############### Graph 9 ############################# 
        # Dynamic obs
        # Start till mål med dynamic obstacle
        # Test : se om ATR kolliderar med vägg för att undvika dyn obs
        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(3.0, -1), (8.0, -1), (8.0, 10.0), (3, 10.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = [[(2, 0), (2, 15), (4, 15), (4, 0)]]

        unexpected_obstacle = []

        #dynamic_obstacles = [
        #    DynamicObstacle([(6.0, 4.0), (6.0, 5.0), (5.5, 5.0), (5.5, 4.0)], (55, 0), (55, 15), (0, 0.05)), 
        #]
        dynamic_obstacles = [
            DynamicObstacle([(5, 8.0), (5, 8.1), (5.51, 8.1), (5.51, 8.0)], (0, 0), (20, 15), (0, -0.01), config.n_hor), 
        ]
        nodes = []
        for i in range(9, -1, -1):
            if nodes == []:
                nodes.append(Node((5, i), []))
            else:
                nodes.append(Node((5, i), [nodes[-1]]))
        

        start = (5, 1, 1.57)
        end = (5, 9, 0)
        start_master = start
        end_coupling_master = start
        end_formation_master = end
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,0,1.57)
        end_coupling_slave = start_slave
        end_formation_slave = (5,10,0)
        end_decoupling_slave = end_formation_slave
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))
        # self.graphs.append(Graph(boundary_coordinates, obstacle_list, start, end, nodes, unexpected_obstacles=unexpected_obstacle, dyn_obs_list = dynamic_obstacles))

        ###############  Graph 10  ############################# 
        # Start to goal, single line, start in opposite dir
        # To be specified in counter-clockwise ordering
        # long straight line
        boundary_coordinates = [(1.0, 0.0), (8.0, 0.0), (8.0, 35.0), (1, 35.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = []

        unexpected_obstacle = [
        ]

        nodes = []
        for i in range(30, 0, -1):
            if nodes == []:
                nodes.append(Node((5, i), []))
            else:
                nodes.append(Node((5, i), [nodes[-1]]))
        

        start_master = (7,4,0)
        end_coupling_master = (5, 3, 1.57)
        end_formation_master = (5, 25, 0)
        # end_formation_master = (5, 7, 0)
        end_decoupling_master = (2, 25, 0)
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (3,3,0)
        end_coupling_slave = (5, 2, 1.57)
        end_formation_slave = (6, 25, 0)
        # end_formation_slave = (6, 7, 0)
        end_decoupling_slave = (2, 23, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle))

        ###############  Graph 11  ############################# 
        # Volvo tuve
        # with more dynamic obs
        # To be specified in counter-clockwise ordering
        x_min = 0
        x_max = 149
        y_min = 0
        y_max = 16
        safety_dist = 1
        prod_line_start = 11
        
        boundary_coordinates = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)] # AKA drivable area
        boundary_coordinates.reverse()

        # To be specified in clock-wise ordering
        obstacle_list = [
                         [(4, 3.5), (4, 8.5), (83, 8.5), (83, 3.5)], 
                         [(103.5, 7), (103.5, 8.5), (112.5, 8.5), (112.5, 7)], 
                         [(87.5, 3.5), (87.5, 8.5), (100, 8.5), (100, 7), (89, 7), (89, 5), (142, 5), (142, 7), (116, 7), (116, 8.5), (144, 8.5), (144, 3.5), ]
                        ]

        dynamic_obstacles = [
            # 0.5 x 0.5 m dyn obs in narrow path
            DynamicObstacle([(101.5, 12.0), (101.5, 12.5), (102, 12.5), (102.0, 12.0)], (0, 6), (200, 15), (0, -0.03), config.n_hor), 
            DynamicObstacle([(31.5, 9.5), (31.5, 10.5), (32, 10.5), (32.0, 9.5)], (0, 6), (200, 15), (0.1, 0), config.n_hor), 
            DynamicObstacle([(51.5, 9.75), (51.5, 10.25), (52, 10.25), (52.0, 9.75)], (0, 6), (200, 15), (0.1, 0), config.n_hor), 
            DynamicObstacle([(1.5, 9.75), (1.5, 10.25), (12, 10.25), (12.0, 9.75)], (0, 6), (200, 15), (0.1, 0), config.n_hor), 
            #DynamicObstacle([(30.5, 9.75), (30.5, 10.25), (31, 10.25), (31.0, 9.25)], (0, 6), (114, 15), (0.1, 0), config.n_hor), 
        ]
        # Add motor transporter
        motor_from_prod_line = 1.9
        dist_between_motors = 1.5
        dy = motor_from_prod_line+prod_line_start
        dx = 0
        for i in range(int(x_max//(2.3+dist_between_motors))):
            # motor = DynamicObstacle([(dx, dy+0.5), (dx, dy+1), (dx+1.5, dy+1), (dx+1.5, dy+1.3), (dx+2.3, dy+1.3), (dx+2.3, dy), (dx+1.5, dy), (dx+1.5, dy+0.5)], (dx, 0), (dx+2.3+dist_between_motors, 100000), (0.1, 0))
            # dynamic_obstacles.append(motor)
            obstacle_list.append([(dx, dy+0.5), (dx, dy+1), (dx+1.5, dy+1), (dx+1.5, dy+1.3), (dx+2.3, dy+1.3), (dx+2.3, dy), (dx+1.5, dy), (dx+1.5, dy+0.5)])
            dx += dist_between_motors+2.3


        unexpected_obstacle = []

        # dynamic_obstacles = [] 

        # start = (safety_dist, 1.1+safety_dist, 0)
        # end = (safety_dist, prod_line_start-safety_dist, 0)
        start  = (114.25, 6, 3.14)
        end = (24, 11, 0)

        nodes = one_dir_nodes((x_min+safety_dist, 1.1+safety_dist), (x_max-safety_dist, 1.1+safety_dist))
        
        nodes =  {**nodes, **one_dir_nodes((x_max-safety_dist, 1.1+safety_dist+1), (x_max-safety_dist, prod_line_start-safety_dist))}
        nodes[(x_max-safety_dist, 1.1+safety_dist)].neighbours.append(nodes[(x_max-safety_dist, 1.1+safety_dist+1)])

        nodes =  {**nodes, **one_dir_nodes((x_max-1-safety_dist, prod_line_start-safety_dist), (safety_dist, prod_line_start-safety_dist))}
        nodes[(x_max-safety_dist, prod_line_start-safety_dist)].neighbours.append(nodes[(x_max-1-safety_dist, prod_line_start-safety_dist)])

        nodes =  {**nodes, **one_dir_nodes((safety_dist, prod_line_start-1-safety_dist), (safety_dist, 1.1+safety_dist+1))}
        nodes[(safety_dist, prod_line_start-safety_dist)].neighbours.append(nodes[(safety_dist, prod_line_start-1-safety_dist)])
        nodes[(safety_dist, 1.1+safety_dist+1)].neighbours.append(nodes[(safety_dist, 1.1+safety_dist)])

        # Add kiting area nodes
        nodes =  {**nodes, **one_dir_nodes((114.25, prod_line_start-safety_dist), (114.25, 6))}
        nodes[(114, prod_line_start-1)].neighbours.append(nodes[(114.25, prod_line_start-safety_dist)])

        nodes =  {**nodes, **one_dir_nodes((113.25, 6), (101.75, 6))}
        nodes[(114.25, 6)].neighbours.append(nodes[(113.25, 6)])

        nodes =  {**nodes, **one_dir_nodes((101.75, 7), (101.75, prod_line_start-safety_dist))}
        nodes[(101.75, 6)].neighbours.append(nodes[(101.75, 7)])
        nodes[(101.75, prod_line_start-safety_dist)].neighbours.append(nodes[(101, prod_line_start-safety_dist)])

        # For demo
        nodes[(24, 11)] = Node((24, 11), [])
        nodes[(24, 10)].neighbours.append(nodes[(24, 11)])
        ####

        nodes = list(nodes.values())


        # for i, node  in enumerate(nodes):
        #     x, y = node.coord
        #     for j in range(i+1, len(nodes)):
        #         node_ = nodes[j]
        #         x_, y_ = node_.coord
        #         if (x==x_ or y==y_) and abs(x-x_ + y-y_)<2:
        #             node.neighbours.append(node_)
        #             node_.neighbours.append(node)

        start_master = (114, 7, 3.14) 
        end_coupling_master = start
        end_formation_master = end 
        end_decoupling_master = (24, 12, 0)
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (116,6,3.14)
        end_coupling_slave = (115.25, 6, 3.14)
        end_formation_slave = (23, 11, 3.14)
        end_decoupling_slave = (23, 12, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))
        #self.graphs.append(Graph(boundary_coordinates, obstacle_list, start, end, nodes, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))


        ############### Graph 12 ############################# 
        # Dynamic obs
        # One path blocked
        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(1, 0), (1, 9), (-8.0, 9), (-8, 0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = [[(-6,2), (-6,7), (-1,7), (-1,2)]]

        unexpected_obstacle = [UnexpectedObstacle([(-1e-1,5),(-1e-1,6),(1e-1,6),(1e-1,5)],appear_after=10)]

        dynamic_obstacles = []

        def connect_all_nodes(nodes):
            for i, node  in enumerate(nodes):
                x, y = node.coord
                for j in range(i+1, len(nodes)):
                    node_ = nodes[j]
                    x_, y_ = node_.coord
                    if (x==x_ or y==y_) and abs(x-x_ + y-y_)<2:
                        node.neighbours.append(node_)
                        node_.neighbours.append(node)
        
        nodes = [Node((0,i), []) for i in range(0,9)]
        nodes.extend([Node((x,1), []) for x in range(-8,0)])
        nodes.extend([Node((-7,y), []) for y in range(2,9)])
        nodes.extend([Node((x,8), []) for x in range(-6,0)])
        connect_all_nodes(nodes)
        

        end = (0, 8, 0)
        start_master = (0, 2, 1.57) 
        end_coupling_master = start_master
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (0,1,1.57)
        end_coupling_slave = start_slave
        end_formation_slave = (-1, 8, 3.14)
        end_decoupling_slave = (-1, 8, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))
        #self.graphs.append(Graph(boundary_coordinates, obstacle_list, start, end, nodes, unexpected_obstacles=unexpected_obstacle, dyn_obs_list = dynamic_obstacles))



        ############### Graph  13 ############################# 
        # Dynamic obs
        # Start till mål med dynamic obstacle
        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(-1.0, -1), (10.0, -1), (10.0, 50.0), (-1, 50.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = []

        unexpected_obstacle = []

        dynamic_obstacles = [
            DynamicObstacle([(5, 8.0), (5, 10), (5.51, 10), (5.51, 8.0)], (0, 0), (20, 15), (0, -0.1), config.n_hor), 
            DynamicObstacle([(4, 18.0), (4, 19), (6, 19), (6, 18)], (0, 0), (20, 30), (0, -0.05), config.n_hor), 
        ]

        nodes = []
        for i in range(40, -1, -1):
            if nodes == []:
                nodes.append(Node((5, i), []))
            else:
                nodes.append(Node((5, i), [nodes[-1]]))
        

        start = (5, 1, 1.5)
        end = (5, 40, 0)
        start_master = start 
        end_coupling_master = start_master
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,0,1.57)
        end_coupling_slave = start_slave
        end_formation_slave = (5, 39, 0)
        end_decoupling_slave = (5, 39, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))
        #self.graphs.append(Graph(boundary_coordinates, obstacle_list, start, end, nodes, unexpected_obstacles=unexpected_obstacle, dyn_obs_list = dynamic_obstacles))


        ############### Graph  14 ############################# 
        # Start till mål med directed graph
        # Test: various corners >= 90 deg
        # To be specified in counter-clockwise ordering


        # Make a connecting rectangle of nodes
        nodes = []
        #nodes.append(Node((5, 9), []))

        for i in range(-5, 3):
            if nodes == []:
                nodes.append(Node((i, 15), []))
            else:
                nodes.append(Node((i, 15), [nodes[-1]]))
        for i in range(3, 0, -1):
            nodes.append(Node((i, 18-i), [nodes[-1]]))

        for i in range(0, 10):
            nodes.append(Node((i, 18), [nodes[-1]]))

        for i in range(7, 0, -1):
            nodes.append(Node((i+3, i+11), [nodes[-1]]))

        for i in range(3, 10):
            nodes.append(Node((i, 11), [nodes[-1]]))

        for i in range(11, 0, -1):
            nodes.append(Node((10, i), [nodes[-1]]))

        for i in range(1, 10):
            nodes.append(Node((7, i), [nodes[-1]]))

        for i in range(9, 0, -1):
            nodes.append(Node((5, i), [nodes[-1]]))

        #nodes.append(Node((6, 1), [nodes[-1]]))

        #nodes.append(Node((5, 9), [nodes[-1]]))

        boundary_coordinates = [(-10.0, 0.0), (-10.0, 20.0), (12.0, 20.0), (12, 0.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = []

        unexpected_obstacle = []

        dynamic_obstacles = []

 

        start = (5, 2, 1.57)
        # start = (1, 18, 3.14)
        end = (-5, 15, 0)
        start_master = start 
        end_coupling_master = start_master
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,1,1.57)
        end_coupling_slave = start_slave
        end_formation_slave = (-5, 14, 0)
        end_decoupling_slave = (5, 14, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))
        #self.graphs.append(Graph(boundary_coordinates, obstacle_list, start, end, nodes, unexpected_obstacles=unexpected_obstacle, dyn_obs_list = dynamic_obstacles))


        ############### Fifteenth Graph ############################# 
        # Start till mål med directed graph
        # To be specified in counter-clockwise ordering
        # Rectangle with obstacle inside
        # Make a connecting rectangle of nodes
        nodes = []

        for i in range(1, 10):
            if nodes == []:
                nodes.append(Node((6, i), [Node((5, 1), [])]))
            else:
                nodes.append(Node((6, i), [nodes[-1]]))
        for i in range(9, 0, -1):
            if nodes == []:
                nodes.append(Node((4, i), []))
            else:
                nodes.append(Node((4, i), [nodes[-1]]))


        #nodes.append(Node((6, 1), [nodes[-1]]))

        #nodes.append(Node((5, 9), [nodes[-1]]))

        boundary_coordinates = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0, 10.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = [[(4.6, 4), (4.6,8.3), (5.4,8.3), (5.4,4)]]

        unexpected_obstacle = []

        dynamic_obstacles = []

 

        start = (4, 2, 1.57)
        end = (6, 3, 0)
        start_master = start 
        end_coupling_master = start_master
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (4,1,1.57)
        end_coupling_slave = start_slave
        end_formation_slave = (6, 4, 0)
        end_decoupling_slave = (6, 4, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))
        #self.graphs.append(Graph(boundary_coordinates, obstacle_list, start, end, nodes, unexpected_obstacles=unexpected_obstacle, dyn_obs_list = dynamic_obstacles))


        ############### Sixteenth Graph ############################# 
        # Start till mål med directed graph
        # To be specified in counter-clockwise ordering
        # Dyn obs pushing into boundry, mostly affecting future dyn obs

        # Make a connecting rectangle of nodes
        nodes = [Node((x,1), []) for x in range(1,11)]
        connect_all_nodes(nodes)


        boundary_coordinates = [(0.0, 0.0), (0.0, 2.0), (11.0, 2.0), (11, 0.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = []

        unexpected_obstacle = []

        dynamic_obstacles = [DynamicObstacle([(4,-1), (4,-2) ,(4.1,-2), (4.1,-1)], (-100,-100), (100,2), (0, 0.04), config.n_hor)]

 

        start = (2, 1, 0)
        end = (10, 1, 0)
        start_master = start 
        end_coupling_master = start_master
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (1,1,0)
        end_coupling_slave = start_slave
        end_formation_slave = (9, 1, 0)
        end_decoupling_slave = (9, 1, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))
        #self.graphs.append(Graph(boundary_coordinates, obstacle_list, start, end, nodes, unexpected_obstacles=unexpected_obstacle, dyn_obs_list = dynamic_obstacles))

        
        ############### Seventeenth Graph ############################# 
        # Obstacle ahead where a lane change is neat but a faster obstacle comes behind and prevents lane change

        # Make a connecting rectangle of nodes
        nodes = [Node((5,y), []) for y in range(1,20)]
        connect_all_nodes(nodes)


        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(0.0, 0.0), (6, 0), (6, 20.0), (0, 20.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = [[(0,-1),(6,-1),(6,-2),(0,-2)], [(0,-1),(-1,-1),(-1,20),(0,20)], [(0,20),(0,21),(6,21),(6,20)], [(6,20),(7,20),(7,0),(6,0)]]

        unexpected_obstacle = [UnexpectedObstacle([(4,11), (6-0.5,11), (6-0.5,10), (4,10)])]

        dynamic_obstacles = [DynamicObstacle([(1,0), (3,0), (3,-1), (1,-1)], (2,-0.5), (2,25), (0,0.15), config.n_hor)]

 

        start = (5, 6.8, 1.57)
        end = (5, 19, 0)
        start_master = start 
        end_coupling_master = (5, 7, 1.57)
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,5.8,1.57)
        end_coupling_slave = (5,6,1.57)
        end_formation_slave = (5, 18, 0)
        end_decoupling_slave = (5, 18, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))
        #self.graphs.append(Graph(boundary_coordinates, obstacle_list, start, end, nodes, unexpected_obstacles=unexpected_obstacle, dyn_obs_list = dynamic_obstacles))

        ############### Eigthteenth Graph ############################# 
        # Dynamic obstacle ahead where a lane change is neat but a faster obstacle comes behind and prevents lane change

        # Make a connecting rectangle of nodes
        nodes = [Node((5,y), []) for y in range(1,20)]
        connect_all_nodes(nodes)


        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(0.0, 0.0), (6, 0), (6, 20.0), (0, 20.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = [[(0,-1),(6,-1),(6,-2),(0,-2)], [(0,-1),(-1,-1),(-1,20),(0,20)], [(0,20),(0,21),(6,21),(6,20)], [(6,20),(7,20),(7,0),(6,0)]]

        unexpected_obstacle = []

        dynamic_obstacles = [DynamicObstacle([(4,10), (6-0.5,10), (6-0.5,9), (4,9)], (-5,-10.5), (5,100), (0,0.02), config.n_hor), DynamicObstacle([(1,0), (3,0), (3,-1), (1,-1)], (2,-0.5), (2,25), (0,0.15), config.n_hor)]

 

        start = (5, 4, 1.57)
        end = (5, 19, 0)
        start_master = start 
        end_coupling_master = start_master
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,3,1.57)
        end_coupling_slave = start_slave
        end_formation_slave = (5, 18, 0)
        end_decoupling_slave = (5, 18, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))


        ############### Nineteenth Graph ############################# 
        # Test everything that is hard without obstacles
        # 45 deg right, 135 deg left , 90 up
        # Make a connecting rectangle of nodes
        nodes = [Node((5,y), []) for y in range(1,16)]
        for i in range(5):
            prev_node = nodes[-1]
            dx = 7/5
            dy = 1
            node = Node((nodes[-1].coord[0]+dx, nodes[-1].coord[1]+dy), neighbours=[])
            prev_node.neighbours.append(node)
            nodes.append(node)
        
        for x in range(12,2,-1):
            nodes.append(Node((x,20), []))

        nodes.extend([Node((3,y), []) for y in range(21,28)])
        connect_all_nodes(nodes)
        

        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(0.0, 0.0), (15, 0), (15, 30.0), (0, 30.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = []

        unexpected_obstacle = []

        dynamic_obstacles = []

 

        start = (5, 2, 1.57)
        end = (3, 27, 0)
        start_master = start 
        end_coupling_master = start_master
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,1,1.57)
        end_coupling_slave = start_slave
        end_formation_slave = (3, 26, 0)
        end_decoupling_slave = end_formation_slave
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))


        ############### Twentieth Graph ############################# 
        # Test everything that is hard with obstacles at the inside of corners
        # 45 deg right, 135 deg left , 90 up
        # Make a connecting rectangle of nodes
        nodes = [Node((5,y), []) for y in range(1,16)]
        for i in range(5):
            prev_node = nodes[-1]
            dx = 7/5
            dy = 1
            node = Node((nodes[-1].coord[0]+dx, nodes[-1].coord[1]+dy), neighbours=[])
            prev_node.neighbours.append(node)
            nodes.append(node)
        
        for x in range(12,2,-1):
            nodes.append(Node((x,20), []))

        nodes.extend([Node((3,y), []) for y in range(21,28)])
        connect_all_nodes(nodes)
        

        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(0.0, 0.0), (15, 0), (15, 30.0), (0, 30.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = [[(5.5,13), (5.5,14.7), (6,15.1)],
                        [(9,19.5), (10.4,19.5), (9,18.5)],
                        [(3.5,20.5), (3.5,21), (4,21), (4,20.5)],]

        unexpected_obstacle = []

        dynamic_obstacles = []

 
        start = (5, 2, 1.57)
        end = (3, 27, 0)
        start_master = start 
        end_coupling_master = start_master
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,1,1.57)
        end_coupling_slave = start_slave
        end_formation_slave = (3, 26, 0)
        end_decoupling_slave = end_formation_slave
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))


        ############### Twentyfirst Graph ############################# 
        # Test everything that is hard with obstacles at the inside of corners

        # Make a connecting rectangle of nodes
        nodes = [Node((5,y), []) for y in range(1,16)]
        for i in range(5):
            prev_node = nodes[-1]
            dx = 7/5
            dy = 1
            node = Node((nodes[-1].coord[0]+dx, nodes[-1].coord[1]+dy), neighbours=[])
            prev_node.neighbours.append(node)
            nodes.append(node)
        
        for x in range(12,2,-1):
            nodes.append(Node((x,20), []))

        nodes.extend([Node((3,y), []) for y in range(21,28)])
        connect_all_nodes(nodes)
        

        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(0.0, 0.0), (15, 0), (15, 30.0), (0, 30.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = [[(5.5,13), (5.5,14.7), (6,15.1)],
                        [(4.3,13), (4.3, 17), (6.5,17), (4.3,15.5)],
                        [(9,19.5), (10.4,19.5), (9,18.5)],
                        [(11,21), (13,21), (13,19), (11, 18), (13,20.3)],
                        [(3.5,20.5), (3.5,21), (4,21), (4,20.5)],
                        [(2,21), (2.3,21), (2.3,20), (2,20)],]

        unexpected_obstacle = []

        dynamic_obstacles = []

 
        start = (5, 2, 1.57)
        end = (3, 27, 0)
        start_master = start 
        end_coupling_master = start_master
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,1,1.57)
        end_coupling_slave = start_slave
        end_formation_slave = (3, 26, 0)
        end_decoupling_slave = end_formation_slave
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))

        ############### Twentysecondth Graph ############################# 
        # Test everything that is hard with obstacles at the inside of corners and dynamic obstacles and unexpected obstacles

        # Make a connecting rectangle of nodes
        nodes = [Node((5,y), []) for y in range(1,16)]
        for i in range(5):
            prev_node = nodes[-1]
            dx = 7/5
            dy = 1
            node = Node((nodes[-1].coord[0]+dx, nodes[-1].coord[1]+dy), neighbours=[])
            prev_node.neighbours.append(node)
            nodes.append(node)
        
        for x in range(12,2,-1):
            nodes.append(Node((x,20), []))

        nodes.extend([Node((3,y), []) for y in range(21,28)])
        connect_all_nodes(nodes)
        

        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(0.0, 0.0), (15, 0), (15, 30.0), (0, 30.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = [[(5.5,13), (5.5,14.7), (6,15.1)],
                        [(4.3,13), (4.3, 17), (6.5,17), (4.3,15.5)],
                        [(9,19.5), (10.4,19.5), (9,18.5)],
                        [(11,21), (13,21), (13,19), (11, 18), (13,20.3)],
                        [(3.5,20.5), (3.5,21), (4,21), (4,20.5)],
                        [(2,21), (2.3,21), (2.3,20), (2,20)],]

        unexpected_obstacle = [UnexpectedObstacle([(4,10), (4,11), (6,11), (6,10)])]

        # dynamic_obstacles = [DynamicObstacle([(2,5), (2,6), (3,6), (3,5)], (0,-100), (10,100), (0.03,0), config.n_hor),
        #                     DynamicObstacle([(6,17), (6,18), (7,18), (7,17)], (-100,17), (10,23), (0.00,0.1), config.n_hor),]
        dynamic_obstacles = []

 
        start = (5, 2, 1.57)
        end = (3, 27, 0)
        start_master = start 
        end_coupling_master = start_master
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,1,1.57)
        end_coupling_slave = start_slave
        end_formation_slave = (3, 26, 0)
        end_decoupling_slave = end_formation_slave
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))


        ############### Twentythird Graph ############################# 
        # Test everything that is hard with obstacles at the inside of corners and dynamic obstacles

        # Make a connecting rectangle of nodes
        nodes = [Node((5,y), []) for y in range(1,16)]
        for i in range(5):
            prev_node = nodes[-1]
            dx = 7/5
            dy = 1
            node = Node((nodes[-1].coord[0]+dx, nodes[-1].coord[1]+dy), neighbours=[])
            prev_node.neighbours.append(node)
            nodes.append(node)
        
        for x in range(12,2,-1):
            nodes.append(Node((x,20), []))

        nodes.extend([Node((3,y), []) for y in range(21,28)])
        connect_all_nodes(nodes)
        

        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(0.0, 0.0), (15, 0), (15, 30.0), (0, 30.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = [[(5.5,13), (5.5,14.7), (6,15.1)],
                        [(4.3,13), (4.3, 17), (6.5,17), (4.3,15.5)],
                        [(9,19.5), (10.4,19.5), (9,18.5)],
                        [(11,21), (13,21), (13,19), (11, 18), (13,20.3)],
                        [(3.5,20.5), (3.5,21), (4,21), (4,20.5)],
                        [(2,21), (2.3,21), (2.3,20), (2,20)],]

        unexpected_obstacle = [UnexpectedObstacle([(4,10), (4,11), (6,11), (6,10)])]

        dynamic_obstacles = [DynamicObstacle([(2,5), (2,6), (3,6), (3,5)], (0,-100), (10,100), (0.03,0), config.n_hor),
                            DynamicObstacle([(6,17), (6,18), (7,18), (7,17)], (-100,17), (10,23), (0.00,0.1), config.n_hor),]

 
        start_master = (4.9,1,0)
        end_coupling_master = (5, 3, 1.57)
        end_formation_master = (3, 27, 0)
        end_decoupling_master = (2, 25, 0)
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (7,3,0)
        end_coupling_slave = (5, 2, 1.57)
        end_formation_slave = (4, 27, 0)
        end_decoupling_slave = (2, 23, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))
        
        

        ############### Twentyfourth Graph ############################# 
        # Narrow passage with dynamic obstacle moving away from us

        nodes = [Node((x,1), []) for x in range(1,20)]
        connect_all_nodes(nodes)

        boundary_coordinates = [(0.0, 0.0), (0.0, 2.0), (20.0, 2.0), (20, 0.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = [[(0,2.5), (0,3), (20,3), (20,2.5)],
                        [(21,2.5), (22,2.5), (22,0), (21,0)],
                        [(21,0), (21,-1), (0,-1), (0,0)],
                        [(0,0), (-1,0), (-1,3), (0,3)]]

        unexpected_obstacle = []

        dynamic_obstacles = [DynamicObstacle([(4,0.5), (4,1.5) ,(4.1,1.5), (4.1,0.5)], (-100,-100), (100,2), (0.03, 0), config.n_hor)]

 

        start = (2, 1, 0)
        end = (19, 1, 0)
        start_master = start 
        end_coupling_master = start_master
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (1,1,0)
        end_coupling_slave = start_slave
        end_formation_slave = (18, 1, 0)
        end_decoupling_slave = end_formation_slave
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))

 


        ############### Twentyfifth Graph ############################# 
        # Narrow passage with dynamic obstacle moving towards us

        nodes = [Node((x,1), []) for x in range(1,20)]
        connect_all_nodes(nodes)

        boundary_coordinates = [(0.0, 0.0), (0.0, 2.0), (20.0, 2.0), (20, 0.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = [[(0,2.5), (0,3), (20,3), (20,2.5)],
                        [(21,2.5), (22,2.5), (22,0), (21,0)],
                        [(21,0), (21,-1), (0,-1), (0,0)],
                        [(0,0), (-1,0), (-1,3), (0,3)]]

        unexpected_obstacle = []

        dynamic_obstacles = [DynamicObstacle([(10,0.9), (10,1.6) ,(11.1,1.6), (11.1,0.9)], (-100,-100), (100,2), (-0.03, 0), config.n_hor)]

 


        start = (2, 1, 0)
        end = (19, 1, 0)
        start_master = start 
        end_coupling_master = start_master
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (1,1,0)
        end_coupling_slave = start_slave
        end_formation_slave = (18, 1, 0)
        end_decoupling_slave = end_formation_slave
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))
 


        ############### Graph  27 ############################# 
        # Dynamic obs
        # L shaped objects
        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(-1.0, -1), (30.0, -1), (30.0, 30.0), (-1, 30.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = [[(1,4),(1,9),(5,9),(5,8),(3,8),(3,4)]]

        unexpected_obstacle = []

        dynamic_obstacles = [
            # comment in this when new obs avoid is in place
            #DynamicObstacle([(5, 8.0), (5, 10), (5.51, 10), (5.51, 8.0)], (0, 0), (20, 15), (0, -0.1), config.n_hor), 
            DynamicObstacle([(4,24.0),(4,29),(8,29),(8,28),(6,18),(6,24)], (0, 0), (20, 30), (0, -0.05), config.n_hor), 
        ]

        nodes = []
        for i in range(30, -1, -1):
            if nodes == []:
                nodes.append(Node((5, i), []))
            else:
                nodes.append(Node((5, i), [nodes[-1]]))
        

        start = (5, 1, 1.5)
        end = (5, 30, 0)
        start_master = start 
        end_coupling_master = start_master
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,0,0)
        end_coupling_slave = start_slave
        end_formation_slave = (5, 29, 0)
        end_decoupling_slave = end_formation_slave
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))

        ############### Graph  28 ############################# 
        # Bounds testing
        # nonconvex bounds
        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(-1.0, -1), (30.0, -1), (30.0, 4.0),(29.0, 5.0),(30.0, 6.0), (28,8), (30,10),(30,20),(20,20),(19,17),(18,20), (-1, 20.0),(-1,10),(5,10),(5,9),(-1,9)] # AKA drivable area


        # To be specified in clock-wise ordering
        obstacle_list = []
        unexpected_obstacle = []

        dynamic_obstacles = [
        ]

        nodes = []
        for i in range(30, -1, -1):
            if nodes == []:
                nodes.append(Node((5, i), []))
            else:
                nodes.append(Node((5, i), [nodes[-1]]))
        

        start = (5, 1, 1.5)
        end = (5, 30, 0)
        start_master = start 
        end_coupling_master = start_master
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,0,0)
        end_coupling_slave = start_slave
        end_formation_slave = (5, 29, 0)
        end_decoupling_slave = end_formation_slave
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))

        ############### Graph  29 ############################# 
        # Zhes map
        # To be specified in counter-clockwise ordering

        boundary_coordinates = [(0.0, 0), (30.0, 0), (30.0, 30.0), (0, 30.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = []
        corner_ll = (9,10)
        height = 5
        width = 5
        def get_rect(corner_ll, height, width):
            rect = [corner_ll,tuple(map(sum, zip(corner_ll, (0,height)))), tuple(map(sum, zip(corner_ll, (width,height)))),tuple(map(sum, zip(corner_ll, (width,0))))]
            return rect

        rect_1 = get_rect(corner_ll, height, width)
        rect_2 = get_rect((16,10), height, width)
        rect_3 = get_rect((16,2), height, width)
        rect_4 = get_rect((9,2), height, width)
        obstacle_list=[rect_1,rect_2, rect_3, rect_4]
        unexpected_obstacle = []

        dynamic_obstacles = [DynamicObstacle([(2,7), (2,10) ,(6,10), (6,7)], (-100,-100), (100,100), (0.1, 0), config.n_hor)]
        nodes = []
        for i in range(30, -1, -1):
            if nodes == []:
                nodes.append(Node((15, i), []))
            else:
                nodes.append(Node((15, i), [nodes[-1]]))
        

        start_master = (15,2,1.57)
        end_coupling_master = (15, 2, 1.57)
        end_formation_master = (15, 25, 0)
        end_decoupling_master = (14, 25, 0)
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (15,1,1.57)
        end_coupling_slave = (15, 1, 1.57)
        end_formation_slave = (15, 24, 0)
        end_decoupling_slave = (14, 23, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle,dyn_obs_list = dynamic_obstacles))

        
        ############### Graph  30 ############################# 
        # A graph where the master has to adjust to the slave for the system to not crash
        # To be specified in counter-clockwise ordering

        boundary_coordinates = [(-10, -10), (10.0, -10), (10, 10), (-10, 10)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = [[(-10,3), (-10,5.3), (5,5.3), (5,3)]]
        unexpected_obstacle = []
        dynamic_obstacles = []
        nodes = []
        for i in range(7, -10, -1):
            if nodes == []:
                nodes.append(Node((5.6, i), []))
            else:
                nodes.append(Node((5.6, i), [nodes[-1]]))
        connect_all_nodes(nodes)
        

        start_master = (5.6,6,-1.57)
        end_coupling_master = (5.6,6,-1.57)
        end_formation_master = (5.6, -9, 0)
        end_decoupling_master = (5.6, -9, 0)
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (0.6,6,-1.57)
        end_coupling_slave = (0.6,6,-1.57)
        # start_slave = (0.5,6,0)
        # end_coupling_slave = (0.5,6,0)
        end_formation_slave = (0.6, -9, 0)
        end_decoupling_slave = (0.6, -9, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle,dyn_obs_list = dynamic_obstacles))

        
        ############### Graph  31 ############################# 
        # Dynamic obs
        # Dyn obs with pos controller
        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(-1.0, -1), (30.0, -1), (30.0, 30.0), (-1, 30.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = []

        unexpected_obstacle = []

        dynamic_obstacles = [
            # comment in this when new obs avoid is in place
            DynamicObstacle([(5, 12.0), (5, 14), (5.51, 14), (5.51, 12.0)], (0, 0), (20, 15), (0, -0.1), config.n_hor), 
            #DynamicObstacle([(5, 8.0), (5, 10), (5.51, 10), (5.51, 8.0)], (0, 0), (20, 15), (0, -0.1)), 
            #DynamicObstacle([(4,24.0),(4,29),(8,29),(8,28),(6,18),(6,24)], (0, 0), (20, 30), (0, -0.05)), 
        ]

        nodes = []
        for i in range(30, -1, -1):
            if nodes == []:
                nodes.append(Node((5, i), []))
            else:
                nodes.append(Node((5, i), [nodes[-1]]))
        

        start = (5, 1, 1.5)
        end = (5, 30, 0)
        start_master = start 
        end_coupling_master = end
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (5,0,0)
        end_coupling_slave = (5,29,0)
        end_formation_slave = (5, 29, 0)
        end_decoupling_slave = end_formation_slave
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))


        ############### 32 Graph ############################# 
        # Test everything that is hard with obstacles at the inside of corners and dynamic obstacles and pointy triangles

        # Make a connecting rectangle of nodes
        nodes = [Node((5,y), []) for y in range(1,16)]
        for i in range(5):
            prev_node = nodes[-1]
            dx = 7/5
            dy = 1
            node = Node((nodes[-1].coord[0]+dx, nodes[-1].coord[1]+dy), neighbours=[])
            prev_node.neighbours.append(node)
            nodes.append(node)
        
        for x in range(12,2,-1):
            nodes.append(Node((x,20), []))

        nodes.extend([Node((3,y), []) for y in range(21,28)])
        connect_all_nodes(nodes)
        

        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(0.0, 0.0), (15, 0), (15, 30.0), (0, 30.0)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = [[(5,13), (5,15), (6,15.7)],
                        [(4.9,13), (4, 13), (4,17), (5.6,15.5), (4.9,15)],
                        [(10, 18.6), (10,20), (12,20)],
                        [(11,21), (13,21), (13,19), (11, 18), (13,20.3)],
                        [(3,20), (3,21), (4,21), (4,20)],
                        [(2,21), (2.7,21), (2.7,20), (2,20)],
                        ]

        unexpected_obstacle = []

        dynamic_obstacles = [DynamicObstacle([(2,5), (2,6), (3,6), (3,5)], (0,-100), (10,100), (0.03,0), config.n_hor),
                            DynamicObstacle([(6,17), (6,18), (7,18), (7,17)], (-100,17), (10,23), (0.00,0.1), config.n_hor),]

 
        start_master = (4.9,1,0)
        end_coupling_master = (5, 3, 1.57)
        end_formation_master = (3, 27, 0)
        end_decoupling_master = (2, 25, 0)
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (7,3,0)
        end_coupling_slave = (5, 2, 1.57)
        end_formation_slave = (6, 27, 0)
        end_decoupling_slave = (2, 23, 0)
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))
        

        ############### Graph  33 ############################# 
        # Dynamic obs
        # Dyn obs with pos controller
        # To be specified in counter-clockwise ordering
        boundary_coordinates = [(0, 0), (12, 0), (12.0, 5.97), (0, 5.97)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = []

        unexpected_obstacle = []

        dynamic_obstacles = [
            # comment in this when new obs avoid is in place
            #DynamicObstacle([(5, 12.0), (5, 14), (5.51, 14), (5.51, 12.0)], (0, 0), (20, 15), (0, -0.1), config.n_hor), 
            #DynamicObstacle([(5, 8.0), (5, 10), (5.51, 10), (5.51, 8.0)], (0, 0), (20, 15), (0, -0.1)), 
            #DynamicObstacle([(4,24.0),(4,29),(8,29),(8,28),(6,18),(6,24)], (0, 0), (20, 30), (0, -0.05)), 
        ]

        nodes = []
        for i in range(10, 0, -1):
            if nodes == []:
                nodes.append(Node((i, 3), []))
            else:
                nodes.append(Node((i, 3), [nodes[-1]]))
        

        start = (2, 4, 1.5)
        end = (10, 3, 0)
        start_master = start 
        end_coupling_master = (2,3,0)
        end_formation_master = end 
        end_decoupling_master = end
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (1,2,0)
        end_coupling_slave = (1,3,0)
        end_formation_slave = (10, 3, 0)
        end_decoupling_slave = end_formation_slave
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))



        ############### Graph  34 ############################# 
        # Dynamic obs
        # Dyn obs with pos controller
        # To be specified in counter-clockwise ordering
        # Pilot plant 

        boundary_coordinates = [(0, 0), (12, 0), (12.0, 5.97), (0, 5.97)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = []

        unexpected_obstacle = []
        # x y
        # 3.47 2.98 
        # 8.44 
                # 5
        start_pos = (3.47,3)
        dynamic_obstacles = [
            # comment in this when new obs avoid is in place
            #DynamicObstacle([(5, 12.0), (5, 14), (5.51, 14), (5.51, 12.0)], (0, 0), (20, 15), (0, -0.1), config.n_hor), 
            #DynamicObstacle([(5, 8.0), (5, 10), (5.51, 10), (5.51, 8.0)], (0, 0), (20, 15), (0, -0.1)), 
            #DynamicObstacle([(4,24.0),(4,29),(8,29),(8,28),(6,18),(6,24)], (0, 0), (20, 30), (0, -0.05)), 
        ]

        nodes = []
        for i in range(3, 5):
            nodes.append(Node((round(3.47,2), i), []))


        for i in range(0,6):
            nodes.append(Node((round(3.47 + i ,2), 5), []))
        
        for i in range(4, 2, -1):
            nodes.append(Node((round(8.47,2), i), []))

        for i in range(4, 0, -1):
            nodes.append(Node((round(3.47 + i ,2), 3), []))

        connect_all_nodes(nodes)
        start_master = (2, 1, 0)
        end_coupling_master = (3.47, 3, 1.5)
        end_formation_master = (8.47, 5, 0) 
        end_decoupling_master = end_formation_master
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (1,2,0)
        end_coupling_slave = (2.47,3,0)
        end_formation_slave = (9.47, 4, 0)
        end_decoupling_slave = end_formation_slave
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))


        ############### Graph  35 ############################# 
        # Dynamic obs
        # Dyn obs with pos controller
        # To be specified in counter-clockwise ordering
        # Pilot plant 

        boundary_coordinates = [(0, 0), (12, 0), (12.0, 5.97), (0, 5.97)] # AKA drivable area

        # To be specified in clock-wise ordering
        obstacle_list = []

        unexpected_obstacle = []
        unexpected_obstacle = [UnexpectedObstacle([(4,4), (5,4), (5,5), (5,4)])]
        start_pos = (3.47,3)
        dynamic_obstacles = [
            # comment in this when new obs avoid is in place
            #DynamicObstacle([(5, 12.0), (5, 14), (5.51, 14), (5.51, 12.0)], (0, 0), (20, 15), (0, -0.1), config.n_hor), 
            #DynamicObstacle([(5, 8.0), (5, 10), (5.51, 10), (5.51, 8.0)], (0, 0), (20, 15), (0, -0.1)), 
            #DynamicObstacle([(4,24.0),(4,29),(8,29),(8,28),(6,18),(6,24)], (0, 0), (20, 30), (0, -0.05)), 
        ]

        nodes = []
        for i in range(3, 5):
            nodes.append(Node((round(3.47,2), i), []))


        for i in range(0,6):
            nodes.append(Node((round(3.47 + i ,2), 5), []))
        
        for i in range(4, 2, -1):
            nodes.append(Node((round(8.47,2), i), []))

        for i in range(4, 0, -1):
            nodes.append(Node((round(3.47 + i ,2), 3), []))

        connect_all_nodes(nodes)
        start_master = (2, 1, 0)
        end_coupling_master = (3.47, 3, 1.5)
        end_formation_master = (8.47, 5, 0) 
        end_decoupling_master = end_formation_master
        positions_master = PositionsFlow(start_master, end_coupling_master, end_formation_master, end_decoupling_master)
        start_slave = (1,2,0)
        end_coupling_slave = (2.47,3,0)
        end_formation_slave = (9.47, 4, 0)
        end_decoupling_slave = end_formation_slave
        positions_slave = PositionsFlow(start_slave, end_coupling_slave, end_formation_slave, end_decoupling_slave)
        self.graphs.append(Graph(boundary_coordinates, obstacle_list, nodes, positions_master, positions_slave, unexpected_obstacles=unexpected_obstacle, dyn_obs_list=dynamic_obstacles))

        self.min_complexity = 0
        self.max_complexity = len(self.graphs) - 1

    def get_graph(self, complexity):
        if not (self.min_complexity <= complexity <= self.max_complexity):
            raise ValueError(f'Complexity should be in the range {self.min_complexity } - {self.max_complexity}')

        return self.graphs[complexity]


