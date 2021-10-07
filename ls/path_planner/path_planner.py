from extremitypathfinder.extremitypathfinder import PolygonEnvironment
from networkx.algorithms import boundary
import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry
from path_planner.a_star_testing import realGraph
from path_planner.obstacle_handler import ObstacleHandler
from time import perf_counter_ns

class PathPlanner:
    """Class which does the global path planning and local path planning. 
    By global path planning a-star is used to find the shortest path along 
    some predefined nodes. Local path finding utilizes visibility graph to 
    find the shortest path around unexpected obstacles.
    """

    def __init__(self, config, plot_config, plot_queues, map_data):
        self.config = config
        self.plot_config = plot_config
        self.plot_queues = plot_queues
        self.obs_handler = ObstacleHandler(config, plot_config, plot_queues)
        self.graph = None
        self.path = None
        self.rg = None
        self.path_shapely = []
        self.traversed_planned_path = [] # ONLY FOR SIMULATION

        self.times = {'global':[], 'local':[], 'unexpected': [], 'plot':[]}

        self.map_data = map_data

        # Convert boundry to a convex hull and the nonconvex parts as obstacles 
        extra_obs, new_bounds = self.obs_handler.bounds_to_obstacles_and_convex_hull(map_data.boundary_coordinates)

        # Add the extra obstacles to graph if they exist
        obstacle_list = map_data.obstacle_list
        if not extra_obs == None:
            for ob in extra_obs:
                obstacle_list.append(ob)
        
        # Replace with convex bounds
        boundary_coords = map_data.boundary_coordinates
        boundary_coords = new_bounds
        
        self.obs_handler.static_obstacles = obstacle_list
        self.obs_handler.dynamic_obstacles = [ob.future_polygons for ob in map_data.dyn_obs_update()] 
        self.obs_handler.boundry = boundary_coords
        self.rg = realGraph()
        self.rg.build_from_nodes(map_data.nodes)
        

    def visability_graph(self, boundry, obstacles, start_pos, end_pos):
        self.env = PolygonEnvironment()
        # Give obstacles and boundaries to environment
        self.env.store(boundry, obstacles)
        # Prepare the visibility graph 
        self.env.prepare()

        path, distance = self.env.find_shortest_path(start_pos, end_pos)
        # If path is empty then there is no path
        if path == []:
            raise Exception("No path to be found")

        return path

    def plot(self):
        start_time = perf_counter_ns()
        # Plot nodes
        data = [[x for x, y in self.rg.G.nodes], [y for x, y in self.rg.G.nodes]]
        try: self.plot_queues['nodes'].put_nowait(data)
        except: pass

        # Plot Planned path
        data = [[x for x, y in self.path], [y for x, y in self.path]]
        try: self.plot_queues['planned_path'].put_nowait(data)
        except: pass

        self.obs_handler.plot() 
        self.times['plot'].append(perf_counter_ns() - start_time)

    def global_path_plan(self, start, end):
        """[summary]

        Args:
            start ([type]): [description]
            end ([type]): [description]
        """
        # This has been moved to graph setter
        # self.rg = realGraph()
        # self.rg.build_from_nodes(self.graph.nodes)

        start_time = perf_counter_ns()
        self.path = self.rg.search_astar(start, end)
        self.path_shapely = [geometry.Point(p) for p in self.path]
        self.times['global'].append(perf_counter_ns() - start_time)

    def update_unexpected_obstacles(self, unexpected_obstacles):
        """Updates the unexpected obstacles. If anything updates tell graph to replot.

        Args:
            unexpected_obstacles (list of lists of tuples): list of lists of tuples of the vertices of
                the unexpected objects in form of the vertices of the objects in clock-wise ordering. 
                For example a square would be: [[(x0, y0), (x1, y1), (x2, y2), (x3, y3)]].
        """
        start_time = perf_counter_ns()
        #if self.graph.unexpected_obstacles_exists == unexpected_obstacles:
        stored_obs, _, _ = self.obs_handler.unexpected_obstacles
        if np.all(stored_obs == unexpected_obstacles):
            return False

        self.obs_handler.unexpected_obstacles = unexpected_obstacles

        self.times['unexpected'].append(perf_counter_ns() - start_time)
        return True

    def update_obstacles(self):
        """Advances the dynamic obstacles 1 time step ahead or calls for new information via ros on unexpected and dynamic obstacles.
        """
        unexpected_updated = self.map_data.unexpected_obs_update()
        dyn_updated = self.map_data.dyn_obs_update()
        
        self.obs_handler.dynamic_obstacles = [ob.future_polygons for ob in dyn_updated]
        self.obs_handler.unexpected_obstacles = unexpected_updated

    def local_path_plan(self, x_cur):
        """This method plans a path around unexpected obstacles using visability graphs.
        The update path is stored in self.path

        Args:
            
            x_cur ([list or tuple]): [x, y], (x, y)

        Returns:
            True/False if the path is updated 
        """
        start_time = perf_counter_ns()

        # Find the start and end pos around obstacle
        _, unexpected_obstacles, unexpected_obstacles_shapely = self.obs_handler.unexpected_obstacles
        collision = self.path_through_obs(unexpected_obstacles_shapely)
        if not collision:
            return False

        start_pos_idx, end_pos_idx = self.find_start_end_pos_around_obs(unexpected_obstacles_shapely)

        # Update path with way around obstacle, visibility graph
        boundry = self.obs_handler.boundry
        _, static_obstacles, _ = self.obs_handler.static_obstacles
        all_obs = [[tuple(pos) for pos in ob] for ob in unexpected_obstacles] + [[tuple(pos) for pos in ob] for ob in static_obstacles]
        
        for i in range(len(start_pos_idx)):
            start = self.path[start_pos_idx[i]]
            end = self.path[end_pos_idx[i]]
            path = self.visability_graph(boundry, all_obs, start, end)
            self.path[start_pos_idx[i] : end_pos_idx[i]+1] = path

        # Update the shapely path
        self.path_shapely = [geometry.Point(p) for p in self.path]

        # Save the traversed portion of the path. Only for simulation
        self.save_traversed_planned_path(x_cur)

        self.times['local'].append(perf_counter_ns() - start_time)
        return True

    def save_traversed_planned_path(self, x_cur):
        """ Save the traversed portion of the path. Only for simulation
        """
        x_cur_node_idx = np.argmin(np.linalg.norm(np.array(self.path)-np.array(x_cur[:2]), ord=2, axis=1))

        self.traversed_planned_path.extend(self.path[:x_cur_node_idx+1])
    

    def find_start_end_pos_around_obs(self, unexpected_obstacles):
        """Calculates the start and end nodes around obstacles intersecting the path.

        Args:
            unexpected_obstacles (list of lists of tuples): a list that contains a list of obstacles. Each obstacle is a polygon defined by tuples containing it's vertices.

        Returns:
            idx_of_start_nodes [list]: nodes at which to start to deviate from the original path to avoid obstacles
            idx_of_end_nodes [list]: nodes at which to end the deviation from the original path to avoid obstacles
        """
        
        obstacles = unexpected_obstacles

        # Find the obs which intersect the path
        idx_of_intersecting_obs = []
        for i, ob in enumerate(obstacles):
            for node in self.path_shapely:
                if not ob.intersection(node).is_empty:
                    idx_of_intersecting_obs.append(i)
                    break

        # Go through the path and find unavailable nodes
        unavailable_nodes = []
        for i, node in enumerate(self.path_shapely):
            for j in idx_of_intersecting_obs:
                ob = obstacles[j]
                if ob.exterior.distance(node) <= self.config.distance_to_avoid_unexpected_obstacles and not node == self.path_shapely[-1]:
                    unavailable_nodes.append(i)
                elif not ob.intersection(node).is_empty:
                    unavailable_nodes.append(i)

        # Find start and end nodes
        main_list = np.setdiff1d([x for x in range(len(self.path))], unavailable_nodes, assume_unique=True)
        idx_of_start_nodes = [main_list[i] for i in range(len(main_list)-1) if not main_list[i]+1==main_list[i+1]]
        idx_of_end_nodes = [main_list[i] for i in range(1, len(main_list)) if not main_list[i-1] == main_list[i]-1]

        return idx_of_start_nodes, idx_of_end_nodes

    def path_through_obs(self, unexpected_obstacles):
        """Find if the path goes through any unexpected obstacle

        Args:
            unexpected_obstacles ([type]): [description]

        Returns:
            Boolean: True if blocked, None if not
        """

        for ob in unexpected_obstacles:
            x, y = ob.exterior.xy
            for node in self.path_shapely:
                if not ob.intersection(node).is_empty:
                # If the point is on the boundry it's ok
                    if (node.bounds[0] in x) and (node.bounds[1] in y):
                        continue
                    return True



    