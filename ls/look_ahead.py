from queue import Full
from shapely.geometry import LineString
from path_planner.obstacle_handler import ObstacleHandler
from time import perf_counter_ns

class LookAhead:
    """Class which looks ahead for crashes
    """

    def __init__(self, config, plot_config, verbose, obstacle_handler, plot_queues):
        self.config = config
        self.plot_config = plot_config
        self.verbose = verbose
        self.obstacle_handler: ObstacleHandler = obstacle_handler
        self.plot_queues = plot_queues

        self.times = {'calculate_crash':[], 'plot':[]}
        

        self.print_name = "[LookAhead]"


    def plot(self, trajectory):
        """Plot line between atrs

        Args:
        """
        start_time = perf_counter_ns()

        # Plot new lines
        x_master = trajectory[:, 0]
        y_master = trajectory[:, 1]
        x_slave = trajectory[:, 3]
        y_slave = trajectory[:, 4]

        data = []
        for i in range(x_slave.size): 
            data.append([[x_master[i], y_master[i]], [x_slave[i], y_slave[i]]])
        try:
            self.plot_queues['look_ahead'].put_nowait(data)
        except Full:
            pass



        self.times['plot'].append(perf_counter_ns() - start_time)

    def check_crash_trajectory(self, trajectory):
        """Checks for crash between master and slave atr

        Args:
            x_master ([type]): [description]
            x_slave ([type]): [description]
        """
        start_time = perf_counter_ns()

       
        x_master = trajectory[:, 0]
        y_master = trajectory[:, 1]
        x_slave = trajectory[:, 3]
        y_slave = trajectory[:, 4]

        lines = [LineString([(x_master[i], y_master[i]), (x_slave[i], y_slave[i])]) for i in range(x_slave.size)]

        collision = False
        _, _, static_obstacles_shapely = self.obstacle_handler.static_obstacles
        _, _, unexpected_obstacles_shapely = self.obstacle_handler.unexpected_obstacles
        for line in lines:
            for ob in static_obstacles_shapely:
                collision |= not line.intersection(ob).is_empty  

            for ob in unexpected_obstacles_shapely:
                collision |= not line.intersection(ob).is_empty    

        if collision and self.verbose:
            print(self.print_name + " COLLISION")

        self.times['calculate_crash'].append(perf_counter_ns() - start_time)

        return collision




