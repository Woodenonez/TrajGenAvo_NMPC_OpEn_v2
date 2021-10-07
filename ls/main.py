from trajectory_generator import TrajectoryGenerator
from path_planner.graphs import Graphs
from utils.config import Configurator
from pathlib import Path
import os
import yaml
from ATRS import ATRS
from utils.plotter import start_plotter
from mpc.mpc_generator import MpcModule
from utils.config import Configurator, SolverParams, Weights

def init(verbose=False, build_solver=False, aut_test_config=None, self_destruct=False, graph_complexity = 1):
        
        file_path = Path(__file__)

        base_fn = 'base.yaml'
        yaml_fp = os.path.join(str(file_path.parent.parent), 'configs', base_fn)
        configurator = Configurator(yaml_fp)
        base = configurator.configurate()

        weights_fn = 'line_follow_weights.yaml'
        yaml_fp = os.path.join(str(file_path.parent.parent), 'configs', weights_fn)
        configurator = Weights(yaml_fp)
        line_follow_weights = configurator.configurate()

        weights_fn = 'line_follow_weights_aggressive.yaml'
        yaml_fp = os.path.join(str(file_path.parent.parent), 'configs', weights_fn)
        configurator = Weights(yaml_fp)
        line_follow_weights_aggressive = configurator.configurate()

        weights_fn = 'traj_follow_weights.yaml'
        yaml_fp = os.path.join(str(file_path.parent.parent), 'configs', weights_fn)
        configurator = Weights(yaml_fp)
        traj_follow_weights = configurator.configurate()

        weights_fn = 'traj_follow_weights_aggressive.yaml'
        yaml_fp = os.path.join(str(file_path.parent.parent), 'configs', weights_fn)
        configurator = Weights(yaml_fp)
        traj_follow_weights_aggressive = configurator.configurate()

        weights_fn = 'pos_goal_weights.yaml'
        yaml_fp = os.path.join(str(file_path.parent.parent), 'configs', weights_fn)
        configurator = Weights(yaml_fp)
        pos_goal_weights = configurator.configurate()

        solver_param = SolverParams(base, line_follow_weights, traj_follow_weights, pos_goal_weights, line_follow_weights_aggressive, traj_follow_weights_aggressive)
        if build_solver:
            MpcModule(solver_param).build()

        # Load plot config file
        plot_config = 'plot_config.yaml'
        plot_config_fp = os.path.join(str(file_path.parent.parent), 'configs', plot_config)
        with open(plot_config_fp) as f:
            plot_config  = yaml.load(f, Loader=yaml.FullLoader)


        plot_queues, plot_process = start_plotter(solver_param.base, plot_config, aut_test_config=aut_test_config)

        
        graphs = Graphs(solver_param.base)
        g = graphs.get_graph(complexity=graph_complexity)
        map_data = g
        robot_data = ATRS(map_data.positions_master.start, map_data.positions_slave.start, solver_param, plot_config, plot_queues)
    
        traj_gen = TrajectoryGenerator(plot_config, plot_queues, plot_process, solver_param, map_data, robot_data, verbose=True, name="TrajGen", self_destruct=self_destruct)

        return traj_gen, plot_process


def main(destroy_plots=True):
    traj_gen, plot_process = init(verbose=True, build_solver=False, self_destruct=destroy_plots, graph_complexity=1)
    try:
        flag = True
        while flag: 
            done = traj_gen.run(plot=True)
            if done: 
                flag = False
                plot_process.kill()

    finally:
        print("Should kill all running processes")
        

if __name__ == '__main__':
    main()
    print("All done.\nGood bye")
