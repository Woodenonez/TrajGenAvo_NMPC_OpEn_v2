import os, sys, json

import numpy as np
import networkx as nx


class GlobalPathPlanner:
    def __init__(self, map_filename):
        self.graph = nx.DiGraph() # directed graph
        self.map_filename = map_filename

        self.__build_graph()

    def __build_graph(self):
        with open(self.map_filename) as f: 
            map = json.load(f)
        for node in map["nodes"]: 
            self.graph.add_node(tuple(node['coord']))
            for neigh in node['neighbour']: 
                if len(neigh) == 2: 
                    self.graph.add_edge(tuple(node['coord']), tuple(neigh))
    
    def __distance_to_goal(self, start, goal):
        return np.hypot(start[0]-goal[0], start[1]-goal[1])

    def search_astar(self, start, end):
        path = nx.astar_path(self.graph, start, end, heuristic=self.__distance_to_goal)
        return path # list of path nodes
