import numpy as np
from typing import Dict, List, Iterator, Tuple, TypeVar, Optional
T = TypeVar('T')
import networkx as nx
Location = TypeVar('Location')
import json

class GlobalPathPlanner:
    def __init__(self, filename):
        self.G = nx.DiGraph()
        self.filename = filename
        self.build_from_nodes()

    def build_from_nodes(self):
        with open(self.filename) as f: 
            map = json.load(f)
        for node in map["nodes"]: 
            self.G.add_node(tuple(node['coord']))
            for neigh in node['neighbour']: 
                if len(neigh) == 2: 
                    self.G.add_edge(tuple(node['coord']), tuple(neigh))
            
        return 1

    def search_astar(self, start, end):
        path=nx.astar_path(self.G, start, end, heuristic=self.estimated_cost_to_goal)
        return path
    
    def estimated_cost_to_goal(self, from_node, goal_node):
        a=np.power(from_node[0]-goal_node[0], 2)
        b=np.power(from_node[1]-goal_node[1], 2)
        cost=np.sqrt(a+b)
        return cost