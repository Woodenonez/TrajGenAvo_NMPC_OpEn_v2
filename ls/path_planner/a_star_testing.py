from shapely.geometry import Polygon, Point, LineString
import numpy as np
import heapq
from typing import Dict, List, Iterator, Tuple, TypeVar, Optional
T = TypeVar('T')
import networkx as nx
Location = TypeVar('Location')

class PrioQue:
    def __init__(self):
        self.elements: List[Tuple[float, T]] = []
    
    def empty(self) -> bool:
        return not self.elements
    
    def put(self, item: T, priority: float):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self) -> T:
        return heapq.heappop(self.elements)[1]

class realGraph:
    def __init__(self, boundaries=[], obstacle_list=[], road_list=[], extra_nodes=[]):
        #Get bounds of boundaries (outmost edges)
        self.roads=[]
        for road in road_list:
            self.roads.append(Polygon(road))
        self.space=Polygon(boundaries)
        self.obstacle_list=[]
        for obstacle in obstacle_list:
            self.obstacle_list.append(Polygon(obstacle))
        self.tol=1
        self.offroad_weight=10
        self.extra_nodes=extra_nodes
        self.visited_nodes=[]
    
    def build_from_nodes(self, node_list):
        self.G=nx.DiGraph()

        for node in node_list:
            self.G.add_node(node.coord)
            for neighbour in node.neighbours:
                self.G.add_edge(node.coord, neighbour.coord)

        return 1

    def search_astar(self, start, end):
        path=nx.astar_path(self.G, start, end, heuristic=self.estimated_cost_to_goal)
        return path
    
    def get_neighbours(self, point : Location):
        x=point[0]
        y=point[1]

        step_dirs=[[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, -1], [-1, 1]] # distance on [2]
            # Get neighbours
        
        neighbour_list=[]
        point_bounds=[]
        for direction in step_dirs:
            
            neighbour_x= x + self.tol*direction[0]
            neighbour_y= y + self.tol*direction[1]
            connecting_line=LineString([(x, y), (neighbour_x, neighbour_y)]) # Line between neighbour and current point
            neighbour_point=Point(neighbour_x, neighbour_y)
            point_bounds.append((neighbour_x, neighbour_y))
            collision_free=True
            #check if distance to any object from the line between the nodes is == 0
            for obstacle in self.obstacle_list:
                if obstacle.distance(connecting_line) == 0.0:
                    collision_free=False

            if self.space.contains(neighbour_point) and collision_free==True:
                neighbour_list.append((neighbour_x, neighbour_y))

        #Create square from neighbours
        point_square=Polygon(point_bounds)
        for node in self.extra_nodes:
            #reused code from above
            collision_free=True
            point_node=Point(node)
            connecting_line=LineString([(x, y), node])
            #check if distance to any object from the line between the nodes is == 0
            for obstacle in self.obstacle_list:
                if obstacle.distance(connecting_line) == 0.0:
                    collision_free=False
            if point_square.contains(point_node) and self.space.contains(point_node) and collision_free == True:
                neighbour_list.append(node)

        
            

        return neighbour_list
    
    def cost(self, from_node, to_node):
        on_road=False
        for road in self.roads:
            if road.contains(Point(to_node)):
                on_road=True
        
        if on_road:
            cost=self.euc_distance(from_node, to_node)
        else:
            cost=self.euc_distance(from_node, to_node)*self.offroad_weight
        return cost
    
    def reconstruct_path(self, came_from: Dict[Location, Location], start: Location, goal: Location) -> List[Location]:
        current: Location = goal
        path: List[Location] = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start) # optional
        path.reverse() # optional
        return path
    
    def estimated_cost_to_goal(self, from_node, goal_node):
        a=np.power(from_node[0]-goal_node[0], 2)
        b=np.power(from_node[1]-goal_node[1], 2)
        cost=np.sqrt(a+b)
        return cost
    
    def euc_distance(self, from_node, to_node):
        a=np.power(from_node[0]-to_node[0], 2)
        b=np.power(from_node[1]-to_node[1], 2)
        dist=np.sqrt(a+b)
        return dist
    