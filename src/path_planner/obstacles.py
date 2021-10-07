import numpy as np

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
            polygon += self.vel
            center_pos = np.mean(polygon, axis=0)

            return polygon, center_pos

        for i in range(self.n_steps_ahead):
            self.polygon, self.center_pos = step(self.polygon, self.center_pos)
            self.future_polygons.append(self.polygon.copy())

              



