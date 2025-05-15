import numpy as np
import matplotlib.pyplot as plt
from map_4 import map  # map() → start, goal, space, obstacle_list

class Node(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent

class RRT(object):
    def __init__(self, start, goal, space, obstacle_list, success_dist_thres=1.0):
        self.start_node = Node(start[0], start[1])
        self.goal_node = Node(goal[0], goal[1])
        self.space = space  # (x_min, x_max, y_min, y_max)
        self.obstalce_list = obstacle_list  # (x, y, r)
        self.node_list = []

        self.max_iter = 5000
        self.goal_sample_rate = 0.1
        self.min_u = 1.0
        self.max_u = 3.0
        self.success_dist_thres = success_dist_thres
        self.collision_check_step = 0.2
        self.stepsize = 0.5

    def plan(self):
        self.node_list = [self.start_node]
        for i in range(self.max_iter):
            rand_node = self.get_random_node()
            nearest_node = self.find_nearest_node(self.node_list, rand_node)
            u = self.stepsize * self.get_random_input(self.min_u, self.max_u)
            new_node = self.create_child_node(nearest_node, rand_node, u)

            if self.is_collide(new_node, self.obstalce_list):
                continue
            if self.is_path_collide(nearest_node, new_node, self.obstalce_list, self.collision_check_step):
                continue

            new_node.set_parent(nearest_node)
            self.node_list.append(new_node)

            if self.check_goal(new_node, self.success_dist_thres):
                print(" [-] GOAL REACHED")
                return self.backtrace_path(new_node)
        return None

    def is_same_node(self, node1, node2):
        return abs(node1.x - node2.x) < 1e-5 and abs(node1.y - node2.y) < 1e-5

    def backtrace_path(self, node):
        current_node = node
        path = [current_node]
        while not self.is_same_node(current_node, self.start_node):
            current_node = current_node.parent
            path.append(current_node)
        return path[::-1]

    def get_random_node(self):
        if np.random.rand() < self.goal_sample_rate:
            return self.goal_node
        else:
            x = np.random.uniform(self.space[0], self.space[1])
            y = np.random.uniform(self.space[2], self.space[3])
            return Node(x, y)

    def check_goal(self, node, success_dist_thres):
        dx = node.x - self.goal_node.x
        dy = node.y - self.goal_node.y
        distance = np.hypot(dx, dy)
        return distance <= success_dist_thres

    @staticmethod
    def create_child_node(nearest_node, rand_node, u):
        dx = rand_node.x - nearest_node.x
        dy = rand_node.y - nearest_node.y
        theta = np.arctan2(dy, dx)
        new_x = nearest_node.x + u * np.cos(theta)
        new_y = nearest_node.y + u * np.sin(theta)
        return Node(new_x, new_y)

    @staticmethod
    def get_random_input(min_u, max_u):
        return np.random.uniform(min_u, max_u)

    @staticmethod
    def find_nearest_node(node_list, rand_node):
        dists = [np.hypot(n.x - rand_node.x, n.y - rand_node.y) for n in node_list]
        min_index = np.argmin(dists)
        return node_list[min_index]

    @staticmethod
    def is_collide(node, obstacle_list):
        for obs in obstacle_list:
            ox, oy, r = obs
            if np.hypot(node.x - ox, node.y - oy) <= r:
                return True
        return False

    @staticmethod
    def is_path_collide(node_from, node_to, obstacle_list, check_step=0.2):
        dx = node_to.x - node_from.x
        dy = node_to.y - node_from.y
        dist = np.hypot(dx, dy)
        steps = int(dist / check_step)

        for i in range(steps):
            t = i / steps
            x = node_from.x + t * dx
            y = node_from.y + t * dy
            for obs in obstacle_list:
                ox, oy, r = obs
                if np.hypot(x - ox, y - oy) <= r:
                    return True
        return False

# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    start, goal, space, obstacle_list = map()

    success_dist_thres = 1.0
    rrt = RRT(start, goal, space, obstacle_list, success_dist_thres)
    path = rrt.plan()

    if path is None:
        print(" [-] Failed to find a path.")
        exit()

    for node in path:
        print(" [-] x = %.2f, y = %.2f " % (node.x, node.y))

    # draw result
    _t = np.linspace(0, 2*np.pi, 30)
    for obs in obstacle_list:
        x, y, r = obs
        _x = x + r * np.cos(_t)
        _y = y + r * np.sin(_t)
        plt.plot(_x, _y, 'k-')

    goal_x = goal[0] + success_dist_thres * np.cos(_t)
    goal_y = goal[1] + success_dist_thres * np.sin(_t)
    plt.plot(goal_x, goal_y, 'g--')

    for i in range(len(path)-1):
        node_i = path[i]
        node_ip1 = path[i+1]
        plt.plot([node_i.x, node_ip1.x], [node_i.y, node_ip1.y], 'r.-')

    plt.plot(start[0], start[1], 'bo', label='Start')
    plt.plot(goal[0], goal[1], 'go', label='Goal')
    plt.grid(True)
    plt.axis("equal")
    plt.title("RRT Path Planning")
    plt.legend()
    plt.show()
