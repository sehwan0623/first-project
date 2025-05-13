import numpy as np
import math
import matplotlib.pyplot as plt
import random
from map_1 import map

show_animation = True

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.f = 0  # cost

    def __eq__(self, other):
        return self.position == other.position

def get_action():
    # dx, dy, cost
    sqrt2 = math.sqrt(2)
    action_set = [
        [0, 1, 1],    # up
        [0, -1, 1],   # down
        [-1, 0, 1],   # left
        [1, 0, 1],    # right
        [-1, 1, sqrt2],  # up-left
        [1, 1, sqrt2],   # up-right
        [-1, -1, sqrt2], # down-left
        [1, -1, sqrt2]   # down-right
    ]
    return action_set

def collision_check(omap, node):
    # Check if the position is in obstacle map
    if node.position in zip(omap[0], omap[1]):
        return True
    return False

def dijkstra(start, goal, map_obstacle):
    start_node = Node(None, tuple(start))
    start_node.f = 0
    goal_node = Node(None, tuple(goal))

    open_list = []
    closed_list = []

    open_list.append(start_node)

    while open_list:
        # Get node with lowest cost
        cur_node = min(open_list, key=lambda node: node.f)
        open_list.remove(cur_node)
        closed_list.append(cur_node)

        # Goal check
        if cur_node == goal_node:
            path = []
            while cur_node is not None:
                path.append(cur_node.position)
                cur_node = cur_node.parent
            return path[::-1]

        # Generate children
        for action in get_action():
            dx, dy, cost = action
            new_pos = (cur_node.position[0] + dx, cur_node.position[1] + dy)
            child = Node(cur_node, new_pos)
            child.f = cur_node.f + cost

            # Skip if collision or out of bounds
            if new_pos[0] < 0 or new_pos[1] < 0:
                continue
            if collision_check(map_obstacle, child):
                continue
            if child in closed_list:
                continue

            existing_node = next((node for node in open_list if node == child), None)
            if existing_node is None:
                open_list.append(child)
            elif child.f < existing_node.f:
                existing_node.f = child.f
                existing_node.parent = cur_node

        # Show animation
        if show_animation:
            plt.plot(cur_node.position[0], cur_node.position[1], 'yo', alpha=0.5)
            if len(closed_list) % 100 == 0:
                plt.pause(0.001)

    return []  # No path found

def main():
    start, goal, omap = map()

    if show_animation:
        plt.figure(figsize=(8,8))
        plt.plot(start[0], start[1], 'bs', markersize=7)
        plt.text(start[0], start[1]+0.5, 'start', fontsize=12)
        plt.plot(goal[0], goal[1], 'rs', markersize=7)
        plt.text(goal[0], goal[1]+0.5, 'goal', fontsize=12)
        plt.plot(omap[0], omap[1], '.k', markersize=10)
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("X [m]"), plt.ylabel("Y [m]")
        plt.title("Dijkstra algorithm", fontsize=20)

    opt_path = dijkstra(start, goal, omap)
    if opt_path:
        print("Optimal path found!")
        opt_path = np.array(opt_path)
        if show_animation:
            plt.plot(opt_path[:,0], opt_path[:,1], "m.-")
            plt.show()
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
