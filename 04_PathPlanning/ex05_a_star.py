import numpy as np
import math
import matplotlib.pyplot as plt
import random
from map_2 import map

show_animation = True

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0  # actual cost
        self.h = 0  # heuristic
        self.f = 0  # total cost (f = g + h)

    def __eq__(self, other):
        return self.position == other.position

def get_action():
    sqrt2 = math.sqrt(2)
    action_set = [
        [0, 1, 1],
        [0, -1, 1],
        [-1, 0, 1],
        [1, 0, 1],
        [-1, 1, sqrt2],
        [1, 1, sqrt2],
        [-1, -1, sqrt2],
        [1, -1, sqrt2]
    ]
    return action_set

def heuristic(current, goal):
    # Euclidean distance
    dx = current[0] - goal[0]
    dy = current[1] - goal[1]
    return math.hypot(dx, dy)

def collision_check(omap, node):
    return node.position in zip(omap[0], omap[1])

def astar(start, goal, map_obstacle, weight=1.0):
    start_node = Node(None, tuple(start))
    goal_node = Node(None, tuple(goal))

    open_list = []
    closed_list = []

    open_list.append(start_node)

    while open_list:
        # Find node with lowest f
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

        for action in get_action():
            dx, dy, cost = action
            new_pos = (cur_node.position[0] + dx, cur_node.position[1] + dy)

            # Check bounds
            if new_pos[0] < 0 or new_pos[1] < 0:
                continue

            child = Node(cur_node, new_pos)

            if collision_check(map_obstacle, child) or child in closed_list:
                continue

            child.g = cur_node.g + cost
            child.h = heuristic(child.position, goal_node.position)
            child.f = child.g + weight * child.h

            # If better path already exists, skip
            existing_node = next((n for n in open_list if n == child), None)
            if existing_node is None:
                open_list.append(child)
            elif child.g < existing_node.g:
                existing_node.g = child.g
                existing_node.f = child.f
                existing_node.parent = cur_node

        # show animation
        if show_animation:
            plt.plot(cur_node.position[0], cur_node.position[1], 'yo', alpha=0.5)
            if len(closed_list) % 100 == 0:
                plt.pause(0.001)

    return []  # No path found

def main():
    start, goal, omap = map()
    weight = 1.0  # Change this to try different Weighted A*

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
        plt.title(f"A* algorithm (weight={weight})", fontsize=20)

    opt_path = astar(start, goal, omap, weight=weight)
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
