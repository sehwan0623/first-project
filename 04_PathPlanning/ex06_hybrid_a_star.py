import numpy as np
import math
import matplotlib.pyplot as plt
from map_3 import map

show_animation = True

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position  # [x, y, yaw]
        self.heading = position[2] if position else 0.0
        self.f = 0
        self.g = 0
        self.h = 0

def isSamePosition(node_1, node_2, epsilon_position=0.3):
    dx = node_1.position[0] - node_2.position[0]
    dy = node_1.position[1] - node_2.position[1]
    return math.hypot(dx, dy) < epsilon_position

def isSameYaw(node_1, node_2, epsilon_yaw=0.2):
    dyaw = abs(node_1.heading - node_2.heading)
    return dyaw < epsilon_yaw

def get_action(R, Vx, delta_time_step):
    yaw_rate = Vx / R
    distance_travel = Vx * delta_time_step
    return [
        [yaw_rate, delta_time_step, distance_travel],
        [-yaw_rate, delta_time_step, distance_travel],
        [yaw_rate / 2, delta_time_step, distance_travel],
        [-yaw_rate / 2, delta_time_step, distance_travel],
        [0.0, delta_time_step, distance_travel]
    ]

# yqw rate가 있는 곡선 주행과 직선 주행을 구분하여 처리
def vehicle_move(position_parent, yaw_rate, delta_time, Vx):
    x, y, yaw = position_parent
    if abs(yaw_rate) > 1e-3:
        R = Vx / yaw_rate
        cx = x - R * math.sin(yaw)
        cy = y + R * math.cos(yaw)
        dtheta = yaw_rate * delta_time
        x_new = cx + R * math.sin(yaw + dtheta)
        y_new = cy - R * math.cos(yaw + dtheta)
        yaw_new = yaw + dtheta
    else:
        x_new = x + Vx * delta_time * math.cos(yaw)
        y_new = y + Vx * delta_time * math.sin(yaw)
        yaw_new = yaw
    yaw_new = (yaw_new + 2 * np.pi) % (2 * np.pi)
    return [x_new, y_new, yaw_new]

# 
def collision_check(position, yaw_rate, delta_time_step, obstacle_list, Vx):
    x, y, _ = vehicle_move(position, yaw_rate, delta_time_step, Vx)
    for obs in obstacle_list:
        dx = x - obs[0]
        dy = y - obs[1]
        if math.hypot(dx, dy) < obs[2]:
            return True
    return False

def isNotInSearchingSpace(position, space):
    x, y = position[0], position[1]
    return not (space[0] <= x <= space[1] and space[2] <= y <= space[3])

def heuristic(cur_node, goal_node):
    dx = cur_node.position[0] - goal_node.position[0]
    dy = cur_node.position[1] - goal_node.position[1]
    return math.hypot(dx, dy)

def a_star(start, goal, space, obstacle_list, R, Vx, delta_time_step, weight):
    start_node = Node(None, start)
    goal_node = Node(None, goal)
    open_list = [start_node]
    closed_list = []

    while open_list:
        open_list.sort(key=lambda n: n.f)
        cur_node = open_list.pop(0)
        closed_list.append(cur_node)

        if isSamePosition(cur_node, goal_node, epsilon_position=0.6):
            path = []
            while cur_node:
                path.append(cur_node.position)
                cur_node = cur_node.parent
            return path[::-1]

        action_set = get_action(R, Vx, delta_time_step)
        for yaw_rate, dt, _ in action_set:
            new_pos = vehicle_move(cur_node.position, yaw_rate, dt, Vx)
            if isNotInSearchingSpace(new_pos, space):
                continue
            if collision_check(cur_node.position, yaw_rate, dt, obstacle_list, Vx):
                continue

            child_node = Node(cur_node, new_pos)
            if any(isSamePosition(child_node, closed_node) and isSameYaw(child_node, closed_node)
                   for closed_node in closed_list):
                continue

            child_node.g = cur_node.g + Vx * dt
            child_node.h = heuristic(child_node, goal_node)
            child_node.f = child_node.g + weight * child_node.h

            if any(isSamePosition(child_node, open_node) and child_node.f >= open_node.f for open_node in open_list):
                continue
            open_list.append(child_node)

        if show_animation:
            plt.plot(cur_node.position[0], cur_node.position[1], 'yo', alpha=0.5)
            if len(closed_list) % 100 == 0:
                plt.pause(0.1)
    return []

def main():
    start, goal, obstacle_list, space = map()

    if show_animation:
        theta = np.linspace(0, 2 * np.pi, 101)
        plt.figure(figsize=(8, 8))
        plt.plot(start[0], start[1], 'bs', markersize=7)
        plt.text(start[0], start[1] + 0.5, 'start', fontsize=12)
        plt.plot(goal[0], goal[1], 'rs', markersize=7)
        plt.text(goal[0], goal[1] + 0.5, 'goal', fontsize=12)
        for obs in obstacle_list:
            x_obs = obs[0] + obs[2] * np.cos(theta)
            y_obs = obs[1] + obs[2] * np.sin(theta)
            plt.plot(x_obs, y_obs, 'k-')
        plt.axis(space)
        plt.grid(True)
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Hybrid A* Algorithm", fontsize=20)

    opt_path = a_star(start, goal, space, obstacle_list, R=5.0, Vx=2.0, delta_time_step=0.5, weight=1.1)
    if opt_path:
        print("Optimal path found!")
        opt_path = np.array(opt_path)
        if show_animation:
            plt.plot(opt_path[:, 0], opt_path[:, 1], "m.-")
            plt.show()
    else:
        print("No path found.")

if __name__ == "__main__":
    main()