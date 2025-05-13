import numpy as np
import matplotlib.pyplot as plt
from lane_2 import lane
from ex01_PathPlanning_BothLane import Global2Local, Polyfit, VehicleModel_Lat, PurePursuit

class LaneWidthEstimator(object):
    def __init__(self, Lw_init=3.0):
        self.Lw = Lw_init

    def update(self, coeff_L, coeff_R, isLaneValid_L, isLaneValid_R):
        if isLaneValid_L and isLaneValid_R:
            # 두 차선 다 인식된 경우: y값 차이로 차선 너비 갱신
            y_center = 0.0
            y_L = np.polyval(coeff_L, y_center)
            y_R = np.polyval(coeff_R, y_center)
            self.Lw = abs(y_R - y_L)


def EitherLane2Path(coeff_L, coeff_R, isLaneValid_L, isLaneValid_R, Lw):
    
    if isLaneValid_L and isLaneValid_R:
        coeff_path = 0.5 * (np.array(coeff_L) + np.array(coeff_R))
    elif isLaneValid_L:
        coeff_path = np.copy(coeff_L)
        coeff_path[-1] -= Lw / 2.0  # 왼쪽 차선 기준 
    elif isLaneValid_R:
        coeff_path = np.copy(coeff_R)
        coeff_path[-1] += Lw / 2.0  # 오른쪽 차선 기준 
    else:
        coeff_path = np.zeros(4)
    return coeff_path


        
if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
    X_lane = np.arange(0.0, 100.0, 0.1)
    Y_lane_L, Y_lane_R, Valid_L, Valid_R = lane(X_lane)
    
    LaneWidth = LaneWidthEstimator()
    ego_vehicle = VehicleModel_Lat(step_time, Vx)
    controller = PurePursuit()
    
    time = []
    X_ego = []
    Y_ego = []
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)
        # Lane Info
        X_ref = np.arange(ego_vehicle.X, ego_vehicle.X+5.0, 1.0)
        Y_ref_L, Y_ref_R, isLaneValid_L, isLaneValid_R = lane(X_ref)

        # === 2. 유효성 직접 판단 ===

        # 왼쪽 차선 유효성 판단
        valid_ratio = np.count_nonzero(np.abs(Y_ref_L) > 0.1) / len(Y_ref_L)
        isLaneValid_L = (valid_ratio > 0.8) and (np.max(np.abs(Y_ref_L)) > 0.5)

        # 오른쪽 차선 유효성 판단
        valid_ratio_R = np.count_nonzero(np.abs(Y_ref_R) > 0.05) / len(Y_ref_R)
        isLaneValid_R = (valid_ratio_R > 0.6) and (np.max(np.abs(Y_ref_R)) > 0.2)


        # Global points (front 5 meters from the ego vehicle)
        global_points_L = np.transpose(np.array([X_ref, Y_ref_L])).tolist()
        global_points_R = np.transpose(np.array([X_ref, Y_ref_R])).tolist()
        # Converted to local frame
        local_points_L = Global2Local(global_points_L, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        local_points_R = Global2Local(global_points_R, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        # 3th order fitting
        coeff_L = Polyfit(local_points_L, num_order=3)
        coeff_R = Polyfit(local_points_R, num_order=3)
        # Lane to path
        LaneWidth.update(coeff_L, coeff_R, isLaneValid_L, isLaneValid_R)
        coeff_path = EitherLane2Path(coeff_L, coeff_R, isLaneValid_L, isLaneValid_R, LaneWidth.Lw)
        # Controller input
        controller.ControllerInput(coeff_path, Vx)
        ego_vehicle.update(controller.u, Vx)

        print(f"[{i:03d}] Lw = {LaneWidth.Lw:.2f}")

        
        Y_ref_L, Y_ref_R, _, _ = lane(X_ref)

    plt.figure(1, figsize=(13,2))
    plt.plot(X_lane, Y_lane_L,'k--')
    plt.plot(X_lane, Y_lane_R,'k--',label = "Reference")
    plt.plot(X_ego, Y_ego,'b.',label = "Vehicle Position")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
#    plt.axis("best")
    plt.grid(True)    
    plt.show()