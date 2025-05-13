import numpy as np
import matplotlib.pyplot as plt
from lane_1 import lane
from ex01_PathPlanning_BothLane import (
    Global2Local, Polyfit, Polyval,
    BothLane2Path, VehicleModel_Lat, PurePursuit
)

class LeadingVehiclePos(object):
    def __init__(self, num_data_store=10):
        self.max_num_array = num_data_store
        self.PosArray = []

    def update(self, pos_lead, Vx, yawrate, dt):
        if len(self.PosArray) >= self.max_num_array:
            self.PosArray.pop(0)
        self.PosArray.append(pos_lead[0])

def HeadingAngleEstimation(coeff_path, PosArray):
    if len(PosArray) < 2:
        return 0.0
    dx = PosArray[-1][0] - PosArray[-2][0]
    dy = PosArray[-1][1] - PosArray[-2][1]
    return np.arctan2(dy, dx)

# TargetFollowingPath 수정
def TargetFollowingPath(PosArray):
    if len(PosArray) < 2:
        return [0.0, 0.0, 0.0, 0.0]
    x = [p[0] for p in PosArray]  # → 이 부분이 global X
    y = [p[1] for p in PosArray]  # → 이 부분이 global Y

    coeffs = np.polyfit(x, y, 1)  # 1차 다항식
    a1, a0 = coeffs
    return [0.0, 0.0, a1, a0]     # 1차만 사용하도록 포맷 맞춤


if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
    X_lane = np.arange(0.0, 100.0, 0.1)
    Y_lane_L, Y_lane_R = lane(X_lane)

    leading_vehicle = VehicleModel_Lat(step_time, Vx)
    ego_vehicle = VehicleModel_Lat(step_time, Vx, Pos=[-10.0, 0.0, 0.0])
    controller_lead = PurePursuit()
    controller_ego = PurePursuit()
    
    leading_vehicle_pos = LeadingVehiclePos()

    plt.figure(figsize=(13, 2))
    for i in range(int(simulation_time / step_time)):
        X_ref = np.arange(leading_vehicle.X, leading_vehicle.X + 5.0, 1.0)
        Y_ref_L, Y_ref_R = lane(X_ref)
        global_points_L = np.transpose(np.array([X_ref, Y_ref_L])).tolist()
        global_points_R = np.transpose(np.array([X_ref, Y_ref_R])).tolist()

        local_points_L = Global2Local(global_points_L, leading_vehicle.Yaw, leading_vehicle.X, leading_vehicle.Y)
        local_points_R = Global2Local(global_points_R, leading_vehicle.Yaw, leading_vehicle.X, leading_vehicle.Y)
        coeff_L = Polyfit(local_points_L, num_order=3)
        coeff_R = Polyfit(local_points_R, num_order=3)
        coeff_path_lead = BothLane2Path(coeff_L, coeff_R)

        # 선행 차량 위치를 ego 로컬로 변환 후 버퍼 저장
        pos_lead = Global2Local([[leading_vehicle.X, leading_vehicle.Y]],
                                ego_vehicle.Yaw,
                                ego_vehicle.X,
                                ego_vehicle.Y)
        leading_vehicle_pos.update(pos_lead, Vx, ego_vehicle.yawrate, step_time)

        # 추종 경로 생성
        coeff_path_ego = TargetFollowingPath(leading_vehicle_pos.PosArray)

        controller_lead.ControllerInput(coeff_path_lead, Vx)
        controller_ego.ControllerInput(coeff_path_ego, Vx)

        leading_vehicle.update(controller_lead.u, Vx)
        ego_vehicle.update(controller_ego.u, Vx)

        print("Ego path coeff:", coeff_path_ego)

        plt.plot(ego_vehicle.X, ego_vehicle.Y, 'bo')
        plt.plot(leading_vehicle.X, leading_vehicle.Y, 'ro')
        plt.axis("equal")
        plt.pause(0.01)

    plt.show()
