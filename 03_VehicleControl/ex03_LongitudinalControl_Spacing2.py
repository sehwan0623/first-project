import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Long import VehicleModel_Long

class PID_Controller_ConstantTimeGap(object):
    def __init__(self, step_time, timegap=1.5, P_Gain=3, D_Gain=0.005, I_Gain=0.0):
        self.timegap = timegap
        self.Kp = P_Gain
        self.Kd = D_Gain
        self.Ki = I_Gain
        self.dt = step_time
        self.e_prev = 0.0
        self.e_sum = 0.0
        self.u = 0.0
        self.space = 0.0

    def ControllerInput(self, target_x, ego_x, ego_vx):
        # 매 timestep마다 기준 거리 갱신
        self.space = ego_vx * self.timegap

        # 거리 오차 = 실제 거리 - 기준 거리
        actual_space = target_x - ego_x
        error = actual_space - self.space
        d_error = (error - self.e_prev) / self.dt
        self.e_sum += error * self.dt

        self.u = self.Kp * error + self.Kd * d_error + self.Ki * self.e_sum
        self.e_prev = error


if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 50.0
    m = 500.0

    # 기록 변수
    vx_ego = []
    vx_target = []
    x_space = []
    x_reference = []
    timegap_actual = []
    time = []

    # 차량 초기화
    target_vehicle = VehicleModel_Long(step_time, m, 0.0, 30.0, 10.0)
    ego_vehicle = VehicleModel_Long(step_time, m, 0.5, 0.0, 5.0)

    # 컨트롤러 초기화
    controller = PID_Controller_ConstantTimeGap(step_time, timegap=1.5)

    for i in range(int(simulation_time / step_time)):
        time.append(step_time * i)

        # 기록
        vx_ego.append(ego_vehicle.vx)
        vx_target.append(target_vehicle.vx)
        gap = target_vehicle.x - ego_vehicle.x
        x_space.append(gap)
        x_reference.append(controller.space if ego_vehicle.vx > 0.1 else 0.0)
        timegap_actual.append(gap / ego_vehicle.vx if ego_vehicle.vx > 0.1 else 0.0)

        # 제어 입력 계산 및 차량 업데이트
        controller.ControllerInput(target_vehicle.x, ego_vehicle.x, ego_vehicle.vx)
        ego_vehicle.update(controller.u)
        target_vehicle.update(0.0)

    # 속도 그래프
    plt.figure(1)
    plt.plot(time, vx_ego, 'r-', label="ego_vx [m/s]")
    plt.plot(time, vx_target, 'b-', label="target_vx [m/s]")
    plt.xlabel('time [s]')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)

    # 간격 그래프
    plt.figure(2)
    plt.plot(time, x_reference, 'k--', label="reference space [m]")
    plt.plot(time, x_space, 'b-', label="actual space [m]")
    plt.xlabel('time [s]')
    plt.ylabel('Distance')
    plt.legend()
    plt.grid(True)

    # Time Gap 그래프
    plt.figure(3)
    plt.plot([0, time[-1]], [controller.timegap, controller.timegap], 'k--', label="reference timegap [s]")
    plt.plot(time, timegap_actual, 'b-', label="actual timegap [s]")
    plt.xlabel('time [s]')
    plt.ylabel('Time Gap [s]')
    plt.legend()
    plt.grid(True)

    plt.show()
