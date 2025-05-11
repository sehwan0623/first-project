import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Lat import VehicleModel_Lat
from ex06_GlobalFrame2LocalFrame import Global2Local, PolynomialFitting, PolynomialValue

class PD_Controller(object):
    def __init__(self, step_time, P_gain=2.0, D_gain=1.0, L=2.5):
        self.dt = step_time
        self.P = P_gain
        self.D = D_gain
        self.L = L
        self.prev_error = 0.0
        self.u = 0.0  # steering angle

    def ControllerInput(self, coeff, Vx, lookahead=2.0):
        a = coeff[0][0]
        b = coeff[1][0]
        c = coeff[2][0]
        d = coeff[3][0]
        x = lookahead

        # 곡률 계산
        dy = 3*a*x**2 + 2*b*x + c
        ddy = 6*a*x + 2*b
        curvature = ddy / ((1 + dy**2)**1.5 + 1e-6)  # 안정성 위해 분모에 epsilon 추가

        # Feedforward 조향각
        delta_ff = np.arctan(self.L * curvature)

        # Feedback (Lookahead lateral error)
        error = a*x**3 + b*x**2 + c*x + d
        d_error = (error - self.prev_error) / self.dt
        self.prev_error = error
        delta_fb = self.P * error + self.D * d_error

        self.u = delta_ff + delta_fb


if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 20.0  # 고속 주행
    X_ref = np.arange(0.0, 100.0, 0.1)
    Y_ref = 2.0 - 2 * np.cos(X_ref / 10)
    num_degree = 3
    num_point = 5
    x_local = np.arange(0.0, 10.0, 0.5)

    # 객체 초기화
    time = []
    X_ego = []
    Y_ego = []
    ego_vehicle = VehicleModel_Lat(step_time, Vx)

    frameconverter = Global2Local(num_point)
    polynomialfit = PolynomialFitting(num_degree, num_point)
    polynomialvalue = PolynomialValue(num_degree, np.size(x_local))
    controller = PD_Controller(step_time)

    for i in range(int(simulation_time / step_time)):
        time.append(step_time * i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)

        # reference points 생성
        X_ref_convert = np.arange(ego_vehicle.X, ego_vehicle.X + 5.0, 1.0)
        Y_ref_convert = 2.0 - 2 * np.cos(X_ref_convert / 10)
        Points_ref = np.transpose(np.array([X_ref_convert, Y_ref_convert]))

        # Local 변환 및 다항식 fitting
        frameconverter.convert(Points_ref, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        polynomialfit.fit(frameconverter.LocalPoints)
        polynomialvalue.calculate(polynomialfit.coeff, x_local)

        # 제어 입력 계산 및 적용
        controller.ControllerInput(polynomialfit.coeff, Vx)
        ego_vehicle.update(controller.u, Vx)

    # 결과 시각화
    plt.figure(1)
    plt.plot(X_ref, Y_ref, 'k--', label="Reference")
    plt.plot(X_ego, Y_ego, 'b-', label="Vehicle Path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="best")
    plt.title("Lateral Control @ High Speed (20 m/s)")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
