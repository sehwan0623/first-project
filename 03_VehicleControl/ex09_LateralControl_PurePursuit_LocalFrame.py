import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Lat import VehicleModel_Lat
from ex06_GlobalFrame2LocalFrame import Global2Local, PolynomialFitting, PolynomialValue


class PurePursuit:
    def __init__(self, step_time, L=2.5, lookahead=3.0):
        self.dt = step_time
        self.L = L              # Wheelbase
        self.ld = lookahead    # Lookahead distance
        self.u = 0.0            # steering angle

    def ControllerInput(self, coeff):
        # Lookahead point (x = ld), y = f(ld)
        a = coeff[0][0]
        b = coeff[1][0]
        c = coeff[2][0]
        d = coeff[3][0]
        x_ld = self.ld
        y_ld = a*x_ld**3 + b*x_ld**2 + c*x_ld + d

        # Pure pursuit steering angle
        self.u = np.arctan2(2 * self.L * y_ld, self.ld**2 + 1e-6)


if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
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
    controller = PurePursuit(step_time)

    for i in range(int(simulation_time / step_time)):
        time.append(step_time * i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)

        # Reference 구간 추출 및 변환
        X_ref_convert = np.arange(ego_vehicle.X, ego_vehicle.X + 5.0, 1.0)
        Y_ref_convert = 2.0 - 2 * np.cos(X_ref_convert / 10)
        Points_ref = np.transpose(np.array([X_ref_convert, Y_ref_convert]))
        frameconverter.convert(Points_ref, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)

        # Local reference fitting
        polynomialfit.fit(frameconverter.LocalPoints)
        polynomialvalue.calculate(polynomialfit.coeff, x_local)

        # Pure pursuit 제어
        controller.ControllerInput(polynomialfit.coeff)
        ego_vehicle.update(controller.u, Vx)

    # 결과 시각화
    plt.figure(1)
    plt.plot(X_ref, Y_ref, 'k--', label="Reference")
    plt.plot(X_ego, Y_ego, 'b-', label="Vehicle Path")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
    plt.title("Pure Pursuit Lateral Control")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
