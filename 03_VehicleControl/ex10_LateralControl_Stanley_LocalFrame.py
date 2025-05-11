import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Lat import VehicleModel_Lat
from ex06_GlobalFrame2LocalFrame import Global2Local
from ex06_GlobalFrame2LocalFrame import PolynomialFitting
from ex06_GlobalFrame2LocalFrame import PolynomialValue

    
if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
    X_ref = np.arange(0.0, 100.0, 0.1)
    Y_ref = 2.0-2*np.cos(X_ref/10)
    num_degree = 3
    num_point = 5
    x_local = np.arange(0.0, 10.0, 0.5)

    class StanleyMethod(object):
        def __init__(self, step_time, coeff, vx, k_gain=1.0):
            self.dt = step_time
            self.k = k_gain
            self.vx = vx
            self.u = 0.0
            self.epsilon = 1e-6
            self.coeff = coeff

        def ControllerInput(self, coeff, vx):
            self.coeff = coeff
            self.vx = vx

            # 3차 다항식 계수
            a, b, c, d = coeff[0][0], coeff[1][0], coeff[2][0], coeff[3][0]

            # x = 0일 때 경로 상 위치와 기울기
            x = 0.0
            y = a*x**3 + b*x**2 + c*x + d
            dy_dx = 3*a*x**2 + 2*b*x + c

            # reference 방향 (곡선 기울기)
            theta_ref = np.arctan(dy_dx)

            # 차량 진행방향은 local 기준으로 0
            theta_vehicle = 0.0

            # heading error
            heading_error = theta_ref - theta_vehicle

            # cross track error: local y좌표
            cross_track_error = y

            # Stanley control law
            self.u = heading_error + np.arctan2(self.k * cross_track_error, self.vx + self.epsilon)

    
    time = []
    X_ego = []
    Y_ego = []
    ego_vehicle = VehicleModel_Lat(step_time, Vx)

    frameconverter = Global2Local(num_point)
    polynomialfit = PolynomialFitting(num_degree,num_point)
    polynomialvalue = PolynomialValue(num_degree,np.size(x_local))
    controller = StanleyMethod(step_time, polynomialfit.coeff, Vx)
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)
        X_ref_convert = np.arange(ego_vehicle.X, ego_vehicle.X+5.0, 1.0)
        Y_ref_convert = 2.0-2*np.cos(X_ref_convert/10)
        Points_ref = np.transpose(np.array([X_ref_convert, Y_ref_convert]))
        frameconverter.convert(Points_ref, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        polynomialfit.fit(frameconverter.LocalPoints)
        polynomialvalue.calculate(polynomialfit.coeff, x_local)
        controller.ControllerInput(polynomialfit.coeff, Vx)
        ego_vehicle.update(controller.u, Vx)

        
    plt.figure(1)
    plt.plot(X_ref, Y_ref,'k-',label = "Reference")
    plt.plot(X_ego, Y_ego,'b-',label = "Position")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
#    plt.axis("best")
    plt.grid(True)    
    plt.show()


