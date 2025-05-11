from vehicle_model import VehicleModel
import numpy as np
import matplotlib.pyplot as plt

class PID_Controller(object):
    def __init__(self, reference, measure, step_time, P_Gain=0.4, D_Gain=0.9, I_Gain=0.05):
        self.Kp = P_Gain
        self.Kd = D_Gain
        self.Ki = I_Gain
        self.dt = step_time
        self.e_prev = measure - reference
        self.e_sum = 0.0
        self.u = 0.0

    def ControllerInput(self, reference, measure):
        error = measure - reference
        d_error = (error - self.e_prev) / self.dt
        self.e_sum += error * self.dt

        self.u = -self.Kp * error - self.Kd * d_error - self.Ki * self.e_sum
        self.e_prev = error
        
class KalmanFilter:
    def __init__(self, y_init, step_time=0.1, m=1.0, Q_x=0.01, Q_v=0.1, R=0.5, P_init=1.0):
        self.dt = step_time

        # 상태 행렬
        self.A = np.array([[1.0, self.dt],
                           [0.0, 1.0]])
        self.B = np.array([[0.5 * self.dt**2],
                           [self.dt]])
        self.C = np.array([[1.0, 0.0]])

        self.Q = np.array([[Q_x, 0.0],
                           [0.0, Q_v]])
        self.R = R
        self.P_estimate = np.eye(2) * P_init
        self.x_estimate = np.array([[y_init], [0.0]])

    def estimate(self, y_measure, input_u):
        # 예측
        x_predict = self.A @ self.x_estimate + self.B * input_u
        P_predict = self.A @ self.P_estimate @ self.A.T + self.Q

        # 보정
        K = P_predict @ self.C.T @ np.linalg.inv(self.C @ P_predict @ self.C.T + self.R)
        self.x_estimate = x_predict + K @ (y_measure - self.C @ x_predict)
        self.P_estimate = (np.eye(2) - K @ self.C) @ P_predict
        
        
if __name__ == "__main__":
    target_y = 0.0
    measure_y =[]
    estimated_y = []
    time = []
    step_time = 0.1
    simulation_time = 30   
    plant = VehicleModel(step_time, 0.25, 0.99, 0.05)
    estimator = KalmanFilter(plant.y_measure[0][0])
    controller = PID_Controller(target_y, plant.y_measure[0][0], step_time)
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        measure_y.append(plant.y_measure[0][0])
        estimated_y.append(estimator.x_estimate[0][0])
        estimator.estimate(plant.y_measure[0][0],controller.u)
        controller.ControllerInput(target_y, estimator.x_estimate[0][0])
        plant.ControlInput(controller.u)
    
    plt.figure()
    plt.plot([0, time[-1]], [target_y, target_y], 'k-', label="reference")
    plt.plot(time, measure_y,'r:',label = "Vehicle Position(Measure)")
    plt.plot(time, estimated_y,'c-',label = "Vehicle Position(Estimator)")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
