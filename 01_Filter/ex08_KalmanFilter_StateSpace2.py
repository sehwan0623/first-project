import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, y_Measure_init, step_time=0.1, m=10.0, b=2.0, k=100.0,
                 Q_x=0.02, Q_v=0.05, R=0.1, errorCovariance_init=10.0):
        self.dt = step_time

        # 연속 상태공간 행렬
        A_c = np.array([[0, 1], [-k/m, -b/m]])
        B_c = np.array([[0], [1/m]])

        # 이산화 (Euler 방법)
        self.A = np.eye(2) + A_c * self.dt
        self.B = B_c * self.dt
        self.C = np.array([[1.0, 0.0]])
        self.Q = np.array([[Q_x, 0.0], [0.0, Q_v]])
        self.R = R

        self.x_estimate = np.array([[y_Measure_init], [0.0]])  # 초기 위치, 속도
        self.P_estimate = np.eye(2) * errorCovariance_init

    def estimate(self, y_measure, input_u):
        # 예측 단계
        x_predict = self.A @ self.x_estimate + self.B * input_u
        P_predict = self.A @ self.P_estimate @ self.A.T + self.Q

        # 칼만 이득
        K = P_predict @ self.C.T @ np.linalg.inv(self.C @ P_predict @ self.C.T + self.R)

        # 보정 단계
        self.x_estimate = x_predict + K @ (y_measure - self.C @ x_predict)
        self.P_estimate = (np.eye(2) - K @ self.C) @ P_predict
        
if __name__ == "__main__":
    signal = pd.read_csv("01_filter/Data/example08.csv")

    y_estimate = KalmanFilter(signal.y_measure[0])
    for i, row in signal.iterrows():
        y_estimate.estimate(signal.y_measure[i],signal.u[i])
        signal.y_estimate[i] = y_estimate.x_estimate[0][0]

    plt.figure()
    plt.plot(signal.time, signal.y_measure,'k.',label = "Measure")
    plt.plot(signal.time, signal.y_estimate,'r-',label = "Estimate")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()



