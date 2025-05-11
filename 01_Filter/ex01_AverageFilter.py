import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AverageFilter:
    def __init__(self, y_initial_measure):
        self.sum = y_initial_measure  # 누적합
        self.count = 1                # 측정 횟수
        self.y_estimate = y_initial_measure  # 초기 추정값

    def estimate(self, y_measure):
        self.count += 1
        self.sum += y_measure
        self.y_estimate = self.sum / self.count  # 평균 계산
    
if __name__ == "__main__":
    #signal = pd.read_csv("01_filter/Data/example_Filter_1.csv")
    signal = pd.read_csv("01_filter/Data/example_Filter_2.csv")

    y_estimate = AverageFilter(signal.y_measure[0])
    for i, row in signal.iterrows():
        y_estimate.estimate(signal.y_measure[i])
        signal.y_estimate[i] = y_estimate.y_estimate

    plt.figure()
    plt.plot(signal.time, signal.y_measure,'k.',label = "Measure")
    plt.plot(signal.time, signal.y_estimate,'r-',label = "Estimate")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


#for i, row in signal.iterrows():
#    #print(signal.time[i])
#    if (i==0):
#        signal.y_estimate[i] = signal.y_measure[i]
#    else:
#        signal.y_estimate[i] = (signal.y_estimate[i-1])*(num_average-1)/num_average + (signal.y_measure[i])/num_average




