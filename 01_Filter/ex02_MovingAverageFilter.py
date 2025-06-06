import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

class MovingAverageFilter:
    def __init__(self, y_initial_measure, num_average=8.0):
        self.num_average = int(num_average)
        self.window = deque([y_initial_measure], maxlen=self.num_average)
        self.y_estimate = y_initial_measure
        
    def estimate(self, y_measure):
        self.window.append(y_measure)
        self.y_estimate = np.mean(self.window)

    
if __name__ == "__main__":
    #signal = pd.read_csv("week_01_filter/Data/example_Filter_1.csv")      
    signal = pd.read_csv("01_filter/Data/example_Filter_2.csv")

    y_estimate = MovingAverageFilter(signal.y_measure[0])
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



