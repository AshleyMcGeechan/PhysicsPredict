import matplotlib.pyplot as plt
import numpy as np

with open("TestRun.csv", 'r') as f:
    TestRun = np.loadtxt(f, dtype=np.float64)
    TestRun = np.array(TestRun).reshape(-1, 6, 2)

with open("OneFramePrediction.csv", 'r') as f:
    OneFramePrediction = np.loadtxt(f, dtype=np.float64)
    OneFramePrediction = np.array(OneFramePrediction).reshape(-1, 6, 2)

with open("TotalPrediction.csv", 'r') as f:
    TotalPrediction = np.loadtxt(f, dtype=np.float64)
    TotalPrediction = np.array(TotalPrediction).reshape(-1, 6, 2)

plt.style.use('dark_background')

for i in range(6):

    plt.subplot(2, 3, i+1)
    plt.plot(TestRun[0, i, 0],
             TestRun[0, i, 1],
             c='cyan',
             marker='*')

    plt.plot(TestRun[1:, i, 0],
             TestRun[1:, i, 1],
             c='cyan',
             marker='o',
             alpha=0.5,
             label="Actual Run")

    plt.plot(OneFramePrediction[0, i, 0],
             OneFramePrediction[0, i, 1],
             c='yellow',
             marker='*')

    plt.plot(OneFramePrediction[1:, i, 0],
             OneFramePrediction[1:, i, 1],
             c='yellow',
             marker='o',
             alpha=0.5,
             label="One frame ahead prediction")

    plt.plot(TotalPrediction[0, i, 0],
             TotalPrediction[0, i, 1],
             c='magenta',
             marker='*')

    plt.plot(TotalPrediction[1:, i, 0],
             TotalPrediction[1:, i, 1],
             c='magenta',
             marker='o',
             alpha=0.5,
             label="Prediction from initial frames")

    plt.axis([0, 20, 0, 40])

plt.legend()
plt.show()

