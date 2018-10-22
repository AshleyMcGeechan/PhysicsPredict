import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.metrics import mean_absolute_error

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


def calcError(x, y):

    cumulativeError = 0
    errorVector = np.zeros(len(y))
    for i in range(len(y)):
        cumulativeError += mean_absolute_error(y[i], x[i])
        errorVector[i] = cumulativeError
    print(errorVector)
    return errorVector


def plotError(errorVector, title):

    fig = plt.figure()
    plt.plot(errorVector)
    plt.ylabel("Cumulative Error")
    plt.xlabel("Frame")
    plt.suptitle(title)
    plt.show()
    return fig

plots = []
plots.append(plotError(calcError(TestRun, OneFramePrediction), "One Frame Prediction"))
plots.append(plotError(calcError(TestRun, TotalPrediction), "Total Prediction"))

pp = PdfPages('Error_Plots.pdf')
for i in plots:
    pp.savefig(i)

pp.close()



