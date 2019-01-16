import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

with open("TestRun.csv", 'r') as f:
    TestRun = np.loadtxt(f, dtype=np.float64)
    TestRun = np.array(TestRun).reshape(-1, 3, 2)

with open("OneFramePrediction.csv", 'r') as f:
    OneFramePrediction = np.loadtxt(f, dtype=np.float64)
    OneFramePrediction = np.array(OneFramePrediction).reshape(-1, 3, 2)

with open("TotalPrediction.csv", 'r') as f:
    TotalPrediction = np.loadtxt(f, dtype=np.float64)
    TotalPrediction = np.array(TotalPrediction).reshape(-1, 3, 2)


plt.style.use('dark_background')

xscaler = joblib.load("lstm_xscaler.save")
yscaler = joblib.load("lstm_yscaler.save")

data = TestRun.reshape(-1, 2)
xdata = xscaler.transform(data[:, ::2])
ydata = yscaler.transform(data[:, 1::2])
xdata = xdata.reshape(-1, 1)
ydata = ydata.reshape(-1, 1)
data = np.concatenate((xdata, ydata), axis=1)
data = data.reshape(-1, 3, 2)
TestRun = data

data = OneFramePrediction.reshape(-1, 2)
xdata = xscaler.transform(data[:, ::2])
ydata = yscaler.transform(data[:, 1::2])
xdata = xdata.reshape(-1, 1)
ydata = ydata.reshape(-1, 1)
data = np.concatenate((xdata, ydata), axis=1)
data = data.reshape(-1, 3, 2)
OneFramePrediction = data

data = TotalPrediction.reshape(-1, 2)
xdata = xscaler.transform(data[:, ::2])
ydata = yscaler.transform(data[:, 1::2])
xdata = xdata.reshape(-1, 1)
ydata = ydata.reshape(-1, 1)
data = np.concatenate((xdata, ydata), axis=1)
data = data.reshape(-1, 3, 2)
TotalPrediction = data

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



