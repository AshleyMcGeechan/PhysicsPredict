import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import glob
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib


folderName = input("\nEnter the name of the folder to be predicted on. \n")
filenames = sorted(glob.glob(folderName + "\*.csv"))
OneFrameError = np.array([])
OneFrameEnergyError = np.array([])
TotalError = np.array([])
TotalEnergyError = np.array([])
OneFrameWallBoundary = np.array([])
OneFrameBallBoundary = np.array([])
TotalWallBoundary = np.array([])
TotalBallBoundary = np.array([])

def calcRMSE(x, y):

    errorVector = np.zeros(len(y))
    for i in range(len(y)):
        errorVector[i] = np.sqrt(mean_squared_error(y[i], x[i]))
    return errorVector


def calcEnergyError(x, y):

    errorVector = np.zeros(len(y))
    for i in range(len(y)-1):
        errorVector[i+1] = np.sqrt(mean_squared_error(abs(x[i] - x[i+1]), abs(y[i] - y[i+1])))
    return errorVector


def plotError(errorVector, title):

    fig = plt.figure()
    plt.plot(errorVector)
    plt.ylabel("Average Positional Error")
    plt.xlabel("Frame")
    plt.suptitle(title)
    plt.show()
    return fig



def plotEnergyError(errorVector, title):

    fig = plt.figure()
    plt.plot(errorVector)
    plt.ylabel("Average Energy Error")
    plt.xlabel("Frame")
    plt.suptitle(title)
    plt.show()
    return fig

def plotWallBoundary(vector, title):

    fig = plt.figure()
    vector = vector.reshape(-1, 2)
    plt.plot(vector[:, 0], vector[:, 1], 'bo', markersize=1)
    outside_rate = 0
    for i in vector:
        if (i[0] < 1 or i[0] > 9 or i[1] < 1 or i[1] > 19):
            outside_rate += 1
    outside_rate = (outside_rate / vector.shape[0]) * 100

    plt.axhline(y=1, color='r')
    plt.axhline(y=19, color='r')
    plt.axvline(x=1, color='r')
    plt.axvline(x=9, color='r')
    plt.suptitle(title)
    plt.show()
    return fig

def plotBallBoundary(vector, title):

    fig = plt.figure()
    vector = vector.reshape(-1, 2)
    minimum = 0
    for i in vector:


    plt.suptitle(title)
    plt.show()
    return fig



def justText(content):

    fig = plt.figure()
    plt.text(0, 0, content, fontsize=12)
    return fig


for i in range(math.floor(len(filenames)/3)):

    TestRun = np.loadtxt(folderName + "\TestRun" + str(i+1) + ".csv", dtype=np.float64)
    TestRun = np.array(TestRun).reshape(-1, 3, 2)

    OneFramePrediction = np.loadtxt(folderName + "\OneFramePrediction" + str(i+1) + ".csv", dtype=np.float64)
    OneFramePrediction = np.array(OneFramePrediction).reshape(-1, 3, 2)
    OneFrameWallBoundary = np.append(OneFrameWallBoundary, OneFramePrediction)
    OneFrameBallBoundary = np.append(OneFrameBallBoundary. OneFramePrediction)

    TotalPrediction = np.loadtxt(folderName + "\TotalPrediction" + str(i+1) + ".csv", dtype=np.float64)
    TotalPrediction = np.array(TotalPrediction).reshape(-1, 3, 2)
    TotalWallBoundary = np.append(TotalWallBoundary, OneFramePrediction)
    TotalBallBoundary = np.append(TotalBallBoundary.OneFramePrediction)

    OneFrameError = np.append(OneFrameError, calcRMSE(TestRun, OneFramePrediction))
    OneFrameEnergyError = np.append(OneFrameEnergyError, calcEnergyError(TestRun, OneFramePrediction))
    TotalError = np.append(TotalError, calcRMSE(TestRun, TotalPrediction))
    TotalEnergyError = np.append(TotalEnergyError, calcEnergyError(TestRun, TotalPrediction))


OneFrameError = OneFrameError.reshape(-1, 60)
TotalError = TotalError.reshape(-1, 60)
OneFrameEnergyError = OneFrameEnergyError.reshape(-1, 60)
TotalEnergyError = TotalEnergyError.reshape(-1, 60)
plots = []
plots.append(plotError(np.mean(OneFrameError, axis=0), "One Frame Prediction"))
plots.append(plotError(np.mean(TotalError, axis=0), "Total Prediction"))

plots.append(plotEnergyError(np.mean(OneFrameEnergyError[:, 1:], axis=0), "One Frame Prediction"))
plots.append(plotEnergyError(np.mean(TotalEnergyError[:, 1:], axis=0), "Total Prediction"))

plots.append(plotWallBoundary(OneFrameWallBoundary, "Perceived Wall Boundary for One Frame Prediction"))
plots.append(plotWallBoundary(TotalWallBoundary, "Perceived Wall Boundary for Total Prediction"))

plots.append(plotBallBoundary(OneFrameBallBoundary, "Perceived Ball Boundary for One Frame Prediction"))
plots.append(plotBallBoundary(TotalBallBoundary, "Perceived Ball Boundary for Total Prediction"))

pp = PdfPages('Error_Plots.pdf')
for i in plots:
    pp.savefig(i)

pp.close()



