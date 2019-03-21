import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

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


def plotGraph(x, i, label):

    fig = plt.figure()
    plt.scatter(x[1:, i, 0],
                x[1:, i, 1],
                c=np.arange(len(x)-1),
                cmap='Reds_r',
                marker='o',
                label=label + " Ball:" + str(i+1))

    plt.scatter(x[0, i, 0],
                x[0, i, 1],
                c='red',
                marker='*',
                label="Start point")

    plt.legend()
    plt.close()
    return fig


def plotGraphRun(x, label):
    fig = plt.figure()
    plt.scatter(x[1:, 0, 0],
                x[1:, 0, 1],
                c=np.arange(len(x)-1),
                cmap='Reds_r',
                marker='o',)

    plt.scatter(x[1:, 1, 0],
                x[1:, 1, 1],
                c=np.arange(len(x)-1),
                cmap='Purples_r',
                marker='o',)

    plt.scatter(x[1:, 2, 0],
                x[1:, 2, 1],
                c=np.arange(len(x)-1),
                cmap='Blues_r',
                marker='o',)


    plt.scatter(x[0, 0, 0],
                x[0, 0, 1],
                c='red',
                marker='*',
                label="Start point. Ball 1:")

    plt.scatter(x[0, 1, 0],
                x[0, 1, 1],
                c='purple',
                marker='*',
                label="Start point. Ball 2:")

    plt.scatter(x[0, 2, 0],
                x[0, 2, 1],
                c='blue',
                marker='*',
                label="Start point. Ball 3:")


    plt.legend()
    fig.suptitle(label)
    plt.axis([0, 10, 0, 20])
    plt.close()
    return fig


def plotGraphOverlayed(x, y, z, i):

        fig = plt.figure()

        plt.scatter(x[1:, i, 0],
                    x[1:, i, 1],
                    c='cyan',
                    marker='o',
                    alpha=0.5,
                    label="Actual Run. Ball:" + str(i+1))

        plt.scatter(y[1:, i, 0],
                    y[1:, i, 1],
                    c='yellow',
                    marker='o',
                    alpha=0.5,
                    label="One frame ahead prediction. Ball:" + str(i+1))

        plt.scatter(z[1:, i, 0],
                    z[1:, i, 1],
                    c='magenta',
                    marker='o',
                    alpha=0.5,
                    label="Prediction from initial frames. Ball:" + str(i+1))

        plt.scatter(x[0, i, 0],
                    x[0, i, 1],
                    c='cyan',
                    marker='*',
                    label="Start point for Actual Run.")

        plt.scatter(y[0, i, 0],
                    y[0, i, 1],
                    c='yellow',
                    marker='*',
                    label="Start point for One Frame Prediction.")

        plt.scatter(z[0, i, 0],
                    z[0, i, 1],
                    c='magenta',
                    marker='*',
                    label="Start point for Total Prediction.")

        plt.axis([0, 10, 0, 20])
        plt.legend()
        plt.close()
        return fig


plots = []
for i in range(3):
    plots.append(plotGraphOverlayed(TestRun, OneFramePrediction, TotalPrediction, i))

for i in range(3):
    plots.append(plotGraph(TestRun, i, "Actual Run."))

for i in range(3):
    plots.append(plotGraph(OneFramePrediction, i, "One frame ahead prediction."))

for i in range(3):
    plots.append(plotGraph(TotalPrediction, i, "Prediction from initial frames."))

plots.append(plotGraphRun(TestRun, "Actual Run."))
plots.append(plotGraphRun(OneFramePrediction, "One frame ahead prediction."))
plots.append(plotGraphRun(TotalPrediction, "Prediction from initial frames."))

pp = PdfPages('Trajectory_Plots.pdf')
for i in plots:
    pp.savefig(i)

pp.close()



