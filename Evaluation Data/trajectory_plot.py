import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

def plotGraph(x, y, z, i):

        fig = plt.figure()
        plt.plot(x[0, i, 0],
                 x[0, i, 1],
                 c='cyan',
                 marker='*')

        plt.plot(x[1:, i, 0],
                 x[1:, i, 1],
                 c='cyan',
                 marker='o',
                 alpha=0.5,
                 label="Actual Run")

        plt.plot(y[0, i, 0],
                 y[0, i, 1],
                 c='yellow',
                 marker='*')

        plt.plot(y[1:, i, 0],
                 y[1:, i, 1],
                 c='yellow',
                 marker='o',
                 alpha=0.5,
                 label="One frame ahead prediction")

        plt.plot(z[0, i, 0],
                 z[0, i, 1],
                 c='magenta',
                 marker='*')

        plt.plot(z[1:, i, 0],
                 z[1:, i, 1],
                 c='magenta',
                 marker='o',
                 alpha=0.5,
                 label="Prediction from initial frames")

        plt.axis([0, 20, 0, 40])
        plt.legend()
        return fig

plot1 = plotGraph(TestRun, OneFramePrediction, TotalPrediction, 0)
plot2 = plotGraph(TestRun, OneFramePrediction, TotalPrediction, 1)
plot3 = plotGraph(TestRun, OneFramePrediction, TotalPrediction, 2)
plot4 = plotGraph(TestRun, OneFramePrediction, TotalPrediction, 3)
plot5 = plotGraph(TestRun, OneFramePrediction, TotalPrediction, 4)
plot6 = plotGraph(TestRun, OneFramePrediction, TotalPrediction, 5)

pp = PdfPages('Trajectory_Plots.pdf')
pp.savefig(plot1)
pp.savefig(plot2)
pp.savefig(plot3)
pp.savefig(plot4)
pp.savefig(plot5)
pp.savefig(plot6)

pp.close()



