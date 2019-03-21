import numpy as np
import glob
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

timestep = 5

modelname = input("\nEnter the name of the model to use for predictions. \n")
predictor = keras.models.load_model(modelname, compile=False)
xscaler = joblib.load("lstm_xscaler.save")
yscaler = joblib.load("lstm_yscaler.save")

folderName = input("\nEnter the name of the folder to be predicted on. \n")
filenames = sorted(glob.glob("test_data\\" + folderName + "\*.csv"))

for run, f in enumerate(filenames, 1):
    TestRun = np.loadtxt(f, dtype=np.float64)
    TestRun = np.array(TestRun).reshape(-1, 1)

    data = TestRun.reshape(-1, 2)
    xdata = xscaler.transform(data[:, 0].reshape(-1, 1))
    ydata = yscaler.transform(data[:, 1].reshape(-1, 1))
    # adata = ascaler.transform(data[:, 2].reshape(-1, 1))
    xdata = xdata.reshape(-1, 1)
    ydata = ydata.reshape(-1, 1)
    # adata = adata.reshape(-1, 1)
    data = np.concatenate((xdata, ydata), axis=1)
    data = data.reshape(-1, 3, 2)

    TestRun = TestRun.reshape(-1, 3, 2)

    predictionset1 = []
    predictionset2 = []

    predictionset2.append(data[0:timestep, :, :])
    predictionset2 = np.array(predictionset2).reshape(-1, timestep, 6)

    for i in range(60):
        predictionset1.append(data[i:i+timestep, :, :])

    predictionset1 = np.array(predictionset1).reshape(-1, timestep, 6)

    OneFramePrediction = predictor.predict_on_batch(predictionset1)
    OneFramePrediction = np.array(OneFramePrediction).reshape(60, 60, 6)
    OneFramePrediction = np.array(OneFramePrediction[:, 0, :]).reshape(60, 6)

    TotalPrediction = predictor.predict(predictionset2)


    with open('Batch_Prediction\TestRun' + str(run) + '.csv', 'w+') as g:
        np.savetxt(g, TestRun[5:65, :, :].flatten())

    with open('Batch_Prediction\OneFramePrediction' + str(run) + '.csv', 'w+') as h:
        np.savetxt(h, OneFramePrediction.flatten())

    with open('Batch_Prediction\TotalPrediction' + str(run) + '.csv', 'w+') as i:
        np.savetxt(i, TotalPrediction.flatten())


