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

fileName = input("\nEnter the name of the file to be predicted on. \n")
with open(fileName, 'r') as f:
    TestRun = np.loadtxt(f, dtype=np.float64)
    TestRun = np.array(TestRun).reshape(-1, 1)

data = TestRun.reshape(-1, 2)
xdata = xscaler.transform(data[:, 0].reshape(-1, 1))
ydata = yscaler.transform(data[:, 1].reshape(-1, 1))
xdata = xdata.reshape(-1, 1)
ydata = ydata.reshape(-1, 1)
data = np.concatenate((xdata, ydata), axis=1)
data = data.reshape(-1, 3, 2)
TestRun = TestRun.reshape(-1, 3, 2)

predictionset1 = []
predictionset2 = []

predictionset2.append(data[0:timestep, :, :])
predictionset2 = np.array(predictionset2).reshape(-1, timestep, 6)

for i in range(60):
    predictionset1.append(data[i:i + timestep, :, :])

predictionset1 = np.array(predictionset1).reshape(-1, timestep, 6)

OneFramePrediction = predictor.predict_on_batch(predictionset1)
OneFramePrediction = np.array(OneFramePrediction).reshape(60, 60, 6)
OneFramePrediction = np.array(OneFramePrediction[:, 0, :]).reshape(60, 6)

TotalPrediction = predictor.predict(predictionset2)


with open('TestRun.csv', 'w') as f:
    np.savetxt(f, TestRun[5:65, :, :].flatten())

with open('OneFramePrediction.csv', 'w') as f:
    np.savetxt(f, OneFramePrediction.flatten())

with open('TotalPrediction.csv', 'w') as f:
    np.savetxt(f, TotalPrediction.flatten())


