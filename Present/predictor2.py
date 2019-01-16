import numpy as np
import glob
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

timestep = 5

predictor = keras.models.load_model('MLP.h5')
xscaler = joblib.load("lstm_xscaler.save")
yscaler = joblib.load("lstm_yscaler.save")

fileName = input("\nEnter the name of the file to be predicted on. \n")
with open(fileName, 'r') as f:
    TestRun = np.loadtxt(f, dtype=np.float64)
    TestRun = np.array(TestRun).reshape(-1, 1)

data = TestRun.reshape(-1, 2)
xdata = xscaler.transform(data[:, ::2])
ydata = yscaler.transform(data[:, 1::2])
xdata = xdata.reshape(-1, 1)
ydata = ydata.reshape(-1, 1)
data = np.concatenate((xdata, ydata), axis=1)
data = data.reshape(-1, 3, 2)
TestRun = TestRun.reshape(-1, 3, 2)


predictor.summary()
print(predictor.get_weights())

predictionset1 = []
predictionset2 = []
predictionset3 = []

predictionset2.append(data[0:timestep, :, :])
predictionset2 = np.array(predictionset2).reshape(-1, timestep, 6)

predictionset3.append(data[0:timestep, :, :])
predictionset3 = np.array(predictionset3).reshape(-1, timestep, 6)

print(predictionset2.shape)

for i in range(16):
    predictionset1.append(data[i:i+timestep, :, :])

predictionset1 = np.array(predictionset1).reshape(-1, timestep, 6)

OneFramePrediction = predictor.predict_on_batch(predictionset1)
OneFramePrediction = np.array(OneFramePrediction).reshape(16, 16, 6)
OneFramePrediction = np.array(OneFramePrediction[:, 0, :]).reshape(16, 1, 6)

# TotalPrediction = np.array([])
TotalPrediction = predictor.predict(predictionset2)

# for i in range(16):
#     newPrediction = predictor.predict(predictionset2)
#     oldPrediction = predictionset2[0, 1:, :].flatten().reshape(1, 4, 6)
#     print(newPrediction)
#     newPrediction = newPrediction[:, 0, :].reshape(1, 1, 6)
#     predictionset2 = np.append(oldPrediction, newPrediction, axis=1).reshape(1, 5, 6)
#     TotalPrediction = np.append(TotalPrediction, newPrediction)


# SoftPrediction = np.array([])
#
# for i in range((TestRun.shape[0]) - 2):
#     newPrediction = predictor.predict(predictionset3)
#     oldPrediction = predictionset3[0, 1, :, :].flatten().reshape(1, 6, 2)
#     predictionset3 = np.append(oldPrediction, newPrediction, axis=0)
#     predictionset3 = np.expand_dims(predictionset3, axis=0)
#     SoftPrediction = np.append(SoftPrediction, newPrediction)


OneFramePrediction = np.array(OneFramePrediction).reshape(-1, 2)
xdata = xscaler.inverse_transform(OneFramePrediction[:, ::2])
ydata = yscaler.inverse_transform(OneFramePrediction[:, 1::2])
xdata = xdata.reshape(-1, 1)
ydata = ydata.reshape(-1, 1)
OneFramePrediction = np.concatenate((xdata, ydata), axis=1)

TotalPrediction = np.array(TotalPrediction).reshape(-1, 2)
xdata = xscaler.inverse_transform(TotalPrediction[:, ::2])
ydata = yscaler.inverse_transform(TotalPrediction[:, 1::2])
xdata = xdata.reshape(-1, 1)
ydata = ydata.reshape(-1, 1)
TotalPrediction = np.concatenate((xdata, ydata), axis=1)
TotalPrediction = TotalPrediction.reshape(-1, 1, 6)

with open('TestRun.csv', 'w') as f:
    np.savetxt(f, TestRun[5:21, :, :].flatten())

with open('OneFramePrediction.csv', 'w') as f:
    np.savetxt(f, OneFramePrediction.flatten())

with open('TotalPrediction.csv', 'w') as f:
    np.savetxt(f, TotalPrediction[:, :, :].flatten())


