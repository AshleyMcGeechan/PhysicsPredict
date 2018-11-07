import numpy as np
import glob
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

timestep = 2

predictor = keras.models.load_model('PhysicsPredict.h5')
scaler = joblib.load("lstm_scaler.save")

predictor.summary()

fileName = input("\nEnter the name of the file to be predicted on. \n")
with open(fileName, 'r') as f:
    TestRun = np.loadtxt(f, dtype=np.float64)
    TestRun = np.array(TestRun).reshape(-1, 1)

TestRun = scaler.transform(TestRun)
TestRun = TestRun.reshape(-1, 12)

predictionset1 = []
for i in range(256):
    predictionset1.append(TestRun[i:i+timestep, :])

predictionset1 = np.array(predictionset1).reshape(-1, timestep, 12)
print(predictionset1.shape)
OneFramePrediction = predictor.predict(predictionset1, batch_size=256)
OneFramePrediction = OneFramePrediction.reshape(-1, 1)
OneFramePrediction = scaler.inverse_transform(OneFramePrediction)

TotalPrediction = np.array([])
predictionset2 = np.zeros((256, 2, 12))
predictionset2[0, :, :] = TestRun[0:timestep, :].reshape(1, timestep, 12)
print(predictionset2.shape)

for i in range(256):
    print(i)
    newPrediction = predictor.predict(predictionset2)
    newPrediction = np.expand_dims(newPrediction[i, :], axis=0)
    oldPrediction = predictionset2[0, 1, :].flatten().reshape(1, 12)
    predictionset2 = np.append(oldPrediction, newPrediction, axis=0)
    predictionset2 = np.expand_dims(predictionset2, axis=0)
    TotalPrediction = np.append(TotalPrediction, newPrediction)

TotalPrediction = TotalPrediction.reshape(-1, 1)
TotalPrediction = scaler.inverse_transform(TotalPrediction)

TestRun = TestRun.reshape(-1, 1)
TestRun = scaler.inverse_transform(TestRun)


with open('TestRun.csv', 'w') as f:
    np.savetxt(f, TestRun.flatten())

with open('OneFramePrediction.csv', 'w') as f:
    np.savetxt(f, OneFramePrediction.flatten())

with open('TotalPrediction.csv', 'w') as f:
    np.savetxt(f, TotalPrediction.flatten())


