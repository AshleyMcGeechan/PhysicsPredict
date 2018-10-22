import numpy as np
import glob
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


predictor = keras.models.load_model('PhysicsPredict.h5')

predictor.summary()
print(predictor.get_weights())

fileName = input("\nEnter the name of the file to be predicted on. \n")
with open(fileName, 'r') as f:
    TestRun = np.loadtxt(f, dtype=np.float64)
    TestRun = np.array(TestRun).reshape(-1, 6, 2)

predictionset1 = []
predictionset2 = []
predictionset3 = []

predictionset2.append(TestRun[0:2, :, :])
predictionset2 = np.array(predictionset2)

predictionset3.append(TestRun[0:2, :, :])
predictionset3 = np.array(predictionset3)

print(predictionset2.shape)

for i in range((TestRun.shape[0]) - 2):
    predictionset1.append(TestRun[i:i+2, :, :])

predictionset1 = np.array(predictionset1)

OneFramePrediction = predictor.predict(predictionset1)

TotalPrediction = np.array([])

for i in range((TestRun.shape[0]) - 2):
    newPrediction = predictor.predict(predictionset2)
    oldPrediction = predictionset2[0, 1, :, :].flatten().reshape(1, 6, 2)
    predictionset2 = np.append(oldPrediction, newPrediction, axis=0)
    predictionset2 = np.expand_dims(predictionset2, axis=0)
    TotalPrediction = np.append(TotalPrediction, newPrediction)


SoftPrediction = np.array([])

for i in range((TestRun.shape[0]) - 2):
    newPrediction = predictor.predict(predictionset3)
    oldPrediction = predictionset3[0, 1, :, :].flatten().reshape(1, 6, 2)
    predictionset3 = np.append(oldPrediction, newPrediction, axis=0)
    predictionset3 = np.expand_dims(predictionset3, axis=0)
    SoftPrediction = np.append(SoftPrediction, newPrediction)


with open('TestRun.csv', 'w') as f:
    np.savetxt(f, TestRun[2:, :, :].flatten())

with open('OneFramePrediction.csv', 'w') as f:
    np.savetxt(f, OneFramePrediction.flatten())

with open('TotalPrediction.csv', 'w') as f:
    np.savetxt(f, TotalPrediction.flatten())


