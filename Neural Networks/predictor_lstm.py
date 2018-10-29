import numpy as np
import glob
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib


predictor = keras.models.load_model('PhysicsPredict.h5')
scaler = joblib.load("lstm_scaler.save")

predictor.summary()
print(predictor.get_weights())

fileName = input("\nEnter the name of the file to be predicted on. \n")
with open(fileName, 'r') as f:
    TestRun = np.loadtxt(f, dtype=np.float64)
    TestRun = np.array(TestRun).reshape(-1, 1)

TestRun = scaler.transform(TestRun)
TestRun = TestRun.reshape(-1, 1, 12)
TestRun = TestRun[0:240, :, :]

OneFramePrediction = predictor.predict(TestRun)

TestRun = TestRun.reshape(-1, 1)
TestRun = scaler.inverse_transform(TestRun)


with open('TestRun.csv', 'w') as f:
    np.savetxt(f, TestRun.flatten())

with open('OneFramePrediction.csv', 'w') as f:
    np.savetxt(f, OneFramePrediction.flatten())

# with open('TotalPrediction.csv', 'w') as f:
#    np.savetxt(f, TotalPrediction.flatten())


