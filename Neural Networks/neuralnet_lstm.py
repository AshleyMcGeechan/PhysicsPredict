import numpy as np
import glob
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

train_data = []
train_labels = []
validation_data = []
validation_labels = []
test_data = []
test_labels = []
seed = 2256

filenames = sorted(glob.glob("training_data\*.csv"))

for run, f in enumerate(filenames, 1):
    data = np.loadtxt(f, dtype=np.float64)

    data = data.reshape(-1, 12)

    if run % 10 == 0:
        test_data.append(data[0:240, :])
        test_labels.append(data[1:241, :])

    elif run % 5 == 0:
        validation_data.append(data[0:240, :])
        validation_labels.append(data[1:241, :])

    else:
        train_data.append(data[0:240, :])
        train_labels.append(data[1:241, :])


x = np.array(train_data, dtype=np.float64)
y = np.array(train_labels, dtype=np.float64)

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(x)
x = scaler.transform(x)
y = scaler.transform(y)
joblib.dump(scaler, "lstm_scaler.save")

x = x.reshape(-1, 1, 12)
y = y.reshape(-1, 12)

validation_data = np.array(validation_data)
validation_labels = np.array(validation_labels)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

print(x.shape, y.shape)

model = keras.Sequential([
    keras.layers.LSTM(12, input_shape=(1, 12)),
    keras.layers.Dense(12, activation='relu')
])

model.compile(keras.optimizers.Adam(),
              loss='mse',
              metrics=['mse', 'mae', 'accuracy'])

model.summary()


results = model.fit(x, y, batch_size=240, epochs=50, verbose=1, shuffle=False)


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.legend()
    plt.show()


plot_history(results)

model.save('PhysicsPredict.h5')
