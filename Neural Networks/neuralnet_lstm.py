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
timestep = 2

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit([[0], [40]])
joblib.dump(scaler, "lstm_scaler.save")

filenames = sorted(glob.glob("training_data\*.csv"))

for run, f in enumerate(filenames, 1):
    data = np.loadtxt(f, dtype=np.float64)
    data = data.reshape(-1, 1)
    data = scaler.transform(data)
    data = data.reshape(-1, 12)

    if run % 10 == 0:
        for i in range(240):
            test_data.append(data[i:i+timestep, :])
            test_labels.append(data[i+timestep, :])

    elif run % 5 == 0:
        for i in range(240):
            validation_data.append(data[i:i+timestep, :])
            validation_labels.append(data[i+timestep, :])

    else:
        for i in range(240):
            train_data.append(data[i:i+timestep, :])
            train_labels.append(data[i+timestep, :])


train_data = np.array(train_data, dtype=np.float64).reshape(-1, timestep, 12)
train_labels = np.array(train_labels, dtype=np.float64).reshape(-1, 12)
# train_labels = scaler.inverse_transform(train_labels)

validation_data = np.array(validation_data, dtype=np.float64).reshape(-1, timestep, 12)
validation_labels = np.array(validation_labels, dtype=np.float64).reshape(-1, 12)
# validation_labels = scaler.inverse_transform(validation_labels)

test_data = np.array(test_data, dtype=np.float64).reshape(-1, timestep, 12)
test_labels = np.array(test_labels, dtype=np.float64).reshape(-1, 12)

model = keras.Sequential([
    keras.layers.LSTM(256, batch_input_shape=(240, timestep, 12), stateful=True, return_sequences=True),
    keras.layers.LSTM(256, batch_input_shape=(240, timestep, 12), stateful=True),
    keras.layers.Dense(12, activation='linear')
])

model.compile(keras.optimizers.Adam(),
              loss='mse',
              metrics=['mse', 'mae'])

model.summary()

results = model.fit(train_data, train_labels, validation_data=(validation_data, validation_labels), batch_size=240, epochs=50, verbose=1, shuffle=False, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label='Val loss')
    plt.legend()
    plt.show()


plot_history(results)

model.save('PhysicsPredict.h5')
