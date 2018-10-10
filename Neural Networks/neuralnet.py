import numpy as np
import glob
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

train_data = []
train_labels = []
seed = 2256
test_split = 0.2

filenames = sorted(glob.glob("training_data\*.csv"))

for f in filenames:
    data = np.loadtxt(f, dtype=np.float64)
    data = np.array(data).reshape(-1, 6, 2)
    for i in range((data.shape[0]) - 3):
        train_data.append(data[i:i+2, :, :])
        train_labels.append(data[i+2, :, :])

x = np.array(train_data)
y = np.array(train_labels)

np.random.seed(seed)
indices = np.arange(len(x))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

train_data = np.array(x[:int(len(x) * (1 - test_split))])
train_labels = np.array(y[:int(len(x) * (1 - test_split))])
test_data = np.array(x[int(len(x) * (1 - test_split)):])
test_labels = np.array(y[int(len(x) * (1 - test_split)):])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2, 6, 2)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(12, activation=tf.nn.relu),
    keras.layers.Reshape((6, 2), input_shape=(12,))
])

model.compile(keras.optimizers.Adam(),
              loss='mse',
              metrics=['mae', 'accuracy'])

model.summary()

results = model.fit(train_data, train_labels, batch_size=1000, epochs=100, validation_split=0.2, verbose=1, shuffle=True, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label='Val loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()


plot_history(results)

[loss, mae, accuracy] = model.evaluate(test_data, test_labels, verbose=1)
print("Test set MAE = " + str(mae))
print("Test set accuracy = " + str(accuracy))

model.save('PhysicsPredict.h5')
