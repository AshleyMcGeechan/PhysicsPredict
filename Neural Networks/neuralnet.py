import numpy as np
import glob
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

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
    data = np.array(data).reshape(-1, 6, 2)
    if run % 10 == 0:
        for i in range((data.shape[0]) - 3):
            test_data.append(data[i+1:i + 3, :, :])
            test_labels.append(data[i + 3, :, :])
    elif run % 5 == 0:
        for i in range((data.shape[0]) - 3):
            validation_data.append(data[i+1:i + 3, :, :])
            validation_labels.append(data[i + 3, :, :])
    else:
        for i in range((data.shape[0]) - 3):
            train_data.append(data[i+1:i + 3, :, :])
            train_labels.append(data[i + 3, :, :])

x = np.array(train_data)
y = np.array(train_labels)

validation_data = np.array(validation_data)
validation_labels = np.array(validation_labels)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

np.random.seed(seed)
indices = np.arange(len(x))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

train_data = np.array(x)
train_labels = np.array(y)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2, 6, 2)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(12, activation='linear'),
    keras.layers.Reshape((6, 2), input_shape=(12,))
])

model.compile(keras.optimizers.Adam(),
              loss='mse',
              metrics=['mse', 'mae', 'accuracy'])

model.summary()

results = model.fit(train_data, train_labels, batch_size=100, epochs=100, validation_data=(validation_data, validation_labels), verbose=1, shuffle=True, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])


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

[loss, mse, mae, accuracy] = model.evaluate(test_data, test_labels, verbose=1)
print("Test set loss = " + str(loss))
print("Test set MSE = " + str(mse))
print("Test set MAE = " + str(mae))
print("Test set accuracy = " + str(accuracy))

model.save('PhysicsPredict.h5')
