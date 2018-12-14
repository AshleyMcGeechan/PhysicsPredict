import numpy as np
import glob
import math
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

train_data = []
train_data2 = []
train_labels = []
validation_data = []
validation_data2 = []
validation_labels = []
test_data = []
test_labels = []
seed = 2256
timestep = 5

xscaler = MinMaxScaler(feature_range=(0, 1))
xscaler = xscaler.fit([[1], [9]])
joblib.dump(xscaler, "lstm_xscaler.save")

yscaler = MinMaxScaler(feature_range=(0, 1))
yscaler = yscaler.fit([[1], [19]])
joblib.dump(yscaler, "lstm_yscaler.save")

filenames = sorted(glob.glob("training_data\ThreeBalls\*.csv"))

for run, f in enumerate(filenames, 1):
    data = np.loadtxt(f, dtype=np.float64)
    data = data.reshape(-1, 2)
    xdata = xscaler.transform(data[:, ::2])
    ydata = yscaler.transform(data[:, 1::2])
    xdata = xdata.reshape(-1, 1)
    ydata = ydata.reshape(-1, 1)
    data = np.concatenate((xdata, ydata), axis=1)
    data = data.reshape(-1, 6)

    if run % 10 == 0:
        for i in range(128):
            validation_data.append(data[i:i+timestep, :])
            validation_data2.append(data[i+timestep-1:i+timestep+timestep-1, :])
            validation_labels.append(data[i+timestep:i+timestep+timestep, :])

    else:
        for i in range(128):
            train_data.append(data[i:i+timestep, :])
            train_data2.append(data[i+timestep-1:i+timestep+timestep-1, :])
            train_labels.append(data[i+timestep:i+timestep+timestep, :])


train_data = np.array(train_data, dtype=np.float64).reshape(-1, timestep, 6)
train_data2 = np.array(train_data, dtype=np.float64).reshape(-1, timestep, 6)
train_labels = np.array(train_labels, dtype=np.float64).reshape(-1, timestep, 6)

validation_data = np.array(validation_data, dtype=np.float64).reshape(-1, timestep, 6)
validation_data2 = np.array(validation_data, dtype=np.float64).reshape(-1, timestep, 6)
validation_labels = np.array(validation_labels, dtype=np.float64).reshape(-1, timestep, 6)

batch = train_data.shape[0]


def stepDecay(epoch, lr):
    return lr * 0.5


def root_mean_squared_error(y_true, y_pred):
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1))


def define_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = keras.layers.Input(shape=(None, n_input))
    encoder = keras.layers.LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]



    # define training decoder
    decoder_inputs = keras.layers.Input(shape=(None, n_output))

    decoder_lstm = keras.layers.LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    decoderA = keras.layers.LSTM(n_units, return_sequences=True)
    decoderB = keras.layers.LSTM(n_units, return_sequences=True)
    decoderC = keras.layers.LSTM(n_units, return_sequences=True)

    decoder_outputs = decoderA(decoder_outputs)
    decoder_outputs = decoderB(decoder_outputs)
    decoder_outputs = decoderC(decoder_outputs)

    decoder_denseA = keras.layers.Dense(n_units, activation='relu')
    decoder_denseB = keras.layers.Dense(n_units, activation='relu')
    decoder_denseC = keras.layers.Dense(n_units, activation='relu')

    decoder_dense_linear = keras.layers.Dense(n_output, activation='linear')

    decoder_outputs = decoder_denseA(decoder_outputs)
    decoder_outputs = decoder_denseB(decoder_outputs)
    decoder_outputs = decoder_denseC(decoder_outputs)

    decoder_outputs = decoder_dense_linear(decoder_outputs)

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # define inference encoder
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    # define inference decoder
    decoder_state_input_h = keras.layers.Input(shape=(n_units,))
    decoder_state_input_c = keras.layers.Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_outputs = decoderA(decoder_outputs)
    decoder_outputs = decoderB(decoder_outputs)
    decoder_outputs = decoderC(decoder_outputs)

    decoder_outputs = decoder_denseA(decoder_outputs)
    decoder_outputs = decoder_denseB(decoder_outputs)
    decoder_outputs = decoder_denseC(decoder_outputs)

    decoder_outputs = decoder_dense_linear(decoder_outputs)

    decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    # return all models
    return model, encoder_model, decoder_model

# 0.0027
model = keras.Sequential([
    # keras.layers.LSTM(1024, input_shape=(timestep, 6), stateful=False, return_sequences=True, unroll=False),
    # keras.layers.LSTM(1024, input_shape=(timestep, 6), stateful=False, return_sequences=False, unroll=False),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(6, activation='linear'),
])


model2, encoder, decoder = define_models(6, 6, 500)

model2.compile(keras.optimizers.Adam(lr=0.001, clipnorm=1.0, amsgrad=True),
              loss='mse',
              metrics=['mse', 'mae'])

model.compile(keras.optimizers.Adam(lr=0.001, clipnorm=1.0, amsgrad=True),
              loss='mse',
              metrics=['mse', 'mae'])

lrate = keras.callbacks.LearningRateScheduler(stepDecay)
early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

# results = model.fit(train_data, train_labels, validation_data=(validation_data, validation_labels), batch_size=32, epochs=100, verbose=1, shuffle=True, callbacks=[early, lrate])
results = model2.fit([train_data, train_data2], train_labels, validation_data=([validation_data, validation_data2], validation_labels), batch_size=16, epochs=100, verbose=1, shuffle=True, callbacks=[early, reduction])


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

# model.save('MLP.h5')
model2.save('Seq2Seq.h5')
encoder.save('encoder.h5')
decoder.save('decoder.h5')


