import math
from functions import *
from glob import glob
import sys
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import tensorflow as tf
from keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Input
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError, MeanSquaredLogarithmicError
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# %% Hyper Parameters
train_split = 0.8       # Percentage of train set from the whole dataset
look_back = 10         # Number of timesteps fed to the network to predict future timesteps
look_ahead = 5         # Number of timesteps to be predicted
nb_epochs = 30           # Number of iterations in the training phase
batch_size = 20         # Number of samples per gradient update
neuron1 = 300           # Number of neurons in the first layer
neuron2 = 150           # Number of neurons in the second layer
neuron3 = 50            # Number of neurons in the third layer

#%%
csv_path = "/content/drive/MyDrive/Untitled folder/client_packets_per_sec.csv"  # Path to CSV file
series = pd.read_csv(csv_path)
times  = list(range(len(series)))

plot_series(times, series, ylabel="Packets / second")

series_mean, series_std = series.mean() , series.std()

series = preprocessing.scale(series).reshape(len(series), 1)

train , test = split_into_train_test(series, train_split)
train_x, train_y = generate_input_output_datasets(train, look_back, look_ahead)
test_x, test_y = generate_input_output_datasets(test, look_back, look_ahead)

"""
 Reshaping the training data to match the input shape expected by LSTM layers.
 The input shape for LSTM layers is [samples, time steps, features].
 In this case, 'samples' represents the number of input sequences, 'time steps' represents the number of time steps in each input sequence,
 and 'features' represents the number of features in each time step.
 Since we are working with a univariate time series, the number of features is 1.
"""
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1], 1))

test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[1], 1))

#%%
# Defining the model architecture
model = Sequential()
model.add(LSTM(neuron1, input_shape=(look_back, 1)))
model.add(RepeatVector(look_ahead))
model.add(LSTM(neuron2, return_sequences=True))
model.add(LSTM(neuron3, return_sequences=True))
model.add(LSTM(neuron3, return_sequences=True))
model.add(LSTM(neuron3, return_sequences=True))
model.add(TimeDistributed(Dense(1)))

model.compile(loss="mse", optimizer="rmsprop", metrics=["mape"])
model.summary()
#%%

history = model.fit(
    train_x,
    train_y,
    epochs=nb_epochs,
    batch_size=batch_size,
    callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=2, mode='auto')],
    validation_split=train_split,
    verbose=2
)

#%%
pred_train = model.predict(train_x)
pred_test = model.predict(test_x)

pred_train = rescale_data(pred_train, series_mean, series_std)
pred_test = rescale_data(pred_test, series_mean, series_std)
test_y = rescale_data(test_y, series_mean, series_std)
train_y = rescale_data(train_y, series_mean, series_std)

errors = []
for i in range(look_ahead):
    errors.append(mean_absolute_percentage(test_y[:, i, :], pred_test[:, i, :]))

#visualize_prediction_errors(pred_test, test_y, errors[0], errors[3], errors[7], errors[15], save_path="comparaison.png")
visualize_prediction_errors(pred_test, test_y, errors[0], errors[1], errors[2], errors[3], save_path="comparaison.png")
