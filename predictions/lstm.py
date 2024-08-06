import math
import numpy as np
from glob import glob
import sys
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.svm import SVR
import tensorflow as tf
from keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, GRU, Input
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError, MeanSquaredLogarithmicError
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import MeanAbsolutePercentageError as MAPE

# %% Hyper Parameters
train_split = 0.8       # Percentage of train set from the whole dataset
look_back = 100         # Number of timesteps fed to the network to predict future timesteps
look_ahead = 20         # Number of timesteps to be predicted
nb_epochs = 3           # Number of iterations in the training phase
batch_size = 10         # Number of samples per gradient update
neuron1 = 100           # Number of neurons in the first layer
neuron2 = 100           # Number of neurons in the second layer
neuron3 = 50            # Number of neurons in the third layer

#%%
csv_path = "/content/drive/My Drive/project/packet_analysis_2.csv"  # Path to CSV file
series = pd.read_csv(csv_path)
times  = list(range(len(series)))


def split_into_train_test(full_dataset, training_proportion):
    """
    Dividing the full dataset into training and testing sets.

    Parameters:
    full_dataset (numpy.ndarray): The complete dataset.
    training_proportion (float): Proportion of the dataset to be used for training.

    Returns:
    tuple: Training and testing datasets.
    """
    training_size = int(len(full_dataset) * training_proportion)
    return full_dataset[:training_size], full_dataset[training_size:]


def generate_input_output_datasets(time_series, past_steps=1, future_steps=1):
    """
    Generating input and output datasets for time series prediction.

    Parameters:
    time_series (numpy.ndarray): The series of Network Traffic Volume.
    past_steps (int): Number of previous time steps to use as input.
    future_steps (int): Number of future time steps to predict.

    Returns:
    tuple: Arrays of input data (X) and corresponding outputs (y).
    """
    input_data, output_data = [], []
    for i in range(len(time_series) - past_steps - future_steps + 1):
        input_window = time_series[i:(i + past_steps)]
        input_data.append(input_window)
        output_data.append(time_series[i + past_steps:i + past_steps + future_steps])
    return np.array(input_data), np.array(output_data)

def rescale_data(normalized_data, original_mean, original_std):
    """
    Rescales normalized data to its original scale.

    Parameters:
    normalized_data (numpy.ndarray): Normalized data.
    original_mean (float): Mean of the original data.
    original_std (float): Standard deviation of the original data.

    Returns:
    numpy.ndarray: Rescaled data.
    """
    for x in np.nditer(normalized_data, op_flags=['readwrite']):
        x[...] = x * original_std + original_mean
    return normalized_data


def plot_series(time, series, format="-", start=0, end=None, figsize=(10,6), xlabel="Time", ylabel="Paclets per Second", path="test.png"):
    """
    Ploting a time series.

    Parameters:
    time (numpy.ndarray): Time indices.
    series (numpy.ndarray): Data series.
    format (str): Line format for plotting.
    start (int): Starting index for plotting.
    end (int): Ending index for plotting.
    figsize (tuple): Size of the figure.
    """
    fig = plt.figure(1, figsize)
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def mean_absolute_percentage(y, y_pred):

    """
    Computeing the Mean Absolute Percentage Error (MAPE).

    Parameters:
    y (numpy.ndarray): True values.
    y_pred (numpy.ndarray): Predicted values.

    Returns:
    float: MAPE value.
    """
    return np.mean(np.abs((y - y_pred) / y)) * 100

def visualize_prediction_errors(predicted_values, true_values, error_1, error_2, error_3, error_4, save_path="test.png"):
    """
    Visualizing 1, 4, 8, and 16 step ahead prediction errors.

    Parameters:
    predicted_values (numpy.ndarray): Predicted values.
    true_values (numpy.ndarray): True values.
    error_1, error_2, error_3, error_4 (float): MAPE for different steps ahead predictions.
    save_path (str): Path to save the plot.
    """
    fig = plt.figure(1, (18, 13))
    plt.subplot(221)
    plt.plot(true_values[:, 0, :], label="Observed")
    plt.plot(predicted_values[:, 0, :], color="red", label="Predicted, MAPE: " + str(round(error_1, 5)) + "%")
    plt.title("1 step ahead prediction")
    plt.ylabel("Number of Packets / second")
    plt.legend(loc=1, fontsize=8, framealpha=0.8)

    plt.subplot(222)
    plt.plot(predicted_values[:, 3, :], color="red", label="Predicted, MAPE: " + str(round(error_2, 5)) + "%")
    plt.plot(true_values[:, 3, :], label="Observed")
    plt.title("4 step ahead prediction")
    plt.legend(loc=1, fontsize=8, framealpha=0.8)

    plt.subplot(223)
    plt.plot(predicted_values[:, 7, :], color="red", label="Predicted, MAPE: " + str(round(error_3, 5)) + "%")
    plt.plot(true_values[:, 7, :], label="Observed")
    plt.title("8 step ahead prediction")
    plt.legend(loc=1, fontsize=8, framealpha=0.8)

    plt.subplot(224)
    plt.plot(predicted_values[:, 15, :], color="red", label="Predicted, MAPE: " + str(round(error_4, 5)) + "%")
    plt.plot(true_values[:, 15, :], label="Observed")
    plt.title("16 step ahead prediction")
    plt.legend(loc=1, fontsize=8, framealpha=0.8)

    plt.savefig(save_path)
    plt.close()

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
model.add(LSTM(neuron1, return_sequences=True))
model.add(LSTM(neuron1, return_sequences=True))
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
for i in range(20):
    errors.append(mean_absolute_percentage(test_y[:, i, :], pred_test[:, i, :]))

visualize_prediction_errors(pred_test, test_y, errors[0], errors[3], errors[7], errors[15], save_path="comparaison.png")