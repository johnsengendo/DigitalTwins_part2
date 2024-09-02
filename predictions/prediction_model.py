# -*- coding: utf-8 -*-
# Importing helper libraries
!pip install tabulate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sklearn as sk
import pandas as pd
from pandas import read_csv
from datetime import datetime
import math
import os

# fixing random seed for reproducibility
seed = 2022
np.random.seed(seed)

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D,Input
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
from tabulate import tabulate


from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv('/content/drive/My Drive/pcap/packets_per_sec_analysis.csv')

# Extracting the column 'packets_per_sec'
data = data['packets_per_sec']

# Calculating the index at which to split the data
split_index = int(len(data) * 0.6)

# Splitting the data into training and validation datasets
train_dataset = data[:split_index]
val_dataset = data[split_index:]

print(train_dataset.shape)
print(val_dataset.shape)

# Selecting the column 'packets_per_sec' as the feature for the model
features = ['packets_per_sec']

# train and validate
train_values = np.asarray(train_dataset.values, dtype=np.float32).reshape(-1, 1)
train_labels = np.asarray(train_dataset.values, dtype=np.float32)

val_values = np.asarray(val_dataset.values, dtype=np.float32).reshape(-1, 1)
val_labels = np.asarray(val_dataset.values, dtype=np.float32)

# imports showing many different scalers that could be adopted
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

# using the StandardScaler
SS1 = StandardScaler()
SS1.fit(train_values)

train_scaled = SS1.transform(train_values)
val_scaled = SS1.transform(val_values)

"""
The function creates a dataset for time-series analysis, specifically for window-based approach.

Parameters:
- features: A numpy array containing the input data for the model.
- labels: A numpy array containing the corresponding output data for the model.
- ahead: An integer that specifies how many steps ahead in the future the labels are.
- window_size: An integer that specifies the size of the sliding window that is used to create the input data.
- max_window_size: An integer that specifies the maximum size of the sliding window.

Returns:
- dataX: A 3D numpy array of shape (num_samples, window_size, num_features) containing the input data samples.
- labels: A 1D numpy array of shape (num_samples,) containing the corresponding labels for the input data samples.

The function calculates the number of samples that can be created based on the size of the features array, the 'ahead' parameter, and the 'max_window_size' parameter. 
It then creates a list of input data samples by sliding a window of size 'window_size' over the features array. Each input data sample is a 2D numpy array of shape 
(window_size, num_features), where 'num_features' is the number of features in the features array. 
The function returns a tuple containing two numpy arrays: the first is a 3D numpy array of shape (num_samples, window_size, num_features) containing the input data samples, 
and the second is a 1D numpy array of shape (num_samples,) containing the corresponding labels for the input data samples.
The labels are shifted 'ahead' steps into the future, and only the labels that correspond to the input data samples are included in the output array.
"""

def create_dataset_windowed(features, labels, ahead=1, window_size=1, max_window_size=400):
    samples = features.shape[0] - ahead - (max_window_size - 1)
    window_size = min(max(window_size, 1), max_window_size)

    dataX = np.array([features[(i + max_window_size - window_size):(i + max_window_size), :] for i in range(samples)])
    dataY = labels[ahead + max_window_size - 1 : ahead + max_window_size - 1 + samples]

    return dataX, dataY

"""
The PlotResults function takes in two arrays, labels and predictions, and an optional parameter binsize with a default value of 10.
The function creates a figure with four subplots and plots various visualizations to compare the labels and predictions.

Parameters:
- labels: An array containing the true values.
- predictions: An array containing the predicted values.
- binsize: An optional parameter that determines the width of the bins in the histogram. Default is 10.

Returns:
- None

The function first creates a figure with a 4x4 grid of subplots using the gridspec module from matplotlib.
It then plots the labels and predictions on the first subplot, the absolute errors between the labels and predictions on the second subplot,
a scatter plot of the labels and predictions on the third subplot, and a histogram of the absolute errors on the fourth subplot.
Finally, it displays the figure using plt.show().
"""

def PlotResults(labels,predictions,binsize = 10):
  num_samples = len(labels)

  fig = plt.figure(figsize=(16,16))
  spec = gridspec.GridSpec(ncols=4, nrows=4, figure=fig)
  ax1 = fig.add_subplot(spec[0, :])
  ax2 = fig.add_subplot(spec[1, :])
  ax3 = fig.add_subplot(spec[2:,0:2])
  ax4 = fig.add_subplot(spec[2:,2:])

  ax1.plot(labels,'k-',label='Labels')
  ax1.plot(predictions,'r-',label='Predictions')
  ax1.set_ylabel('Packets/Sec')
  ax1.legend()

  errors=np.absolute(labels-predictions)
  ax2.plot(errors,'k-')
  ax2.set_ylabel("Absolute error")

  ax3.scatter(labels,predictions)
  ax3.set_xlabel('Labels')
  ax3.set_ylabel('Predictions')

  bins = np.arange(0,(np.ceil(np.max(errors)/binsize)+1)*binsize,binsize)

  ax4.hist(errors,bins=bins)
  ax4.set_xlabel('Absolute error')
  ax4.set_ylabel('Frequency')

  plt.show()

"""
The plot_history function takes a history object as input and plots the training and validation mean absolute error (MAE) values for each epoch.

Parameters:
- history: A history object containing the training and validation metrics for each epoch.

Returns:
- None

The function uses the matplotlib library to create a line plot with the epoch numbers on the x-axis and the MAE values on the y-axis.
The training MAE values are plotted in green and the validation MAE values are plotted in red. The function also adds a legend to the plot to distinguish between the two lines.
"""
def plot_history(history):
  plt.figure(figsize = (6,4))

  plt.xlabel('Epoch')
  plt.ylabel('Mae')
  plt.plot(history.epoch, np.array(history.history['mae']),'g-',
           label='Train MAE')
  plt.plot(history.epoch, np.array(history.history['val_mae']),'r-',
           label = 'Validation MAE')
  plt.legend()
  plt.show()

"""
Here the script trains a CNN model with different window sizes for prediction.

The script uses the `create_dataset_windowed` function to create input data samples by sliding a window of size `window_size` over the features array.
The script then trains a CNN model with the input data samples and corresponding labels. The model is compiled with the Adam optimizer, mean absolute error (MAE) loss function, and MAE and MSE metrics.
The model is trained for 50 epochs with a batch size of 64. The training and validation MAE are calculated and stored in a pandas DataFrame.
The results are displayed as a table using the `tabulate` library and saved to a CSV file.

Parameters:
- window_sizes: A list of integers specifying the window sizes to use for training the model.
- train_scaled: A numpy array containing the scaled input data for training the model.
- train_labels: A numpy array containing the corresponding output data for training the model.
- val_scaled: A numpy array containing the scaled input data for validating the model.
- val_labels: A numpy array containing the corresponding output data for validating the model.

Returns:
- None

The script prints the model summary, training and validation MAE for each window size, and a table summarizing the results. The results are also saved to a CSV file.
"""

# Defining window sizes and ahead values
window_sizes = [60, 90, 120, 180, 240, 300, 360]
ahead_values = [60, 90, 120, 180, 240, 300]
results = pd.DataFrame()

# Main loop over window sizes and ahead values
for WINDOW in window_sizes:
    for AHEAD in ahead_values:

        # Preparing the data based on current WINDOW and AHEAD
        X_train_w, r_train_w = create_dataset_windowed(train_scaled, train_labels, window_size=WINDOW)
        X_val_w, r_val_w = create_dataset_windowed(val_scaled, val_labels, window_size=WINDOW)

        # Defining and compiling the CNN model
        CNNmodel = Sequential()

        CNNmodel.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu', input_shape=(WINDOW, 1)))
        CNNmodel.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
        CNNmodel.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'))
        CNNmodel.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))

        CNNmodel.add(Flatten())

        CNNmodel.add(Dense(64, activation='relu'))
        CNNmodel.add(Dense(1))
        CNNmodel.add(Activation('linear'))

        CNNmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mae', metrics=['mae', 'mse'])

        # Printing the model summary to check the shape
        print(CNNmodel.summary())

        batch_size = 64
        epochs = 50
        CNN_history = CNNmodel.fit(X_train_w, r_train_w,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   verbose=1,
                                   validation_data=(X_val_w, r_val_w),
                                   shuffle=True)

        # Predicting and calculating MAE for training and validation sets
        y_train_CNNmodel = CNNmodel.predict(X_train_w)
        y_val_CNNmodel = CNNmodel.predict(X_val_w)

        mae_train_CNNmodel = mean_absolute_error(r_train_w, y_train_CNNmodel)
        mae_val_CNNmodel = mean_absolute_error(r_val_w, y_val_CNNmodel)

        # Storing the results
        new_row = pd.DataFrame({
            'Window Size': [WINDOW],
            'Ahead': [AHEAD],
            'Train MAE': [mae_train_CNNmodel],
            'Validation MAE': [mae_val_CNNmodel]
        })
        results = pd.concat([results, new_row], ignore_index=True)

        print(f"Window size: {WINDOW}, Ahead: {AHEAD}")
        print(f"Train MAE: {mae_train_CNNmodel}")
        print(f"Validation MAE: {mae_val_CNNmodel}")

        # Ploting the results
        PlotResults(r_val_w[:1000], y_val_CNNmodel[:1000, 0])
        plot_history(CNN_history)

print("Loop stopped")

# Displaying the results as a table
print("\nResults Summary:")

table_str = tabulate(results, headers='keys', tablefmt='fancy_grid')

# Replacing the characters to create a solid line effect in the table
table_str = table_str.replace('─', '━').replace('│', '┃').replace('┼', '╋')
table_str = table_str.replace('┌', '━').replace('┐', '━').replace('└', '━').replace('┘', '━')
table_str = table_str.replace('┏', '━').replace('┓', '━').replace('┗', '━').replace('┛', '━')
table_str = table_str.replace('┣', '━').replace('┫', '━').replace('┳', '━').replace('┻', '━')

print(table_str)

# Saving the results table to a CSV file
results.to_csv('results.csv', index=False)
