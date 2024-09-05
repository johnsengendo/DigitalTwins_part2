# -*- coding: utf-8 -*-
# --------------------------------------------------------------
# Importing helper libraries
# --------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sklearn as sk
import pandas as pd
from pandas import read_csv
from datetime import datetime
import math
import os

# Fixing random seed for reproducibility
seed = 2022
np.random.seed(seed)

# --------------------------------------------------------------
# TensorFlow and keras libraries
# --------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras

# Print TensorFlow version for verification
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Input
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.losses import MAE

# --------------------------------------------------------------
# Google Drive Mounting
# --------------------------------------------------------------

# from google.colab import drive
# drive.mount('/content/drive')

# --------------------------------------------------------------
# Data loading and initial plotting
# --------------------------------------------------------------

# Loading the dataset
data = pd.read_csv('packets_per_sec_analysis.csv')

# Extracting the column 'packets_per_sec'
data = data['packets_per_sec']

# Plotting the extracted data
plt.figure(figsize=(14, 6))
plt.plot(data, label='Packets/Sec (Original)', color='blue')
plt.title('Extracted Packets/Sec Data from the pcap file')
plt.xlabel('Time (Seconds)')
plt.ylabel('Packets/Sec')
plt.legend()
plt.show()

# --------------------------------------------------------------
# Data splitting for training and validation
# --------------------------------------------------------------

# Calculating the index at which to split the data
split_index = int(len(data) * 0.6)

# Spliting the data into training and validation datasets
train_dataset = data[:split_index]
val_dataset = data[split_index:]

# Printing the shapes of the split datasets
print(train_dataset.shape)
print(val_dataset.shape)

# Selecting the column 'packets_per_sec' as the feature for our model
features = ['packets_per_sec']

# Converting the training and validation data to NumPy arrays
train_values = np.asarray(train_dataset.values, dtype=np.float32).reshape(-1, 1)
train_labels = np.asarray(train_dataset.values, dtype=np.float32)

val_values = np.asarray(val_dataset.values, dtype=np.float32).reshape(-1, 1)
val_labels = np.asarray(val_dataset.values, dtype=np.float32)

# --------------------------------------------------------------
# Data scaling using scikit-learn
# --------------------------------------------------------------

# Importing different scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer

# Example: Using standardScaler
SS1 = StandardScaler()
SS1.fit(train_values)

# Applying the scaler to training and validation data
train_scaled = SS1.transform(train_values)
val_scaled = SS1.transform(val_values)

# --------------------------------------------------------------
# Defining window sizes and prediction horizons
# --------------------------------------------------------------

window_sizes = [60, 120, 300, 360]
ahead_values = [4, 30, 60, 120, 300, 360]

# --------------------------------------------------------------
# Creating windowed datasets
# --------------------------------------------------------------

def create_dataset_windowed(features, labels, ahead=4, window_size=1, max_window_size=500):
    """
    The fuction creates a dataset for time-series analysis with window-based features.

    Parameters:
    - features: Input data for the model.
    - labels: Corresponding output labels.
    - ahead: Steps ahead to predict.
    - window_size: Size of the sliding window.
    - max_window_size: Maximum size of the sliding window.

    Returns:
    - dataX: Input data samples.
    - dataY: Corresponding labels.
    """
    samples = features.shape[0] - ahead - (max_window_size - 1)
    window_size = min(max(window_size, 1), max_window_size)

    dataX = np.array([features[(i + max_window_size - window_size):(i + max_window_size), :] for i in range(samples)])
    dataY = labels[ahead + max_window_size - 1 : ahead + max_window_size - 1 + samples]

    return dataX, dataY

# --------------------------------------------------------------
# Plotting functions for results and history
# --------------------------------------------------------------

def PlotResults(labels, predictions, binsize=10):
    """
    The function plots the comparison between actual labels and predictions.
    """
    fig = plt.figure(figsize=(16, 16))
    spec = gridspec.GridSpec(ncols=4, nrows=4, figure=fig)
    ax1 = fig.add_subplot(spec[0, :])
    ax1.plot(labels, 'k-', label='Labels')
    ax1.plot(predictions, 'r-', label='Predictions')
    ax1.set_ylabel('Packets/Sec')
    ax1.legend()
    plt.show()

def plot_history(history):
    """
    The function plots the training and validation history of the model.
    """
    plt.figure(figsize=(6, 4))
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.plot(history.epoch, np.array(history.history['mae']), 'g-', label='Train MAE')
    plt.plot(history.epoch, np.array(history.history['val_mae']), 'r-', label='Validation MAE')
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# CNN Model training and evaluation
# --------------------------------------------------------------

from sklearn.metrics import mean_absolute_error
from tabulate import tabulate

# Initializing a dictionary to store the histories of different models
histories = {}

# Defining the folder path for saving results
folder_path = '/content/drive/My Drive/pcap/'

# Initializing a DataFrame to store results
results = pd.DataFrame()

# Looping through each window size and ahead value combination
for WINDOW in window_sizes:
    for AHEAD in ahead_values:

        print(f"Training model with window size: {WINDOW}")

        # Creating windowed datasets for training and validation
        X_train_w, r_train_w = create_dataset_windowed(train_scaled, train_labels, window_size=WINDOW)
        X_val_w, r_val_w = create_dataset_windowed(val_scaled, val_labels, window_size=WINDOW)

        # Building the CNN model
        CNNmodel = Sequential()

        # Adding input layer explicitly with Input()
        CNNmodel.add(Input(shape=(WINDOW, X_train_w.shape[-1])))

        CNNmodel.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
        CNNmodel.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
        CNNmodel.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'))
        CNNmodel.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))

        CNNmodel.add(Flatten())

        # Output layer
        CNNmodel.add(Dense(64, activation='relu'))
        CNNmodel.add(Dense(1))
        CNNmodel.add(Activation('linear'))

        # Compiling the model
        CNNmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mae', metrics=['mae', 'mse'])

        # Printing the model summary to check the shape
        print(CNNmodel.summary())

        # Defining training parameters
        batch_size = 64
        epochs = 50

        # Training the model
        CNN_history = CNNmodel.fit(X_train_w, r_train_w,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   verbose=1,
                                   validation_data=(X_val_w, r_val_w),
                                   shuffle=True)

        # Storing the training history
        histories[WINDOW] = CNN_history.history

        # Making predictions
        y_train_CNNmodel = CNNmodel.predict(X_train_w)
        y_val_CNNmodel = CNNmodel.predict(X_val_w)

        # Calculating Mean Absolute Error (MAE) for training and validation sets
        mae_train_CNNmodel = mean_absolute_error(r_train_w, y_train_CNNmodel)
        mae_val_CNNmodel = mean_absolute_error(r_val_w, y_val_CNNmodel)

        # Storing the results in the DataFrame
        new_row = pd.DataFrame({
            'Window Size / Seconds in history': [WINDOW],
            'Ahead / Seconds in the future': [AHEAD],
            'Train MAE': [mae_train_CNNmodel],
            'Validation MAE': [mae_val_CNNmodel]
        })
        results = pd.concat([results, new_row], ignore_index=True)

        # Printing MAE results
        print(f"Window size: {WINDOW}, Ahead: {AHEAD}")
        print(f"Train MAE: {mae_train_CNNmodel}")
        print(f"Validation MAE: {mae_val_CNNmodel}")

        # Saving predictions to a CSV file
        predictions_df = pd.DataFrame({
            'Actual': r_val_w.flatten(),
            'Predicted': y_val_CNNmodel.flatten()
        })
        predictions_df.to_csv(f'{folder_path}predictions_window_{WINDOW}_ahead_{AHEAD}.csv', index=False)

        # Ploting the results
        PlotResults(r_val_w[:1000], y_val_CNNmodel[:1000, 0])
        plot_history(CNN_history)

# Saving the summary of results to a CSV file
results.to_csv(f'{folder_path}results_summary.csv', index=False)
print("Loop stopped")
