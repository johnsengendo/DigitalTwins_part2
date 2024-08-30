# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tabulate import tabulate


# Set random seed for reproducibility
seed = 2022
np.random.seed(seed)

# Import TensorFlow and its components
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

# Mount Google Drive for data access
from google.colab import drive
drive.mount('/content/drive')

# Load and preprocess the dataset
data = pd.read_csv('/content/drive/My Drive/pcap/packets_per_sec_analysis.csv')
data = data['packets_per_sec']

# Split the data into training and validation datasets
split_index = int(len(data) * 0.6)
train_dataset = data[:split_index]
val_dataset = data[split_index:]

# Convert datasets to numpy arrays
train_values = np.asarray(train_dataset.values, dtype=np.float32).reshape(-1, 1)
train_labels = np.asarray(train_dataset.values, dtype=np.float32)
val_values = np.asarray(val_dataset.values, dtype=np.float32).reshape(-1, 1)
val_labels = np.asarray(val_dataset.values, dtype=np.float32)

# Standardize the data
scaler = StandardScaler()
scaler.fit(train_values)
train_scaled = scaler.transform(train_values)
val_scaled = scaler.transform(val_values)

# Function to create dataset for time-series analysis
def create_dataset_windowed(features, labels, ahead=4, window_size=1, max_window_size=500):
    samples = features.shape[0] - ahead - (max_window_size - 1)
    window_size = min(max(window_size, 1), max_window_size)

    dataX = [features[(i + max_window_size - window_size):(i + max_window_size), :] for i in range(samples)]
    return np.array(dataX), labels[ahead + max_window_size - 1:]

# Function to plot results
def plot_results(labels, predictions, binsize=10):
    num_samples = len(labels)
    fig = plt.figure(figsize=(16, 16))
    spec = gridspec.GridSpec(ncols=4, nrows=4, figure=fig)
    ax1 = fig.add_subplot(spec[0, :])
    ax2 = fig.add_subplot(spec[1, :])
    ax3 = fig.add_subplot(spec[2:, 0:2])
    ax4 = fig.add_subplot(spec[2:, 2:])

    ax1.plot(labels, 'k-', label='Labels')
    ax1.plot(predictions, 'r-', label='Predictions')
    ax1.set_ylabel('Packets/Sec')
    ax1.legend()

    errors = np.absolute(labels - predictions)
    ax2.plot(errors, 'k-')
    ax2.set_ylabel("Absolute error")

    ax3.scatter(labels, predictions)
    ax3.set_xlabel('Labels')
    ax3.set_ylabel('Predictions')

    bins = np.arange(0, (np.ceil(np.max(errors) / binsize) + 1) * binsize, binsize)
    ax4.hist(errors, bins=bins)
    ax4.set_xlabel('Absolute error')
    ax4.set_ylabel('Frequency')

    plt.show()

# Function to plot training history
def plot_history(history):
    plt.figure(figsize=(6, 4))
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.plot(history.epoch, np.array(history.history['mae']), 'g-', label='Train MAE')
    plt.plot(history.epoch, np.array(history.history['val_mae']), 'r-', label='Validation MAE')
    plt.legend()
    plt.show()

# Train and evaluate CNN model with various window sizes
window_sizes = [60, 90, 120, 180, 240, 300, 360, 420, 480]
histories = {}
results = pd.DataFrame(columns=['Window Size', 'Train MAE', 'Validation MAE'])

for window in window_sizes:
    print(f"Training model with window size: {window}")

    X_train_w, r_train_w = create_dataset_windowed(train_scaled, train_labels, window_size=window)
    X_val_w, r_val_w = create_dataset_windowed(val_scaled, val_labels, window_size=window)

    # Define the CNN model
    model = Sequential([
        Conv1D(filters=128, kernel_size=4, padding='same', activation='relu', input_shape=(window, X_train_w.shape[-1])),
        Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'),
        Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'),
        Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mae', metrics=['mae', 'mse'])
    print(model.summary())

    # Train the model
    history = model.fit(X_train_w, r_train_w, batch_size=64, epochs=50, verbose=1, validation_data=(X_val_w, r_val_w), shuffle=True)
    histories[window] = history.history

    # Evaluate the model
    y_train_pred = model.predict(X_train_w)
    y_val_pred = model.predict(X_val_w)

    mae_train = np.mean(np.abs(r_train_w - y_train_pred))
    mae_val = np.mean(np.abs(r_val_w - y_val_pred))

    results = results.append({'Window Size': window, 'Train MAE': mae_train, 'Validation MAE': mae_val}, ignore_index=True)

    print(f"Train MAE: {mae_train}, Validation MAE: {mae_val}")

    plot_results(r_val_w[:1000], y_val_pred[:1000, 0])
    plot_history(history)

# Display results summary
print("\nResults Summary:")
print(tabulate(results, headers='keys', tablefmt='pretty'))

# Save results to CSV
results.to_csv('results.csv', index=False)
