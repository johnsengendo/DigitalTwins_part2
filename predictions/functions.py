# functions.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from config import *

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

def plot_series(time, series, format="-", start=0, end=None, figsize=(12,8), dpi=300, xlabel="Time", ylabel="Packets per Second", path="test.png"):
    """
    Plotting a time series with high resolution.

    Title: Visualizing Network Traffic: Packets per Second Over Time

    Parameters:
    time (numpy.ndarray): Time indices.
    series (numpy.ndarray): Data series.
    format (str): Line format for plotting.
    start (int): Starting index for plotting.
    end (int): Ending index for plotting.
    figsize (tuple): Size of the figure.
    dpi (int): Dots per inch for the figure.
    """
    fig = plt.figure(1, figsize=figsize, dpi=dpi)
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.title("Visualizing network traffic extracted from the pcap file: Over Time")
    plt.savefig(path)
    plt.close()

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
  
def visualize_prediction_errors(predicted_values, true_values, error_1, error_2, error_3, ahead_value, save_path="test.png"):
    """
    Visualizing 1, 4, and 5 step ahead prediction errors for a specific ahead value.

    Parameters:
    predicted_values (numpy.ndarray): Predicted values.
    true_values (numpy.ndarray): True values.
    error_1, error_2, error_3 (float): MAPE for different steps ahead predictions.
    ahead_value (int): The current ahead value being visualized.
    save_path (str): Path to save the plot.

    Returns:
    dict of pandas.DataFrame: DataFrames containing the predicted and true values.
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 15), dpi=600)
    fig.delaxes(axs[1, 1])  # Remove the unused subplot

    titles = ["1 step prediction", "4 step prediction", "5 step prediction"]
    errors = [error_1, error_2, error_3]
    indices = [(0, 0), (0, 1), (1, 0)]

    for i, (row, col) in enumerate(indices):
        ax = axs[row, col]
        ax.plot(true_values[:, i, :], label="Observed", color='blue')
        ax.plot(predicted_values[:, i, :], color="red", label=f"Predicted, MAPE: {errors[i]:.5f}%")
        ax.set_title(titles[i], fontsize=14)
        ax.set_ylabel("Number of Packets / second", fontsize=12)
        ax.set_xlabel("Time", fontsize=12)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    
    # Add a main title that includes the ahead value    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    # Create DataFrames for the unfiltered data
    df_pred_1 = pd.DataFrame({"Predicted_1": predicted_values[:, 0, :].reshape(-1), "True_1": true_values[:, 0, :].reshape(-1)})
    df_pred_4 = pd.DataFrame({"Predicted_4": predicted_values[:, 3, :].reshape(-1), "True_4": true_values[:, 3, :].reshape(-1)})
    df_pred_5 = pd.DataFrame({"Predicted_5": predicted_values[:, 4, :].reshape(-1), "True_5": true_values[:, 4, :].reshape(-1)})
    
    return {"1_step": df_pred_1, "4_step": df_pred_4, "5_step": df_pred_5}

def plot_training_history(history, save_path=None):
    """
    Plots the training and validation loss curves.

    Parameters:
    history (History): The history object returned by the model.fit() method.
    save_path (str, optional): Path to save the plot. If None, the plot will not be saved.

    Returns:
    None
    """
    plt.figure(figsize=(12, 6))

    # Plot training & validation loss values
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    if save_path:
        plt.savefig(save_path)
    plt.show()

