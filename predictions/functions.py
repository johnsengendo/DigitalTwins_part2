# functions.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

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

def plot_series(time, series, format="-", start=0, end=None, figsize=(10,6), xlabel="Time", ylabel="Packets per Second", path="test.png"):
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
    plt.plot(predicted_values[:, 4, :], color="red", label="Predicted, MAPE: " + str(round(error_3, 5)) + "%")
    plt.plot(true_values[:, 4, :], label="Observed")
    plt.title("5 step ahead prediction")
    plt.legend(loc=1, fontsize=8, framealpha=0.8)

    plt.savefig(save_path)
    plt.close()