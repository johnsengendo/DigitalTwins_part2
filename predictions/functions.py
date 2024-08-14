# functions.py
import numpy as np
import pandas as pd
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


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def visualize_filtered_prediction_errors(predicted_values, true_values, error_1, error_2, error_3, save_path="filtered_test.png"):
    """
    Visualizes 1, 4, and 5 step ahead prediction errors after truncating values to the integer part and filtering out unique values.

    Parameters:
    predicted_values (numpy.ndarray): Predicted values.
    true_values (numpy.ndarray): True values.
    error_1, error_2, error_3 (float): MAPE for different steps ahead predictions.
    save_path (str): Path to save the plot.

    Returns:
    dict of pandas.DataFrame: DataFrames containing the filtered and truncated predicted and true values, stored separately.
    """
    fig = plt.figure(1, (18, 13))

    # Truncate values to the integer part
    truncated_pred = np.floor(predicted_values).astype(int)
    truncated_true = np.floor(true_values).astype(int)

    # Filter unique values and reshape to 1D using numpy.unique
    filtered_pred_1 = np.unique(truncated_pred[:, 0, :].reshape(-1), return_counts=False)
    filtered_true_1 = np.unique(truncated_true[:, 0, :].reshape(-1), return_counts=False)
    plt.subplot(221)
    plt.plot(filtered_true_1, label="Observed")
    plt.plot(filtered_pred_1, color="red", label="Predicted, MAPE: " + str(round(error_1, 5)) + "%")
    plt.title("1 step ahead prediction (Filtered)")
    plt.ylabel("Number of Packets / second")
    plt.legend(loc=1, fontsize=8, framealpha=0.8)

    filtered_pred_4 = np.unique(truncated_pred[:, 3, :].reshape(-1), return_counts=False)
    filtered_true_4 = np.unique(truncated_true[:, 3, :].reshape(-1), return_counts=False)
    plt.subplot(222)
    plt.plot(filtered_pred_4, color="red", label="Predicted, MAPE: " + str(round(error_2, 5)) + "%")
    plt.plot(filtered_true_4, label="Observed")
    plt.title("4 step ahead prediction (Filtered)")
    plt.legend(loc=1, fontsize=8, framealpha=0.8)

    filtered_pred_5 = np.unique(truncated_pred[:, 4, :].reshape(-1), return_counts=False)
    filtered_true_5 = np.unique(truncated_true[:, 4, :].reshape(-1), return_counts=False)
    plt.subplot(223)
    plt.plot(filtered_pred_5, color="red", label="Predicted, MAPE: " + str(round(error_3, 5)) + "%")
    plt.plot(filtered_true_5, label="Observed")
    plt.title("5 step ahead prediction (Filtered)")
    plt.legend(loc=1, fontsize=8, framealpha=0.8)

    plt.savefig(save_path)
    plt.close()

    # Create DataFrames separately for predicted and true values, with truncated values
    df_pred_1 = pd.DataFrame({"Filtered_Predicted_1": filtered_pred_1})
    df_true_1 = pd.DataFrame({"Filtered_True_1": filtered_true_1})

    df_pred_4 = pd.DataFrame({"Filtered_Predicted_4": filtered_pred_4})
    df_true_4 = pd.DataFrame({"Filtered_True_4": filtered_true_4})

    df_pred_5 = pd.DataFrame({"Filtered_Predicted_5": filtered_pred_5})
    df_true_5 = pd.DataFrame({"Filtered_True_5": filtered_true_5})

    # Returning a dictionary of DataFrames
    return {
        "filtered_pred_1": df_pred_1,
        "filtered_true_1": df_true_1,
        "filtered_pred_4": df_pred_4,
        "filtered_true_4": df_true_4,
        "filtered_pred_5": df_pred_5,
        "filtered_true_5": df_true_5
    }
  
def visualize_prediction_errors(predicted_values, true_values, error_1, error_2, error_3, error_4, save_path="test.png"):
    """
    Visualizing 1, 4, and 5 step ahead prediction errors.

    Parameters:
    predicted_values (numpy.ndarray): Predicted values.
    true_values (numpy.ndarray): True values.
    error_1, error_2, error_3 (float): MAPE for different steps ahead predictions.
    save_path (str): Path to save the plot.

    Returns:
    dict of pandas.DataFrame: DataFrames containing the predicted and true values.
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
    
    # Create DataFrames for the unfiltered data
    df_pred_1 = pd.DataFrame({"Predicted_1": predicted_values[:, 0, :].reshape(-1), "True_1": true_values[:, 0, :].reshape(-1)})
    df_pred_4 = pd.DataFrame({"Predicted_4": predicted_values[:, 3, :].reshape(-1), "True_4": true_values[:, 3, :].reshape(-1)})
    df_pred_5 = pd.DataFrame({"Predicted_5": predicted_values[:, 4, :].reshape(-1), "True_5": true_values[:, 4, :].reshape(-1)})
    
    return {"1_step": df_pred_1, "4_step": df_pred_4, "5_step": df_pred_5}

def plot_training_history(history, save_path=None):
    """
    Plots the training and validation loss curves, as well as the Mean Absolute Error (MAE) curves.

    Parameters:
    history (History): The history object returned by the model.fit() method.
    save_path (str, optional): Path to save the plot. If None, the plot will not be saved.

    Returns:
    None
    """
    plt.figure(figsize=(12, 6))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Plot training & validation MAE values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
