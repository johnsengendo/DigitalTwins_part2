import math
import os
import sys
import pandas as pd
import numpy as np
from functions_7 import *
from config import *
from glob import glob
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2


# %% Data Loading and Preprocessing
CSV_PATH = "/content/drive/MyDrive/DT_Part_2/packet_analysis.csv"
column_name = 'packets_per_sec'
column_name_2 = 'packet_size'

# Read the CSV file and select the specified column
series = pd.read_csv(CSV_PATH, usecols=[column_name])
series_2 = pd.read_csv(CSV_PATH, usecols=[column_name_2])
times = list(range(len(series)))


# Calling the  function to plot both 'Packets / second' and 'Packet_size'
plot_series(times, series, ylabel="Packets / second", path="packets_per_sec.png")

plot_series(times, series_2, ylabel="Packet Size", path="packet_size.png")

series_mean, series_std = series.mean(), series.std()

# Standardizing the training and testing data separately
train, test = split_into_train_test(series, TRAIN_SPLIT)
scaler = StandardScaler()
train = scaler.fit_transform(train).reshape(-1, 1)
test = scaler.transform(test).reshape(-1, 1)

# Initialize dictionaries to store results for each look-ahead value
errors_dict = {}
models_dict = {}

# Loop through each LOOK_AHEAD value
for LOOK_AHEAD in LOOK_AHEAD_VALUES:
    print(f"Training model with LOOK_AHEAD = {LOOK_AHEAD}")

    # Generating input-output datasets for training and testing
    train_x, train_y = generate_input_output_datasets(train, LOOK_BACK, LOOK_AHEAD)
    test_x, test_y = generate_input_output_datasets(test, LOOK_BACK, LOOK_AHEAD)

    # Reshaping the data to match the input shape expected by LSTM layers
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[1], 1))

    # %% Model Definition
    model = Sequential()
    model.add(
        LSTM(
            NEURONS_1,
            input_shape=(LOOK_BACK, 1),
            kernel_regularizer=l2(0.001),
        )
    )
    model.add(Dropout(DROPOUT_RATE))
    model.add(RepeatVector(LOOK_AHEAD))
    model.add(
        LSTM(
            NEURONS_2,
            return_sequences=True,
            kernel_regularizer=l2(0.001),
        )
    )
    model.add(Dropout(DROPOUT_RATE))
    model.add(
        LSTM(
            NEURONS_3,
            return_sequences=True,
            kernel_regularizer=l2(0.001),
        )
    )
    model.add(Dropout(DROPOUT_RATE))
    model.add(TimeDistributed(Dense(1)))

    model.compile(loss="mse", optimizer="Adam", metrics=["mae"])
    model.summary()

    # %% Model Training
    history = model.fit(
        train_x,
        train_y,
        epochs=NB_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                patience=3,
                verbose=2,
                mode="auto",
            )
        ],
        validation_split=TRAIN_SPLIT,
        verbose=2,
    )

    # Save model and training history
    models_dict[LOOK_AHEAD] = model

    # Plotting the training history
    plot_training_history(history, save_path=f"loss_curves_look_ahead_{LOOK_AHEAD}.png")

    # %% Model Evaluation and Prediction
    pred_train = model.predict(train_x)
    pred_test = model.predict(test_x)

    # Rescaling the predictions and actual values
    pred_train = rescale_data(pred_train, series_mean, series_std)
    pred_test = rescale_data(pred_test, series_mean, series_std)
    test_y = rescale_data(test_y, series_mean, series_std)
    train_y = rescale_data(train_y, series_mean, series_std)

    # Calculating mean absolute percentage errors for all timesteps
    errors = [mean_absolute_percentage(test_y[:, i, :], pred_test[:, i, :]) for i in range(LOOK_AHEAD)]
    errors_dict[LOOK_AHEAD] = errors

    # Visualizing prediction errors and save the DataFrames to CSV files
    unfiltered_dfs = visualize_prediction_errors(pred_test, test_y, errors[0], errors[1], errors[2], errors[3], save_path=f"comparaison_look_ahead_{LOOK_AHEAD}.png")
    for step, df in unfiltered_dfs.items():
        df.to_csv(f"unfiltered_{step}_data_look_ahead_{LOOK_AHEAD}.csv", index=False)

# Create a DataFrame from the MAPE results
mape_df = pd.DataFrame([errors_dict], index=["MAPE"])
print(mape_df)

# Save the DataFrame to a CSV file
mape_df.to_csv("errors_dict.csv", index=True)

print("MAPE results saved to mape_results.csv")
# After training, you can compare the errors from the different models
for LOOK_AHEAD, errors in errors_dict.items():
    print(f"Errors for LOOK_AHEAD = {LOOK_AHEAD}: {errors}")
