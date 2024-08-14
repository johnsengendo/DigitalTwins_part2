import math
import os
import sys
import pandas as pd
import numpy as np
from functions import *
from glob import glob
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,RepeatVector,TimeDistributed,Input,Dropout
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError, MeanSquaredLogarithmicError
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# %% Hyper Parameters
TRAIN_SPLIT = 0.8  # Percentage of train set from the whole dataset
LOOK_BACK = 10  # Number of timesteps fed to the network to predict future timesteps
LOOK_AHEAD = 10  # Number of timesteps to be predicted
NB_EPOCHS = 20  # Number of iterations in the training phase
BATCH_SIZE = 32  # Number of samples per gradient update
NEURONS_1 = 300  # Number of neurons in the first LSTM layer
NEURONS_2 = 150  # Number of neurons in the second LSTM layer
NEURONS_3 = 50  # Number of neurons in the third LSTM layer
DROPOUT_RATE = 0.2  # Dropout rate for regularization

# %% Data Loading and Preprocessing
CSV_PATH = "/content/drive/MyDrive/Untitled folder/main/data/packet_analysis_2.csv"
series = pd.read_csv(CSV_PATH)
times = list(range(len(series)))

# Plotting the original series
plot_series(times, series, ylabel="Packets / second")
series_mean, series_std = series.mean() , series.std()
# Standardizing the series
scaler = StandardScaler()
series = scaler.fit_transform(series).reshape(-1, 1)

# Splitting the data into training and testing sets
train, test = split_into_train_test(series, TRAIN_SPLIT)

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
model.add(
    LSTM(
        NEURONS_3,
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

# Plotting the training history
plot_training_history(history, save_path="loss_curves.png")

# %% Model Evaluation and Prediction
pred_train = model.predict(train_x)
pred_test = model.predict(test_x)

# Rescaling the predictions and actual values

pred_train = rescale_data(pred_train, series_mean, series_std)
pred_test = rescale_data(pred_test, series_mean, series_std)
test_y = rescale_data(test_y, series_mean, series_std)
train_y = rescale_data(train_y, series_mean, series_std)

# Calculating mean absolute percentage errors
errors = [mean_absolute_percentage(test_y[:, i, :], pred_test[:, i, :]) for i in range(LOOK_AHEAD)]

# Visualizing prediction errors and save the DataFrames to CSV files
unfiltered_dfs = visualize_prediction_errors(pred_test, test_y, errors[0], errors[1], errors[2], errors[3], save_path="comparaison8.png")
filtered_dfs = visualize_filtered_prediction_errors(pred_test, test_y, errors[0], errors[1], errors[2], save_path="filtered_comparaison8.png")

for step, df in filtered_dfs.items():
    df.to_csv(f"filtered_{step}_data.csv", index=False)

for step, df in unfiltered_dfs.items():
    df.to_csv(f"unfiltered_{step}_data.csv", index=False)
