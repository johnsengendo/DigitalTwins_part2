"""
This script contains the hyperparameters for a time series prediction model using LSTM (Long Short-Term Memory) networks.
The model is trained on a portion of the dataset (specified by TRAIN_SPLIT), and the remaining portion is used for testing.
The model uses a look-back mechanism (LOOK_BACK) to consider a certain number of previous time steps when making predictions.
The model's performance is compared for different look-ahead values (LOOK_AHEAD_VALUES), which are the number of time steps into the future that the model predicts.
The model is trained for a certain number of iterations (NB_EPOCHS), and the weights are updated in batches of a certain size (BATCH_SIZE).
The LSTM network consists of three layers, with the number of neurons in each layer specified by NEURONS_1, NEURONS_2, and NEURONS_3.
Dropout regularization is used to prevent overfitting, with the dropout rate specified by DROPOUT_RATE.
"""
# %% Hyper Parameters
TRAIN_SPLIT = 0.8  # Percentage of train set from the whole dataset
LOOK_BACK = 10  # Number of timesteps fed to the network to predict future timesteps
LOOK_AHEAD_VALUES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Different look-ahead values for comparison
NB_EPOCHS = 20  # Number of iterations in the training phase
BATCH_SIZE = 32  # Number of samples per gradient update
NEURONS_1 = 300  # Number of neurons in the first LSTM layer
NEURONS_2 = 150  # Number of neurons in the second LSTM layer
NEURONS_3 = 50  # Number of neurons in the third LSTM layer
DROPOUT_RATE = 0.2  # Dropout rate for regularization
