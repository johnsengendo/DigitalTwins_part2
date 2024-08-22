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
