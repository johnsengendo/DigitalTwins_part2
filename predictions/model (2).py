# Importing necessary libraries and modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, BatchNormalization
from keras import regularizers
from keras.utils import register_keras_serializable
import matplotlib.pyplot as plt

# %% Data Loading and Preprocessing
CSV_PATH = "/content/drive/MyDrive/Untitled folder/main/data/packet_analysis_2.csv"
# Features and labels
data_final = pd.read_csv(CSV_PATH)
features = ['packets_per_sec']
labels = data_final['packets_per_sec']

# Calculating indices for the split
n = len(data_final)
train_end = int(0.5 * n)
val_end = int(0.75 * n)

# Splitting the data sequentially
train_data = data_final.iloc[:train_end]
val_data = data_final.iloc[train_end:val_end]
test_data = data_final.iloc[val_end:]

train_labels = labels.iloc[:train_end]
val_labels = labels.iloc[train_end:val_end]
test_labels = labels.iloc[val_end:]

# Converting to numpy arrays
train_values = np.asarray(train_data[features], dtype=np.float32)
train_labels = np.asarray(train_labels, dtype=np.float32)

val_values = np.asarray(val_data[features], dtype=np.float32)
val_labels = np.asarray(val_labels, dtype=np.float32)

test_values = np.asarray(test_data[features], dtype=np.float32)
test_labels = np.asarray(test_labels, dtype=np.float32)

# Scaling the data using MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_values)
val_scaled = scaler.transform(val_values)
test_scaled = scaler.transform(test_values)

# Setting the desired WINDOW_SIZE
WINDOW_SIZE = 10

# Defining the create_dataset_windowed function
def create_dataset_windowed(features, labels, ahead=5, window_size=WINDOW_SIZE, max_window_size=50):
    samples = features.shape[0] - ahead - (max_window_size - 1)
    window_size = min(max(window_size, 1), max_window_size)

    dataX = []
    for i in range(samples):
        a = features[(i + max_window_size - window_size):(i + max_window_size), :]
        dataX.append(a)
    return np.array(dataX), labels[ahead + max_window_size - 1:]

# Creating the windowed dataset from training, validation, and test data
X_train_w, r_train_w = create_dataset_windowed(train_scaled, train_labels)
X_val_w, r_val_w = create_dataset_windowed(val_scaled, val_labels)
X_test_w, r_test_w = create_dataset_windowed(test_scaled, test_labels)

# Defining the ElasticNetRegularizer class
class ElasticNetRegularizer(regularizers.Regularizer):
    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, x):
        return self.l1 * tf.reduce_sum(tf.abs(x)) + self.l2 * tf.reduce_sum(tf.square(x))

    def get_config(self):
        return {'l1': self.l1, 'l2': self.l2}

# Registering ElasticNetRegularizer with Keras
register_keras_serializable()(ElasticNetRegularizer)

# Defining the model creation function
def create_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(512, return_sequences=True,
                                 kernel_regularizer=ElasticNetRegularizer(l1=0.001, l2=0.01)), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(256, return_sequences=True,
                                 kernel_regularizer=ElasticNetRegularizer(l1=0.001, l2=0.01))))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, return_sequences=False,
                                 kernel_regularizer=ElasticNetRegularizer(l1=0.001, l2=0.01))))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu', kernel_regularizer=ElasticNetRegularizer(l1=0.001, l2=0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='linear'))

    # Compiling the model using Nadam optimizer and mean absolute error loss
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001), loss='mae')
    return model

# Creating and fitting the model
LSTM_model = create_model((WINDOW_SIZE, 1))
LSTM_history = LSTM_model.fit(X_train_w, r_train_w, epochs=40, batch_size=8, validation_data=(X_val_w, r_val_w))

# Plotting the training and validation loss curves
train_loss = LSTM_history.history['loss']
val_loss = LSTM_history.history['val_loss']

# Plotting the training and validation loss
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'b', label='Train_set MAE')
plt.plot(epochs, val_loss, 'r', label='Validation_set MAE')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# Evaluating the model performance on the test set
y_test_LSTM = LSTM_model.predict(X_test_w)
mae_test_LSTM = np.mean(np.abs(r_test_w - y_test_LSTM))

# Printing the mean absolute error from the test set
print("Test set MAE = ", mae_test_LSTM)

# Visualizing first 300 predictions from the test set
def scale_predictions_to_labels(labels, predictions):
    scale_factor = np.mean(labels) / np.mean(predictions)
    scaled_predictions = predictions * scale_factor
    return scaled_predictions

# Scaling the predictions
scaled_y_test_LSTM = scale_predictions_to_labels(r_test_w, y_test_LSTM)

# Function to plot the predictions against the true values
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:300], 'k-', label='Actual_packets_per_sec')
    plt.plot(y_pred[:300], 'r-', label='Predicted_packets_per_sec')
    plt.ylabel('packets_per_sec')
    plt.xlabel('time_steps')
    plt.title(title)
    plt.legend()
    plt.show()
# Plotting the predictions on the test set
plot_predictions(r_test_w, scaled_y_test_LSTM, 'Predictions on the Test set')
