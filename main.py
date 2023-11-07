
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load your CSV data
data = pd.read_csv("dataset.csv")

data.head

X = data[['Time', 'ADS Pressure', 'P/F ratio']].values
y = data['Purity'].values

# Calculate the index to split the data into training and test sets
split_index = int(0.7 * len(X))

# Normalize input data
X_scaler = MinMaxScaler()
X = X_scaler.fit_transform(X)

# Normalize output data
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(y.reshape(-1, 1))

# Split the data into training and test sets
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=100, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape the data to be 3D (batch_size, time_steps, input_features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Train the model
model.fit(X_train, y_train, epochs=500, batch_size=50)

# Make predictions on the test set
test_predictions = model.predict(X_test)
# Make predictions on the training set
train_predictions = model.predict(X_train)

# Denormalize the predictions
denormalized_test_predictions = y_scaler.inverse_transform(test_predictions)
denormalized_train_predictions = y_scaler.inverse_transform(train_predictions)

# Denormalize the test and training sets
denormalized_y_test = y_scaler.inverse_transform(y_test)
denormalized_y_train = y_scaler.inverse_transform(y_train)

# Calculate the Mean Squared Error (MSE) for the test and training sets
test_mse = mean_squared_error(denormalized_y_test, denormalized_test_predictions)
train_mse = mean_squared_error(denormalized_y_train, denormalized_train_predictions)
print("Test MSE:", test_mse)
print("Training MSE:", train_mse)

# Merge and plot the target and predictions for the test and training sets
plt.figure(figsize=(12, 6))
plt.plot(range(len(denormalized_y_train)), denormalized_y_train, label='Training Actual', color='blue')
plt.plot(range(len(denormalized_y_train), len(denormalized_y_train) + len(denormalized_y_test)), denormalized_y_test, label='Test Actual', color='green')
plt.plot(range(len(denormalized_y_train)), denormalized_train_predictions, label='Training Predicted', color='red')
plt.plot(range(len(denormalized_y_train), len(denormalized_y_train) + len(denormalized_y_test)), denormalized_test_predictions, label='Test Predicted', color='orange')
plt.legend()
plt.title('Time Series - Target vs. Prediction')
plt.show()
