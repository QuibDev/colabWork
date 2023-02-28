import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from google.colab import drive
drive.mount("/content/gdrive")

# Define the input attributes and output variable
features = ['co','NO','No2','NOx','O3','PM10','PM2.5','SO2','WIND_DIREC','WIND_SPEED']
target = 'PM2.5'

# Load the dataset
df = pd.read_csv('/content/gdrive/My Drive/dataset_1.csv', usecols=['co','NO','No2','NOx','O3','PM10','PM2.5','SO2','WIND_DIREC','WIND_SPEED'])
dataset = df.to_numpy()
print(dataset.shape)

# Convert the dataframe to a numpy array
dataset = df.values.astype('float32')

# Split the dataset into training and testing sets
# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]

# Scale the input data to a range of 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Split the input and output data for training and testing sets
train_X, train_y = train_data[:, :-1], train_data[:, -1]
test_X, test_y = test_data[:, :-1], test_data[:, -1]

# Reshape the input data to be 3-dimensional for the LSTM-CNN model
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# Define the LSTM-CNN model architecture
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(1, 9)))
model.add(MaxPooling1D(pool_size=1))
model.add(Bidirectional(LSTM(50)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Train the LSTM-CNN model
history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# Define the LSTM-CNN model
model = Sequential()
model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(10, 1)))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(train_X, train_y, epochs=100, batch_size=32, validation_data=(test_X, test_y))

# Evaluate the model on the test data
mse = model.evaluate(test_X, test_y)
print('Mean Squared Error:', mse)

# Make predictions on new data
new_data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
new_data_scaled = scaler.transform(new_data)
new_data_reshaped = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))
prediction = model.predict(new_data_reshaped)
print('Predicted PM2.5:', prediction)

