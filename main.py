import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Admission_Predict_Ver1.1.csv')
# Display the first few rows of the dataset
print(data.head())
# Check for missing values
print(data.isnull().sum())
# Drop the 'Serial No.' column as it is not a feature
data = data.drop(columns=['Serial No.'])
# Separate features and target variable
X = data.drop(columns=['Chance of Admit '])
y = data['Chance of Admit ']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Normalize the feature data to range [0, 1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Build a Sequential neural network model

model = Sequential([
    # First hidden layer with 64 neurons and ReLU activation
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    # Second hidden layer with 32 neurons and ReLU activation
    Dense(32, activation='relu'),
    # Output layer with 1 neuron (for regression) and linear activation
    Dense(1, activation='linear')
])

# Print model architecture summary
model.summary()

# Compile the model
# - Loss: mean squared error (for regression tasks)
# - Optimizer: Adam (adaptive learning rate)
# - Metrics: mean absolute error (for evaluation)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

# Train the model
# - epochs=100: number of training iterations
# - validation_split=0.2: 20% of training data used for validation
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)
# Evaluate the model on unseen test data
test_loss, test_mae = model.evaluate(X_test, y_test)
print("Mean Absolute Error on test data:", test_mae)

# Plot training and validation loss across epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train loss', 'val loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot training and validation mean absolute error across epochs
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.legend(['train MAE', 'val MAE'])
plt.title('Model Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.show()