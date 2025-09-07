import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load the dataset
data = pd.read_csv('Admission_Predict_Ver1.1.csv')

# Clean column names (remove trailing spaces)
data.rename(columns=lambda x: x.strip(), inplace=True)

# Drop the 'Serial No.' column
data = data.drop(columns=['Serial No.'])

# Separate features and target variable
X = data.drop(columns=['Chance of Admit'])
y = data['Chance of Admit']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Normalize the feature data to range [0, 1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, "scaler.pkl")

# Build the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mean_absolute_error']
)

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=200,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate on test data
test_loss, test_mae = model.evaluate(X_test, y_test)
print("Mean Absolute Error on test data:", test_mae)

# Save the model
model.save("admission_model.h5")
print("Model and scaler saved successfully.")
