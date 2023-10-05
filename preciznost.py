import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load your data from a CSV file
data = pd.read_csv('cars.csv')  # Replace 'your_data.csv' with your CSV file path
X = data.iloc[:, :2]  # Features
y = data.iloc[:, 2]   # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameters to test
hidden_layers = [2, 5, 15]
epochs_values = [100, 400]

results = []

for num_layers in hidden_layers:
    for num_epochs in epochs_values:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
        for _ in range(num_layers):
            model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Check for NaN values in X_train and y_train
        if X_train.isnull().any().any() or y_train.isnull().any().any():
            print("NaN values found in training data. Check data preprocessing.")
            continue

        model.fit(X_train, y_train, epochs=num_epochs, verbose=0)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred = np.round(y_pred)

        # Check for NaN values in predictions
        if np.isnan(y_pred).any():
            print("NaN values found in predictions. Check training process.")
            continue

        accuracy = accuracy_score(y_test, y_pred)
        accuracy_percentage = accuracy * 100

        results.append((num_layers, num_epochs, accuracy_percentage))

# Print the results
for num_layers, num_epochs, accuracy_percentage in results:
    print(f"Hidden Layers: {num_layers}, Epochs: {num_epochs}, Accuracy (%): {accuracy_percentage:.2f}%")