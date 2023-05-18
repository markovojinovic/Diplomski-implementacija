import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def KNNprediction(df):

    # Data is argument of function
    # Split the dataset into input features and target variable
    input_columns = df[[df.columns[0], df.columns[1]]]
    output_column = df[df.columns[2]]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(input_columns, output_column, test_size=0.2)

    # Create the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the classifier using the training data
    knn.fit(x_train, y_train)

    # Make predictions on the test data
    y_pred = knn.predict(x_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    input1_value = None
    input2_value = None
    # Example usage: Predicting a single instance
    new_instance = [[input1_value, input2_value]]
    prediction = knn.predict(new_instance)
    print(f"Prediction: {prediction}")


def NeuralNetwork(df):

    # Load the dataset from a CSV file
    input_columns = df[[df.columns[0], df.columns[1]]]
    output_column = df[df.columns[2]]

    # Extract the input and output columns from the DataFrame
    X = df[input_columns].values.astype(np.float32)
    Y = df[output_column].values.reshape(-1, 1).astype(np.float32)

    # Define the neural network architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_shape=(len(input_columns),), activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X, Y, epochs=1000, verbose=0)

    # Make predictions
    predictions = model.predict(X)
    print(predictions)


def DecisionTree(df):

    # Load the dataset from a CSV file
    input_columns = df[[df.columns[0], df.columns[1]]]
    output_column = df[df.columns[2]]

    # Extract the input and output columns from the DataFrame
    X = df[input_columns]
    Y = df[output_column]

    # Create a decision tree classifier
    model = DecisionTreeClassifier()

    # Train the decision tree model
    model.fit(X, Y)

    # Make predictions
    predictions = model.predict(X)
    print(predictions)
