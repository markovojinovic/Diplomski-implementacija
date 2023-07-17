import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import threading


def KNNprediction(df, parameter):
    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Split the dataset into input features and target variable
    input_columns = df[[df.columns[0], df.columns[1]]]
    output_column = df[df.columns[2]]

    # Assign valid feature names to the input DataFrame
    input_columns.columns = [df.columns[0], df.columns[1]]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(input_columns, output_column, test_size=0.2, random_state=42,
                                                        shuffle=True)

    # Create the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=parameter)

    # Train the classifier using the training data
    knn.fit(x_train, y_train)

    # Make predictions on the test data
    y_pred = knn.predict(x_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    input1_value = 5.0
    input2_value = 3.2
    # Example usage: Predicting a single instance
    new_instance = pd.DataFrame([[input1_value, input2_value]], columns=[df.columns[0], df.columns[1]])
    prediction = knn.predict(new_instance)
    print(f"Prediction: {prediction}")


def NeuralNetwork(df, number_of_hidden_layers, hidden_layer_function, output_layer_function, optimizer, loss_function,
                  number_of_epochs):
    # Load the dataset from a CSV file
    input_columns = [df.columns[0], df.columns[1]]
    output_column = df.columns[2]

    # Extract the input and output columns from the DataFrame
    X = df[input_columns].values.astype(np.float32)
    Y = df[output_column].values.reshape(-1, 1).astype(np.float32)

    # Define the neural network architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(number_of_hidden_layers, input_shape=(len(input_columns),),
                              activation=hidden_layer_function),
        tf.keras.layers.Dense(1, activation=output_layer_function)
    ])

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    # Define a function for training the model
    def train_model():
        # Train the model
        model.fit(X, Y, epochs=number_of_epochs, verbose=0)

        # Make predictions
        predictions = model.predict(X)
        print(predictions)

    # Create a new thread for training the model
    training_thread = threading.Thread(target=train_model)

    # Start the training thread
    training_thread.start()


def DecisionTree(df, max_dept):
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
