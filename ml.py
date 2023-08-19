from tkinter import filedialog

import joblib
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import threading

# global values
column1_name = ""
column2_name = ""


def train_knn(df, parameter):
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
    model = KNeighborsClassifier(n_neighbors=parameter)

    # Train the classifier using the training data
    model.fit(x_train, y_train)

    return model


def retrain_knn(model, column1, column2):
    return


def save_knn(model):
    filepath = filedialog.askdirectory()
    if filepath:
        filepath += "/model.joblib"
        joblib.dump(model, filepath)
        return True
    return False


def load_knn():
    filepath = filedialog.askopenfilename(filetypes=[("KNN files", "*.joblib")])
    if filepath:
        return joblib.load(filepath)
    return None


def predict_knn(model, column1, column2):
    # Input data for prediction
    input_data = np.array([[column1, column2]])

    # Standardize the input data using the same scaler you used during training
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Make predictions using the trained KNN model
    return model.predict(input_data_scaled)


# ======================================================================================================================

def train_neural(df, number_of_hidden_layers, hidden_layer_function, output_layer_function, optimizer, loss_function,
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

    # Create a new thread for training the model
    training_thread = threading.Thread(target=train_model)

    training_thread.start()
    training_thread.join()

    return model


def retrain_neural(model, column1, column2):
    return


def load_neural():
    filepath = filedialog.askopenfilename(filetypes=[("Neural network files", "*.h5")])
    if filepath:
        return tf.keras.models.load_model(filepath)
    return None


def save_neural(model):
    filepath = filedialog.askdirectory()
    if filepath:
        filepath += "/model.joblib"
        model.save(filepath)
        return True
    return False


def predict_neural(model, column1, column2):
    # Input data for prediction
    input_data = np.array([[column1, column2]])

    # Standardize the input data using the same scaler you used during training
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Make predictions using the loaded model
    return model.predict(input_data_scaled)


# ======================================================================================================================

def train_decision_tree(df, max_dept):
    input_columns = [df.columns[0], df.columns[1]]
    output_column = df.columns[2]

    global column1_name
    global column2_name
    column1_name = df.columns[0]
    column2_name = df.columns[1]

    # Extract the input and output data from the DataFrame
    X = df[input_columns]
    Y = df[output_column]

    # Create a decision tree classifier
    model = DecisionTreeClassifier(max_depth=max_dept)

    # Train the decision tree model
    model.fit(X, Y)

    return model


def retrain_decision_tree(model, column1, column2):
    return


def load_decision_tree():
    filepath = filedialog.askopenfilename(filetypes=[("Decision Tree files", "*.joblib")])
    if filepath:
        return joblib.load(filepath)
    return None


def save_decision_tree(model):
    filepath = filedialog.askdirectory()
    if filepath:
        filepath += "/model.joblib"
        joblib.dump(model, filepath)
        return True
    return False


def predict_decision_tree(model, column1, column2):
    # Input data for prediction
    input_data = np.array([[column1, column2]])

    # Standardize the input data using the same scaler you used during training
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Make predictions using the loaded Decision Tree model
    global column1_name
    global column2_name
    model.feature_names_ = [column1_name, column2_name]
    return model.predict(input_data_scaled)
