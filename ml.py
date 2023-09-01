from tkinter import filedialog

import joblib
import tensorflow as tf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import threading

# global values
column1_name = ""
column2_name = ""
scaler = None


# TODO: proveriti da li treba scaler za neural


def train_knn(df, number_of_neibhours, leaf_size, p, weight, metric, algorithm):
    global scaler

    df.dropna(inplace=True)
    input_columns = df.iloc[:, :2]
    output_column = df.iloc[:, 2]
    scaler = StandardScaler()
    input_columns_scaled = scaler.fit_transform(input_columns)
    model = KNeighborsClassifier(n_neighbors=number_of_neibhours, leaf_size=leaf_size, p=p, weights=weight,
                                 metric=metric, algorithm=algorithm)
    model.fit(input_columns_scaled, output_column)

    return model


def save_knn(model, df):
    global scaler

    filetypes = [("Joblib files", "*.joblib")]
    filepath = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=filetypes)

    if filepath:
        if not filepath.endswith(".joblib"):
            filepath += ".joblib"

        objects_to_save = {
            'model': model,
            'df': df,
            'scaler': scaler
        }

        joblib.dump(objects_to_save, filepath)
        return True

    return False


def load_knn():
    filetypes = [("Joblib files", "*.joblib")]
    filepath = filedialog.askopenfilename(filetypes=filetypes)

    if filepath:
        global scaler
        loaded_objects = joblib.load(filepath)
        scaler = loaded_objects['scaler']
        return loaded_objects

    return None


def predict_knn(model, column1, column2):
    global scaler
    input_data = np.array([[column1, column2]])
    input_data_scaled = scaler.transform(input_data)

    return model.predict(input_data_scaled)


# ======================================================================================================================

def train_neural(df, number_of_hidden_layers, hidden_layer_function, output_layer_function, optimizer, loss_function,
                 number_of_epochs):
    global scaler

    df.dropna(inplace=True)
    input_columns = df.iloc[:, :2]
    output_column = df.iloc[:, 2]
    scaler = StandardScaler()
    input_columns_scaled = scaler.fit_transform(input_columns)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(number_of_hidden_layers, input_shape=(input_columns_scaled.shape[1],),
                              activation=hidden_layer_function),
        tf.keras.layers.Dense(1, activation=output_layer_function)
    ])

    def train_model():
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        model.fit(input_columns_scaled, output_column, epochs=number_of_epochs, verbose=0)

    training_thread = threading.Thread(target=train_model)
    training_thread.start()
    training_thread.join()

    return model


def retrain_neural(model, column1, column2, output, number_of_epochs):
    global scaler

    input_data = np.array([[column1, column2]])
    input_data_scaled = scaler.transform(input_data)
    output_data = np.array([output])

    model.fit(input_data_scaled, output_data, epochs=number_of_epochs)

    return


def load_neural():
    global scaler

    filetypes = [("Joblib files", "*.joblib")]
    filepath = filedialog.askopenfilename(filetypes=filetypes)

    if filepath:
        # Load DataFrame and scaler using joblib
        loaded_data = joblib.load(filepath)

        # Load TensorFlow model using tf.keras.models.load_model()
        model_path = filepath + "_model"
        loaded_model = tf.keras.models.load_model(model_path)

        # Extract DataFrame, scaler, and model
        loaded_df = loaded_data['df']
        scaler = loaded_data['scaler']

        return loaded_model, loaded_df

    return None, None


def save_neural(model, df):
    global scaler

    filetypes = [("Joblib files", "*.joblib")]
    filepath = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=filetypes)

    if filepath:
        if not filepath.endswith(".joblib"):
            filepath += ".joblib"

        # Save DataFrame and scaler using joblib
        data_to_save = {
            'df': df,
            'scaler': scaler
        }
        joblib.dump(data_to_save, filepath)

        # Save TensorFlow model using model.save()
        model.save(filepath + "_model")

        return True

    return False


def predict_neural(model, column1, column2):
    global scaler
    input_data = np.array([[column1, column2]])
    input_data_scaled = scaler.transform(input_data)

    return model.predict(input_data_scaled)


# ======================================================================================================================

def train_decision_tree(df, max_dept, criterion, splitter, random_state):
    global column1_name
    global column2_name
    global scaler

    df.dropna(inplace=True)
    input_columns = df.iloc[:, :2]
    output_column = df.iloc[:, 2]
    scaler = StandardScaler()
    input_columns_scaled = scaler.fit_transform(input_columns)

    model = DecisionTreeClassifier(max_depth=max_dept, criterion=criterion, splitter=splitter,
                                   random_state=random_state)
    model.fit(input_columns_scaled, output_column)

    return model


def retrain_decision_tree(model, column1, column2, output):
    global scaler

    input_data = np.array([[column1, column2]])
    input_data_scaled = scaler.transform(input_data)
    output_data = np.array([output])
    model.fit(input_data_scaled, output_data)

    return


def load_decision_tree():
    filetypes = [("Joblib files", "*.joblib")]
    filepath = filedialog.askopenfilename(filetypes=filetypes)

    if filepath:
        global scaler
        loaded_objects = joblib.load(filepath)
        scaler = loaded_objects['scaler']
        return loaded_objects

    return None


def save_decision_tree(model, df):
    global scaler

    filetypes = [("Joblib files", "*.joblib")]
    filepath = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=filetypes)

    if filepath:
        if not filepath.endswith(".joblib"):
            filepath += ".joblib"

        objects_to_save = {
            'model': model,
            'df': df,
            'scaler': scaler
        }

        joblib.dump(objects_to_save, filepath)
        return True

    return False


def predict_decision_tree(model, column1, column2):
    global column1_name
    global column2_name
    global scaler

    input_data = np.array([[column1, column2]])
    input_data_scaled = scaler.transform(input_data)

    model.feature_names_ = [column1_name, column2_name]

    return model.predict(input_data_scaled)
