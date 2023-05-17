import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def KNNprediction(df):
    # Data is argument of function
    # Split the dataset into input features and target variable
    X = df[['input1', 'input2']]
    y = df['output']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the classifier using the training data
    knn.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = knn.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    input1_value = None
    input2_value = None
    # Example usage: Predicting a single instance
    new_instance = [[input1_value, input2_value]]
    prediction = knn.predict(new_instance)
    print(f"Prediction: {prediction}")
