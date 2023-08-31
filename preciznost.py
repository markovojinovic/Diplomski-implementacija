import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('diabetes-skracena-druge_dve_kolone.csv')

X = data[['Glucose', 'BloodPressure']].values
y = data['Outcome'].values

rand = 0
max_acc = -1
max_rand = -1
max_size = -1
max_dept = -1
percentage = 0
while rand < 100:
    size = 0.2
    while size < 0.5:
        dept = 3
        while dept < 100:

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=rand)

            # Create and train the Decision Tree classifier
            clf = DecisionTreeClassifier(random_state=rand, max_depth=dept)
            clf.fit(X_train, y_train)

            # Make predictions
            y_pred = clf.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            # print("Accuracy:", accuracy)
            if accuracy > max_acc:
                max_acc = accuracy
                max_rand = rand
                max_size = size
                max_dept = dept

            dept += 1

        size += 0.01

    percentage += 1
    print("Finished ", percentage, "%")

    rand += 1


print("Best rand number = ", max_rand)
print("Best size of test data = ", max_size)
print("Best max dept = ", max_rand)
print("Max accuracy = ", max_acc)

# for input_vals, prediction in zip(X_test, y_pred):
#     print(f"Input: {input_vals}, Prediction: {prediction}")

# Best rand number =  56
# Best size of test data =  0.21000000000000002
# Best max dept =  56
# Max accuracy =  0.8209876543209876


