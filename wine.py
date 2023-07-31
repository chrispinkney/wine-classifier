import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def svm(X_train, y_train, X_test, y_test):
    # Create an instance of the LinearSVC classifier
    clf = LinearSVC()

    # Start timing the training process
    start_time = time.time()

    # Train the LinearSVC classifier
    clf.fit(X_train, y_train)

    # Perform predictions on the test set
    y_pred = clf.predict(X_test)

    # End timing the training process
    elapsed_time = time.time() - start_time

    # Evaluate the accuracy of the classifier on the test set
    accuracy = clf.score(X_test, y_test)

    # Display the confusion matrix to evaluate classification performance
    y_true = y_test  # True class labels for the test set
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("SVM Results:")
    print(f"Confusion Matrix: {conf_matrix}")
    print(f"Elapsed Time: {elapsed_time * 1000:.2f} ms")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Total time to run program: {elapsed_time:.2f} seconds")


def fishers(X_train, y_train, X_test, y_test):
    # Create an instance of the LinearDiscriminantAnalysis classifier
    clf_fld = LDA()

    # Start timing the training process
    start_time = time.time()

    # Train the LinearDiscriminantAnalysis classifier
    clf_fld.fit(X_train, y_train)

    # Perform predictions on the test set
    y_pred = clf_fld.predict(X_test)

    # End timing the training process
    elapsed_time = time.time() - start_time

    # Evaluate the accuracy of the classifier on the test set
    accuracy = clf_fld.score(X_test, y_test)

    # Display the confusion matrix to evaluate classification performance
    y_true = y_test  # True class labels for the test set
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("\nFisher's Linear Discriminant Results:")
    print(f"Confusion Matrix: {conf_matrix}")
    print(f"Elapsed Time: {elapsed_time * 1000:.2f} ms")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Total time to run program: {elapsed_time:.2f} seconds")


# Load red and white wine data
rawTrainingDataRed = np.genfromtxt('winequality-red.csv', delimiter=';', skip_header=1, max_rows=1000)
rawTrainingDataWhite = np.genfromtxt('winequality-white.csv', delimiter=';', skip_header=1, max_rows=1000)

# Create labels for training data: 0 for red wine, 1 for white wine
red_labels = np.zeros((rawTrainingDataRed.shape[0], 1))
white_labels = np.ones((rawTrainingDataWhite.shape[0], 1))

# Combine red and white wine data into one array
combinedData = np.concatenate((rawTrainingDataRed, rawTrainingDataWhite), axis=0)

# Prepare attributes (X) and classes (y) from the combined data
X = combinedData[:, :11]  # Extract the first 11 columns as attributes
y = np.vstack((red_labels, white_labels)).flatten() # Combine the class labels for red and white wine data into a single one-dimensional array 'y'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

svm(X_train, y_train, X_test, y_test)
fishers(X_train, y_train, X_test, y_test)
