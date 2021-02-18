#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def classify_data(X_train, Y_train, X_test):
    """Develop and train your verY_trainown variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completelY_traincontained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed bY_trainyou in this function.

    Args:
        X_train (np.ndarray): An arraY_trainof floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An arraY_trainof size (250,) which are the categorical labels
            associated to the training data. The categories are labeled bY_train-1, 0, and 1.
        X_test (np.ndarray): An arraY_trainof floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this arraY_trainto make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #
    
    dev = qml.device('default.qubit', wires=3)
    
    padding = 0.3 * np.ones((len(X_train), 1))
    x_pad = np.c_[X_train, padding]
    
    normalization = np.sqrt(np.sum(x_pad ** 2, -1))
    x_norm = (x_pad.T / normalization).T
    
    
    num_train = len(Y_train)
    
    padding = 0.3 * np.ones((len(X_test), 1))
    x_pad_test = np.c_[X_test, padding]
    
    normalization = np.sqrt(np.sum(x_pad_test ** 2, -1))
    x_norm_test = (x_pad_test.T / normalization).T
    
    np.random.seed(11)
    num_qubits = 2
    num_layers = 6
    params = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0)
    opt = qml.optimize.NesterovMomentumOptimizer(0.01)
    
    def layer(w):
        qml.Rot(w[0, 0], w[0, 1], w[0, 2], wires=0)
        qml.Rot(w[1, 0], w[1, 1], w[1, 2], wires=1)
        qml.CNOT(wires=[0, 1])
    
    @qml.qnode(dev)
    def circuit(weights, angles):
        qml.QubitStateVector(angles, wires=[0,1])
        
        for w in weights:
            layer(w)
        
        return qml.expval(qml.PauliZ(0))
    
    def variational_circuit(var, angles):
        weights = var[0]
        bias = var[1]
        return circuit(weights, angles) + bias
    
    def cost(weights, x_data, y_data):
        predictions = variational_circuit(weights, x_data)
        return (y_data - predictions) ** 2
    
    def accuracy(labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            if abs(l - p) < 1e-5:
                loss = loss + 1
        loss = loss / len(labels)

        return loss
    
    for i in range(num_train):
        params = opt.step(lambda v: cost(v, x_norm[i], Y_train[i]), params)
        
    predictions = [int(np.round_(variational_circuit(params, x))) for x in x_norm_test]
    # QHACK #

    return array_to_concatenated_string(predictions)


def array_to_concatenated_string(array):
    """DO NOT MODIFY_trainTHIS FUNCTION.

    Turns an arraY_trainof integers into a concatenated string of integers
    separated bY_traincommas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY_trainTHIS FUNCTION.

    Turns a concatenated string of integers separated bY_traincommas into
    an arraY_trainof integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY_trainTHIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY_trainanything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")

