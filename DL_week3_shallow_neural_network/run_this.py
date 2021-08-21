'''
The general methodology to build a Neural Network is to:
1. Define the neural network structure ( # of input units, # of hidden units, etc).
2. Initialize the model's parameters
3. Loop:
    - Implement forward propagation
    - Compute loss
    - Implement backward propagation to get the gradients
    - Update parameters (gradient descent)
'''

import numpy as np
import copy
import matplotlib.pyplot as plt

from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from tqdm import tqdm


def laryer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache


def compute_cost(A2, Y):
    m = Y.shape[1]

    logprovs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = -np.sum(logprovs) / m

    cost = float(np.squeeze(cost))

    return cost


def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = copy.deepcopy(parameters['W1'])
    b1 = copy.deepcopy(parameters['b1'])
    W2 = copy.deepcopy(parameters['W2'])
    b2 = copy.deepcopy(parameters['b2'])
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)

    # step1:define the neural network structure
    n_x = laryer_sizes(X, Y)[0]
    n_y = laryer_sizes(X, Y)[2]

    # step2:initialize the model's parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    cost = 0
    # step3:loop
    for i in tqdm(range(0, num_iterations)):
        # 1.Implement forward propagation
        A2, cache = forward_propagation(X, parameters)
        # 2.Compute loss
        cost = compute_cost(A2, Y)
        # 3.Implement backward propagation to get the gradients
        grads = backward_propagation(parameters, cache, X, Y)
        # 4.Update parameters (gradient descent)
        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print("cost after iteration %i:%f" % (i, cost))

    return parameters


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    return predictions


def plot_predict(parameters, X, Y):
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()


def tuning_layer_size(X, Y):
    plt.figure(figsize=(16, 32))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 50]
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(9, 2, i + 1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X, Y, n_h, num_iterations=10000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
        print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))


if __name__ == '__main__':
    X, Y = load_planar_dataset()

    tuning_layer_size(X, Y)

    # parameters = nn_model(X, Y, 4, num_iterations=10000, print_cost=True)
    # predictions = predict(parameters, X)
    # print(f"Accuracy:{float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)}%")
