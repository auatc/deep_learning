'''
first step:packages
'''
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

'''
second step:preprocess the  dataset
'''


def preprocess():
    # get data
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    # get data_values
    m_train, m_test = train_set_x_orig.shape[0], test_set_x_orig.shape[0]  # number of train/test examples
    num_px = train_set_x_orig[0].shape[0]  # =height=width of a training image

    # reshape the training/test data
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],
                                                   -1).T  # image(num_px,num_px,3)---->single vectors(num_px*num_px*3,1)
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    # standardize dataset
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    return train_set_x, train_set_y, test_set_x, test_set_y


'''
third step:build function for model
1.sigmoid
    compute the sigmoid of z
2.initialize_with_zeros
    create a vector of zeros of shape(dim,1) for w and initializes b to 0.0
3.propagate
    implement the cost function and its gradient for the propagation explained above
4.optimize
    optimizes w and b by running a gradient descent algorithm
5.predict
    predict whether the label is 0 or 1 using learned logistic regression parameters(w,b) 
'''


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def initialize_with_zero(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b


def propagate(w, b, X, Y):
    # number of sample
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / -m
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs, dw, db = [], [], []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw, db = grads['dw'], grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"cost after iteration {i}: {cost}")
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


'''
fourth step:merge all function into a model
'''


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=True):
    w, b = initialize_with_zero(X_train.shape[0])

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w, b = params['w'], params['b']
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    if print_cost:
        print("train accuracy:{} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy:{} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


if __name__ == '__main__':
    train_set_x, train_set_y, test_set_x, test_set_y = preprocess()
    logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000,
                                      learning_rate=0.005, print_cost=True)
    # print(logistic_regression_model)
