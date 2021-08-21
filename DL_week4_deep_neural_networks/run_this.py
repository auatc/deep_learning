import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from use_function import *

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)


class run_this():
    # load and process the data
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    # reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(m_train,
                                           -1).T  # image(num_px,num_px,3)---->single vectors(num_px*num_px*3,1)
    test_x_flatten = test_x_orig.reshape(m_test, -1).T

    # standardize data to have feature values between 0 and 1
    train_x = train_x_flatten / 255
    test_x = test_x_flatten / 255

    def two_layer_model(self, X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
        '''
         def initialize_parameters(n_x, n_h, n_y):
             ...
             return parameters
         def linear_activation_forward(A_prev, W, b, activation):
             ...
             return A, cache
         def compute_cost(AL, Y):
             ...
             return cost
         def linear_activation_backward(dA, cache, activation):
             ...
             return dA_prev, dW, db
         def update_parameters(parameters, grads, learning_rate):
             ...
             return parameters
        '''

        np.random.seed(1)
        grads = {}
        costs = []
        m = X.shape[1]
        (n_x, n_h, n_y) = layers_dims

        # step1:initialize parameters
        parameters = initialize_parameters(n_x, n_h, n_y)

        W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]

        for i in range(0, num_iterations):
            # step2:forward
            A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
            A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

            # step3:compute cost
            cost = compute_cost(A2, Y)

            # step4:backward
            dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
            dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
            dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
            grads['dW1'], grads['db1'], grads['dW2'], grads['db2'] = dW1, db1, dW2, db2

            # step4:update parameters
            parameters = update_parameters(parameters, grads, learning_rate)

            W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]

            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)

        return parameters, costs

    def L_layer_model(self, X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=True):
        '''
        def initialize_parameters_deep(layers_dims):
            ...
            return parameters
        def L_model_forward(X, parameters):
            ...
            return AL, caches
        def compute_cost(AL, Y):
            ...
            return cost
        def L_model_backward(AL, Y, caches):
            ...
            return grads
        def update_parameters(parameters, grads, learning_rate):
            ...
            return parameters
        '''
        np.random.seed(1)
        costs = []

        # step1:initialize parameters
        parameters = initialize_parameters_deep(layers_dims)

        for i in range(0, num_iterations):
            # step2:forward
            AL, caches = L_model_forward(X, parameters)

            # step3:compute cost
            cost = compute_cost(AL, Y)

            # step4:backward
            grads = L_model_backward(AL, Y, caches)

            # step5:update parameters
            parameters = update_parameters(parameters, grads, learning_rate)

            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)

        return parameters, costs

    def plot_costs(self, costs, learning_rate=0.0075):
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    def train_2_layer_model(self):
        n_x = 12288  # num_px * num_px * 3
        n_h = 7
        n_y = 1
        learning_rate = 0.0075
        parameters, costs = self.two_layer_model(self.train_x, self.train_y, layers_dims=(n_x, n_h, n_y),
                                                 num_iterations=2500,
                                                 print_cost=True)
        self.plot_costs(costs, learning_rate)

        print("train:")
        predictions_train = predict(self.train_x, self.train_y, parameters)
        print("test:")
        predictions_test = predict(self.test_x, self.test_y, parameters)

    def train_L_layer_model(self):
        layers_dims = [12288, 20, 7, 5, 1]
        parameters, costs = self.L_layer_model(self.train_x, self.train_y, layers_dims, num_iterations=2500,
                                               print_cost=True)

        print("train:")
        pred_train = predict(self.train_x, self.train_y, parameters)
        print("test:")
        pred_test = predict(self.test_x, self.test_y, parameters)

        print_mislabeled_images(self.classes, self.test_x, self.test_y, pred_test)

        return parameters

    def run_predict(self, parameters, train_x):
        my_image = train_x  # change this to the name of your image file
        my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)
        fname = "datasets/" + my_image
        image = np.array(Image.open(fname).resize((self.num_px, self.num_px)))
        plt.imshow(image)
        image = image / 255.
        image = image.reshape((1, self.num_px * self.num_px * 3)).T

        my_predicted_image = predict(image, my_label_y, parameters)

        print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + self.classes[
            int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")


if __name__ == '__main__':
    test = run_this()
    param = test.train_L_layer_model()
    test.run_predict(param, 'cat_in_iran.jpg')
