import matplotlib.pyplot as plt
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import run_this


def plot_learning_curve(logistic_regression_model):
    costs=np.squeeze(logistic_regression_model['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations(per hundreds')
    plt.title("Learning rate ="+str(logistic_regression_model["learning_rate"]))
    plt.show()

def choice_learning_rate():
    learning_rates=[0.01,0.001,0.0001]
    models={}
    train_set_x, train_set_y, test_set_x, test_set_y = run_this.preprocess()
    for lr in learning_rates:
        print(f"Training a model with learning rate {str(lr)}")
        models[str(lr)]=run_this.model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations=5000,learning_rate=lr,print_cost=True)
        print('\n'+"-----------------------------------------------"+'\n')

    for lr in learning_rates:
        plt.plot(np.squeeze(models[str(lr)]['costs']),label=str(models[str(lr)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations(hundreds)')

    legend=plt.legend(loc='upper center',shadow=True)
    frame=legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


if __name__ == '__main__':
    # 1.learn to plot learning curve
    train_set_x, train_set_y, test_set_x, test_set_y = run_this.preprocess()
    logistic_regression_model = run_this.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000,
                                           learning_rate=0.005, print_cost=True)

    plot_learning_curve(logistic_regression_model)


    # 2.learn to choice learning rate
    choice_learning_rate()


