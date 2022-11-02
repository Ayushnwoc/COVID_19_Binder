import numpy as np
import matplotlib.pyplot as plt
import sys

"""
Python script to plot the loss curve
"""
def plot_losses(mse, epochs):
    plt.plot(mse)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.axis([0, epochs, 0, np.max(mse)+0.1])
    plt.title('Loss Curve')
    plt.savefig('training_loss_'+str(epochs)+'.png')
    plt.clf()
