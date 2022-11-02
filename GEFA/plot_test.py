import numpy as np
import matplotlib.pyplot as plt

"""
Python script to plot the prediction
"""
def plot_test(G, P):
    plt.plot(P.tolist(), G.tolist(), 'b+')
    x = np.linspace(4, 12, 2000)
    plt.plot(x, x, '-r', label='Ideal case curve')
    plt.xlabel('Predicted Values')
    plt.ylabel('Measured Values')
    plt.axis([4, 12, 4, 12])
    plt.savefig('test_davis.png')
    plt.clf()