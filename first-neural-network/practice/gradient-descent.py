import numpy as np


def sigmoid(x):
    """ Sigmoid function for activations. """
    return 1/(1+np.exp(-x))


def sigmoid_prime(x):
    """ Derivative of the sigmoid function. """
    return sigmoid(x) * (1 - sigmoid(x))


X = np.array([0.1, 0.3])
Y = 0.2
WEIGHTS = np.array([-0.8, 0.5])
LR = 0.5

# linear combination performed by the node (h in f(h) and f'(h))
h = X[0]*WEIGHTS[0] + X[1]*WEIGHTS[1]

# TODO: What is h?

# output: y-hat
nn_output = sigmoid(h)


# output error: (y - y-hat)
error = y - nn_output

# output gradient (f'(h))
output_grad = sigmoid_prime(h)

# error term (lowercase delta)
error_term = error * output_grad

# gradient descent step
del_w = [ LR * error_term * x[0],
          LR * error_term * x[1]]

# or del_w = LR * error_term * x
