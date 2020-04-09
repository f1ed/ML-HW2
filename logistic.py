#################
# Data:2020-04-05
# Author: Fred Lau
# ML-Lee: HW2 : Binary Classification
###########################################################
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

##########################################################
# prepare data
X_train = np.genfromtxt('./data/X_train', delimiter=',')
Y_train = np.genfromtxt('./data/Y_train', delimiter=',')
X_test = np.genfromtxt('./data/X_test', delimiter=',')
f = open('./output/logistic.csv', 'w')
sys.stdout = f

X_train = X_train[1:, 1:]
Y_train = Y_train[1:, 1:]
X_test = X_test[1:, 1:]


def _normaliztion(X, train=True, X_mean=None, X_std=None):
    # This function normalize columns of X.
    # Output:
    #       X: normalized data
    #       X_mean, X_std
    if train:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
    for j in range(X.shape[1]):
        if X_std[j] != 0:
            X[:, j] = (X[:, j] - X_mean[j]) / X_std[j]
    return X, X_mean, X_std


def _train_dev_split(X, Y, dev_ratio=0.25):
    # This function splits data into training set and development set.
    train_size = int(X.shape[0] * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


# Normalize train_data and test_data
X_train, X_mean, X_std = _normaliztion(X_train, train=True)
X_test, _, _ = _normaliztion(X_test, train=False, X_mean=X_mean, X_std=X_std)

# Split data into train data and development data
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]

print('In logistic model:\n')
print('Size of training set:', train_size)
print('Size of development set:', dev_size)
print('Size of test set:', test_size)
print('Dimension of data:', data_dim)


np.random.seed(0)

###############################################################
# useful function

def _shuffle(X, Y):
    # This function shuffles two two list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return X[randomize], Y[randomize]

def _sigmod(z):
    # Sigmod function can be used to calculate probability
    # To avoid overflow
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


def _f(X, w, b):
    # This is the logistic function, parameterized by w and b
    #
    # Arguments:
    #   X: input data, shape = [batch_size, data_dimension]
    #   w: weight vector, shape = [data_dimension]
    #   b: bias, scalar
    # Output:
    #       predict probability of each row of X being positively labeled, shape = [batch_size, 1]
    Z = np.dot(X, w) + b
    return _sigmod(np.dot(X, w) + b)


def _predict(X, w, b):
    # This fucntion returns a truth value prediction for each row of X by logistic regression
    return np.around(_f(X, w, b)).astype(np.int)


def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    # y_pred: 0 or 1
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc


def _cross_entropy_loss(y_pred, Y_label):
    # This function calculates the cross entropy of Y_pred and Y_label
    #
    # Argument:
    #          y_pred: predictions, float vector
    #          Y_label: truth labels, bool vector
    cross_entropy = - np.dot(Y_label.T, np.log(y_pred)) - np.dot((1 - Y_label).T, np.log(1 - y_pred))
    return cross_entropy[0]


def _gradient(X, Y_label, w, b):
    # This function calculates the gradient of cross entropy
    # X, Y_label, shape = [batch_size, ]
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = - np.dot(X.T, pred_error)
    b_grad = - np.sum(pred_error)
    return w_grad, float(b_grad)


#######################################
# training by logistic model

# Initial weights and bias
w = np.zeros((data_dim, 1))
b = np.float(0.)
w_grad_sum = np.full((data_dim, 1), 0.1)  # avoid divided by zeros
b_grad_sum = np.float(0.1)

# Some parameters for training
epoch = 100
batch_size = 2**3
learning_rate = 0.3

# Keep the loss and accuracy history at every epoch for plotting
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# Iterative training
for it in range(epoch):
    # Random shuffle at every epoch
    X_train, Y_train = _shuffle(X_train, Y_train)

    # Mini-batch training
    for id in range(int(np.floor(train_size / batch_size))):
        X = X_train[id*batch_size: (id+1)*batch_size]
        Y = Y_train[id*batch_size: (id+1)*batch_size]

        # calculate gradient
        w_grad, b_grad = _gradient(X, Y, w, b)

        # adagrad gradient update
        w_grad_sum = w_grad_sum + w_grad**2
        b_grad_sum = b_grad_sum + b_grad**2
        w_ada = np.sqrt(w_grad_sum)
        b_ada = np.sqrt(b_grad_sum)
        w = w - learning_rate * w_grad / w_ada
        b = b - learning_rate * b_grad / b_ada

    # compute loss and accuracy of training set and development set at every epoch
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.around(y_train_pred)
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train)/train_size)
    train_acc.append(_accuracy(Y_train_pred, Y_train))

    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.around(y_dev_pred)
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev_pred)/dev_size)
    dev_acc.append(_accuracy(y_dev_pred, Y_dev_pred))

print('Training loss: {}'.format(train_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development loss: {}'.format(dev_loss[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))

f.close()

###################
# Plotting Loss and accuracy curve
# Loss curve
plt.plot(train_loss, label='train')
plt.plot(dev_loss, label='dev')
plt.title('Loss')
plt.legend()
plt.savefig('./output/loss.png')
plt.show()

plt.plot(train_acc, label='train')
plt.plot(dev_acc, label='dev')
plt.title('Accuracy')
plt.legend()
plt.savefig('./output/acc.png')
plt.show()







