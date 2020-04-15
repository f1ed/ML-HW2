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
X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_Train'
X_test_fpath = './data/X_test'
output_fpath = './logistic_output/output_logistic.csv'
fpath = './logistic_output/logistic'

X_train = np.genfromtxt(X_train_fpath, delimiter=',')
Y_train = np.genfromtxt(Y_train_fpath, delimiter=',')
X_test = np.genfromtxt(X_test_fpath, delimiter=',')

X_train = X_train[1:, 1:]
Y_train = Y_train[1:, 1:]
X_test = X_test[1:, 1:]


def _normalization(X, train=True, X_mean=None, X_std=None):
    # This function normalize columns of X.
    # Output:
    #       X: normalized data
    #       X_mean, X_std
    if train:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
    for j in range(X.shape[1]):
        X[:, j] = (X[:, j] - X_mean[j]) / (X_std[j] + 1e-8)  # avoid X_std==0
    return X, X_mean, X_std


def _train_dev_split(X, Y, dev_ratio=0.25):
    # This function splits data into training set and development set.
    train_size = int(X.shape[0] * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


# Normalize train_data and test_data
X_train, X_mean, X_std = _normalization(X_train, train=True)
X_test, _, _ = _normalization(X_test, train=False, X_mean=X_mean, X_std=X_std)

# Split data into train data and development data
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]

with open(fpath, 'w') as f:
    f.write('In logistic model:\n')
    f.write('Size of Training set: {}\n'.format(train_size))
    f.write('Size of development set: {}\n'.format(dev_size))
    f.write('Size of test set: {}\n'.format(test_size))
    f.write('Dimension of data: {}\n'.format(data_dim))


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
    #   w: weight vector, shape = [data_dimension, 1]
    #   b: bias, scalar
    # Output:
    #       predict probability of each row of X being positively labeled, shape = [batch_size, 1]
    return _sigmod(np.dot(X, w) + b)


def _predict(X, w, b):
    # This fucntion returns a truth value prediction for each row of X by logistic regression
    return np.around(_f(X, w, b)).astype(np.int)


def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    # Y_pred: 0 or 1
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc


def _cross_entropy_loss(y_pred, Y_label):
    # This function calculates the cross entropy of Y_pred and Y_label
    #
    # Argument:
    #          y_pred: predictions, float vector
    #          Y_label: truth labels, bool vector
    cross_entropy = - np.dot(Y_label.T, np.log(y_pred)) - np.dot((1 - Y_label).T, np.log(1 - y_pred))
    return cross_entropy[0][0]


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
w_grad_sum = np.full((data_dim, 1), 1e-8)  # avoid divided by zeros
b_grad_sum = np.float(1e-8)

# Some parameters for training
epoch = 20
batch_size = 2**3
learning_rate = 0.2

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
        w = w - learning_rate * w_grad / np.sqrt(w_grad_sum)
        b = b - learning_rate * b_grad / np.sqrt(b_grad_sum)

    # compute loss and accuracy of training set and development set at every epoch
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.around(y_train_pred)
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train)/train_size)
    train_acc.append(_accuracy(Y_train_pred, Y_train))

    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.around(y_dev_pred)
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev)/dev_size)
    dev_acc.append(_accuracy(y_dev_pred, Y_dev))

with open(fpath, 'a') as f:
    f.write('Training loss: {}\n'.format(train_loss[-1]))
    f.write('Training accuracy: {}\n'.format(train_acc[-1]))
    f.write('Development loss: {}\n'.format(dev_loss[-1]))
    f.write('Development accuracy: {}\n'.format(dev_acc[-1]))

###################
# Plotting Loss and accuracy curve
# Loss curve
plt.plot(train_loss, label='train')
plt.plot(dev_loss, label='dev')
plt.title('Loss')
plt.legend()
plt.savefig('./logistic_output/loss.png')
plt.show()

plt.plot(train_acc, label='train')
plt.plot(dev_acc, label='dev')
plt.title('Accuracy')
plt.legend()
plt.savefig('./logistic_output/acc.png')
plt.show()

#################################
# Predict
predictions = _predict(X_test, w, b)
with open(output_fpath, 'w') as f:
    f.write('id, label\n')
    for id, label in enumerate(predictions):
        f.write('{}, {}\n'.format(id, label[0]))

###############################
# Output the weights and bias
ind = (np.argsort(np.abs(w), axis=0)[::-1]).reshape(1, -1)

with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
content = content[1:]

with open(fpath, 'a') as f:
    for i in ind[0, 0: 10]:
       f.write('{}: {}\n'.format(content[i], w[i]))