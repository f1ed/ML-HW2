
import numpy as np

np.random.seed(0)
##############################################
# Prepare data
X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'
output_fpath = './generative_output/output_{}.csv'
fpath = './generative_output/generative'

X_train = np.genfromtxt(X_train_fpath, delimiter=',')
Y_train = np.genfromtxt(Y_train_fpath, delimiter=',')
X_test = np.genfromtxt(X_test_fpath, delimiter=',')

X_train = X_train[1:, 1:]
Y_train = Y_train[1:, 1:]
X_test = X_test[1:, 1:]

def _normalization(X, train=True, X_mean=None, X_std=None):
    # This function normalize columns of X
    # Output:
    #       X: normalized data
    #       X_mean, X_std
    if train:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
    for j in range(X.shape[1]):
       X[:, j] = (X[:, j] - X_mean[j]) / (X_std[j] + 1e-8)  # avoid X_std==0
    return X, X_mean, X_std

# Normalize train_data and test_data
X_train, X_mean, X_std = _normalization(X_train, train=True)
X_test, _, _ = _normalization(X_test, train=False, X_mean=X_mean, X_std=X_std)

train_size = X_train.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]

with open(fpath, 'w') as f:
    f.write('In generative model:\n')
    f.write('Size of training data: {}\n'.format(train_size))
    f.write('Size of test set: {}\n'.format(test_size))
    f.write('Dimension of data: {}\n\n'.format(data_dim))

########################
# Useful functions
def _sigmod(z):
    # Sigmod function can be used to compute probability
    # To avoid overflow
    return np.clip(1/(1.0 + np.exp(-z)), 1e-8, 1-(1e-8))

def _f(X, w, b):
    # This function is the linear part of sigmod function
    # Arguments:
    #   X: input data, shape = [size, data_dimension]
    #   w: weight vector, shape = [data_dimension, 1]
    #   b: bias, scalar
    # Output:
    #   predict probabilities
    return _sigmod(np.dot(X, w) + b)

def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X belonging to class1(label=0)
    return np.around(_f(X, w, b)).astype(np.int)

def _accuracy(Y_pred, Y_label):
    # This function computes prediction accuracy
    # Y_pred: 0 or 1
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

#######################
# Generative Model: closed-form solution, can be computed directly

# compute in-class mean
X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])

mean_0 = np.mean(X_train_0, axis=0)
mean_1 = np.mean(X_train_1, axis=0)

# compute in-class covariance
cov_0 = np.zeros(shape=(data_dim, data_dim))
cov_1 = np.zeros(shape=(data_dim, data_dim))

for x in X_train_0:
    # (D,1)@(1,D) np.matmul(np.transpose([x]), x)
    cov_0 += np.matmul(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
for x in X_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

# shared covariance
cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train.shape[0])

# compute weights and bias
# Since covariance matrix may be nearly singular, np.linalg.in() may give a large numerical error.
# Via SVD decomposition, one can get matrix inverse efficiently and  accurately.
# cov = u@s@vh
# cov_inv = dot(vh.T * 1 / s, u.T)
u, s, vh = np.linalg.svd(cov, full_matrices=False)
s_inv = s  # s_inv avoid <1e-8
for i in range(s.shape[0]):
    if s[i] < (1e-8):
        break
    s_inv[i] = 1./s[i]
cov_inv = np.matmul(vh.T * s_inv, u.T)

w = np.matmul(cov_inv, np.transpose([mean_0 - mean_1]))
b = (-0.5) * np.dot(mean_0, np.matmul(cov_inv, mean_0.T)) + (0.5) * np.dot(mean_1, np.matmul(cov_inv, mean_1.T)) + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])

# compute accuracy on training set
Y_train_pred = 1 - _predict(X_train, w, b)
with open(fpath, 'a') as f:
    f.write('\nTraining accuracy: {}\n'.format(_accuracy(Y_train_pred, Y_train)))

# Predict
predictions = 1 - _predict(X_test, w, b)
with open(output_fpath.format('generative'), 'w') as f:
    f.write('id, label\n')
    for i, label in enumerate(predictions):
        f.write('{}, {}\n'.format(i, label))

# Output the most significant weight
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
content = content[1:]

ind = np.argsort(np.abs(np.concatenate(w)))[::-1]
with open(fpath, 'a')as f:
    for i in ind[0:10]:
        f.write('{}: {}\n'.format(content[i], w[i]))
