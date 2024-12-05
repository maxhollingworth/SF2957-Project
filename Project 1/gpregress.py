from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
#choose a seed
seed = 42
np.random.seed(seed)

# import some data to play with
digits = datasets.load_digits()

X = digits.data
y = np.array(digits.target, dtype = int)
X,y = shuffle(X,y)
N,d = X.shape
# Initialize OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse_output=False)
# Transform labels to one-hot encoding
y_onehot = one_hot_encoder.fit_transform(y.reshape(-1, 1))

sizelist = [20, 50, 100, 500, 1000, 1500]
kernelList = [DotProduct(sigma_0=1.0), RBF([1.0]), Matern(length_scale=1.0, nu=1.5), RationalQuadratic(length_scale=1.0, alpha=0.5)]

for kernel in kernelList:
    test_error_rateList = []
    train_error_rateList = []
    print(kernel)

    for i in sizelist:
        # Split into training and testing sets
        Ntrain = i
        Ntest = 100

        Xtrain = X[:Ntrain, :]
        ytrain = y_onehot[:Ntrain, :] #Uses the vector here instead
        Xtest = X[Ntrain+1:Ntrain+Ntest+1, :]
        ytest = y[Ntrain+1:Ntrain+Ntest+1] #Just as before

        # Gaussian process regression
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10).fit(Xtrain, ytrain)

        # Predictions for training data
        yp_train = gpr.predict(Xtrain)
        train_predictions = np.argmax(yp_train, axis=1)  # Convert to class labels
        train_error_rate = np.mean(train_predictions != np.argmax(ytrain, axis=1))
        train_error_rateList.append(train_error_rate)

        # Predictions for test data
        yp_test = gpr.predict(Xtest)
        test_predictions = np.argmax(yp_test, axis=1)  # Convert to class labels
        test_error_rate = np.mean(test_predictions != ytest)
        test_error_rateList.append(test_error_rate)

        print(f"Test error rate: {test_error_rate}")

    # Plot results
    plt.figure(1)
    plt.plot(sizelist, test_error_rateList, label=f"{kernel}")
    plt.figure(2)
    plt.plot(sizelist, train_error_rateList, label=f"{kernel}")

plt.figure(1)
plt.legend()
plt.grid()
plt.xlabel("Training set size")
plt.ylabel("Error rate")
plt.title("Test Error for different kernels and training sizes")
plt.figure(2)
plt.legend()
plt.grid()
plt.xlabel("Training set size")
plt.ylabel("Error rate")
plt.title("Train Error for different kernels and training sizes")
plt.show()
