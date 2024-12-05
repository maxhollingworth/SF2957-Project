"""
=====================================================
Gaussian process classification (GPC) 
=====================================================

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import RBF

#choose a seed
seed = 42
np.random.seed(seed)

# import some data to play with
digits = datasets.load_digits()

X = digits.data
y = np.array(digits.target, dtype = int)
X,y = shuffle(X,y)
N,d = X.shape

sizelist=[20,50,100,500,1000,1500]
kernelList=[DotProduct(sigma_0=1.0),RBF([1.0]),Matern(length_scale=1.0, nu=1.5),RationalQuadratic(length_scale=1.0, alpha=0.5)]
for kernel in kernelList:
    test_error_rateList = []
    train_error_rateList = []
    print(kernel)
    for i in sizelist:
        #N = 600
        Ntrain = i
        Ntest = 100


        Xtrain = X[0:Ntrain,:]
        ytrain = y[0:Ntrain]
        Xtest = X[Ntrain+1:Ntrain+Ntest+1,:]
        ytest = y[Ntrain+1:Ntrain+Ntest+1]


        #kernel = RBF([1.0]) # isotropic kernel
        #kernel = DotProduct(1.0)
        #kernel = 1.0 * RBF(length_scale=1.0)
        #kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
        #kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.5)
        #kernel = 1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0)
        #kernel = DotProduct(sigma_0=1.0)
        #kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        #GaussianProcessClassifier performs hyperparameter estimation, this means that the the value specified above may not be the final hyperparameters
        #If you dont want it to do hyperparameter optimization set optimizer = None like this GaussianProcessClassifier(kernel=kernel,optimizer = None).fit(Xtrain, ytrain)
        #You can check the final hyperparameters with gpc_rbf.kernel_.get_params()['kernels']
        gpc_rbf = GaussianProcessClassifier(kernel=kernel,max_iter_predict=1000, optimizer=None).fit(Xtrain, ytrain)
        yp_train = gpc_rbf.predict(Xtrain)
        train_error_rate = np.mean(np.not_equal(yp_train,ytrain))
        train_error_rateList.append(train_error_rate)
        yp_test = gpc_rbf.predict(Xtest)
        test_error_rate = np.mean(np.not_equal(yp_test,ytest))
        test_error_rateList.append(test_error_rate)
        #print('Training error rate')
        #print(train_error_rate)
        #print('Test error rate')
        print(test_error_rate)

    plt.figure(1)
    plt.plot(sizelist,test_error_rateList, label=f"{kernel}")
    plt.figure(2)
    plt.plot(sizelist,train_error_rateList, label=f"{kernel}")

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

"""
=====================================================
Gaussian process Regression (GPR) 
=====================================================

"""
from sklearn.preprocessing import OneHotEncoder


# import some data to play with
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
    plt.figure(3)
    plt.plot(sizelist, test_error_rateList, label=f"{kernel}")
    plt.figure(4)
    plt.plot(sizelist, train_error_rateList, label=f"{kernel}")

plt.figure(3)
plt.legend()
plt.grid()
plt.xlabel("Training set size")
plt.ylabel("Error rate")
plt.title("Test Error for different kernels and training sizes")
plt.figure(4)
plt.legend()
plt.grid()
plt.xlabel("Training set size")
plt.ylabel("Error rate")
plt.title("Train Error for different kernels and training sizes")
plt.show()
