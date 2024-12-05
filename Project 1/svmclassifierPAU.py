import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct

# import some data to play with
digits = datasets.load_digits()

def svmsubgradient(Theta, x, y):
#  Returns a subgradient of the objective empirical hinge loss
#
# The inputs are Theta, of size n-by-K, where K is the number of classes,
# x of size n, and y an integer in {0, 1, ..., 9}.
    G = np.zeros(Theta.shape)
    ## IMPLEMENT THE SUBGRADIENT CALCULATION -- YOUR CODE HERE

    scores = Theta.T @ x  # Size: (K,)

    # Calculate the hinge loss terms for all classes except the true label
    loss_contributions = 1 + scores - scores[y]
    loss_contributions[y] = -np.inf  # Ignore the true class

    # Find the class with the maximum hinge loss contribution
    j_star = np.argmax(loss_contributions)

    # Check if hinge loss is active (S_j_star > 0)
    if loss_contributions[j_star] > 0:
        # Update the subgradient
        G[:, j_star] += x  # Add x to the column for class j_star
        G[:, y] -= x
    return(G)

def sgd(Xtrain, ytrain, maxiter, init_stepsize, l2_radius,alpha): #maxiter = 10, init_stepsize = 1.0, l2_radius = 10000):
#
# Performs maxiter iterations of projected stochastic gradient descent
# on the data contained in the matrix Xtrain, of size n-by-d, where n
# is the sample size and d is the dimension, and the label vector
# ytrain of integers in {0, 1, ..., 9}. Returns two d-by-10
# classification matrices Theta and mean_Theta, where the first is the final
# point of SGD and the second is the mean of all the iterates of SGD.
#
# Each iteration consists of choosing a random index from n and the
# associated data point in X, taking a subgradient step for the
# multiclass SVM objective, and projecting onto the Euclidean ball
# The stepsize is init_stepsize / sqrt(iteration).
    K = 10
    NN, dd = Xtrain.shape
    #print(NN)
    Theta = np.zeros(dd*K)
    Theta.shape = dd,K
    mean_Theta = np.zeros(dd*K)
    mean_Theta.shape = dd,K
    ## YOUR CODE HERE -- IMPLEMENT PROJECTED STOCHASTIC GRADIENT DESCENT

    for k in range(1, maxiter + 1):
        # Randomly pick an index
        idx = np.random.randint(0, NN)
        x = Xtrain[idx]  # Feature vector
        y = ytrain[idx]  # Label

        # Compute the subgradient
        G = svmsubgradient(Theta, x, y)
        #alpha = 0.5
        # Step size
        stepsize = init_stepsize*k**(-alpha)

        # Gradient descent step
        Theta = Theta - stepsize * G

        # Project Theta onto the Euclidean ball
        norm = np.linalg.norm(Theta, ord='fro')  # Frobenius norm
        if norm > l2_radius:
            Theta = (l2_radius / norm) * Theta  # Scale down to the ball

        # Update mean Theta
        mean_Theta = ((k - 1) * mean_Theta + Theta) / k  # Running average

    return Theta, mean_Theta

def Classify(Xdata, Theta):
#
# Takes in an N-by-d data matrix Adata, where d is the dimension and N
# is the sample size, and a classifier X, which is of size d-by-K,
# where K is the number of classes.
#
# Returns a vector of length N consisting of the predicted digits in
# the classes.
    scores = np.matmul(Xdata, Theta)
    inds = np.argmax(scores, axis = 1)
    return(inds)


#choose a seed
seed = 42
np.random.seed(seed)
# Load data into train set and test set
digits = datasets.load_digits()
X = digits.data
y = np.array(digits.target, dtype = int)
X,y = shuffle(X,y)
N,d = X.shape
Ntest = 100
training_sets = [20,50,100,500,1000,1500]
alpha_v = [0.1,0.3,0.5,0.7,0.9,1]
#errors = []
#alpha = 0.5
for alpha in alpha_v:
    errors = []
    for Ntrain in training_sets:
        Xtrain = X[0:Ntrain,:]
        ytrain = y[0:Ntrain]
        Xtest = X[Ntrain:Ntrain+Ntest,:]
        ytest = y[Ntrain:Ntrain+Ntest]
        l2_radius = 40.0
        M_raw = np.sqrt(np.mean(np.sum(np.square(Xtrain))))
        init_stepsize = l2_radius/M_raw
        maxiter = 40000
        Theta, mean_Theta = sgd(Xtrain, ytrain, maxiter, init_stepsize, l2_radius, alpha)#maxiter = 10, init_stepsize = 1.0, l2_radius = 10000
        print('Error rate')
        print(np.sum(np.not_equal(Classify(Xtest, mean_Theta),ytest)/Ntest))
        errors.append(np.sum(np.not_equal(Classify(Xtest, mean_Theta),ytest)/Ntest))
        plt.figure(1)
    plt.plot(training_sets, errors, label=f"{alpha}")

# plt.figure(figsize=(10, 6))
# plt.plot(training_sets, errors, marker='o')
# plt.title("Classifier Performance vs Training Set Size")
# plt.xlabel("Training Set Size")
# plt.ylabel("Error Rate")
# plt.grid()
# plt.show()
plt.figure(1)
plt.plot(training_sets, errors)
plt.title("Classifier Performance vs Learning rate")
plt.xlabel("Training set size")
plt.ylabel("Error Rate")
plt.legend()
plt.grid()
plt.show()

