#!/usr/bin/env python
# coding: utf-8

# In[35]:

import sys
from scipy.stats import mode
from sklearn.datasets import load_iris
import numpy as np
from scipy.stats import multivariate_normal 
iris = load_iris()
X=np.loadtxt(sys.argv[1], delimiter=",", usecols=range(4))
k = int(sys.argv[2])

# In[42]:

#The function
class EM:
    def __init__(self, k, max_iter=5):
        self.k = k
        self.max_iter = int(max_iter)
        
#Initialising the vallues
    def initialize(self, X):
        # returns the (r,c) value of the numpy array of X
        self.shape = X.shape 
        # n has the number of rows while m has the number of columns of dataset X
        self.n, self.m = self.shape 
        # initial weights given to each cluster are stored in phi or P(Ci=j)
        self.phi = np.full(shape=self.k, fill_value=1/self.k) 
        # initial weights given to each data point wrt to each cluster or P(Xi/Ci=j)
        self.weights = np.full(shape=self.shape, fill_value=1/self.k)
        # dataset is divided randomly into k parts of unequal sizes
        rand_row = np.random.randint(low=0, high=self.n, size=self.k)
        # initial value of mean of k Gaussians
        self.mean = [  X[row_index,:] for row_index in rand_row ] 
        # initial value of covariance matrix of k Gaussians
        self.covar = [ np.cov(X.T) for _ in range(self.k) ] 
       # theta =(mu1,sigma1,mu2,simga2......muk,sigmak)
        
#Expectation Step: update weights and phi holding mu and sigma constant
    def expectation_step(self, X):
        # updated weights or P(Xi|Ci=j)
        self.weights = self.predict_proba(X)
        # mean of sum of probability of all data points wrt to one cluster is new updated probability of cluster k
        self.phi = self.weights.mean(axis=0)

#Maximisation Step: update meu and sigma holding phi and weights constant
    def maximisation_step(self, X):
        for i in range(self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            #Mean
            self.mean[i] = (X * weight).sum(axis=0) / total_weight
            #Covariance
            self.covar[i] = np.cov(X.T,aweights=(weight/total_weight).flatten(), bias=True)

# responsible for clustering the data points correctly
    def fit(self, X):
        # initialise parameters like weights, phi, meu, sigma of all Gaussians in dataset X
        self.initialize(X)
        for iteration in range(self.max_iter):
            permutation = np.array([mode(iris.target[em.predict(X) == i]).mode.item() for i in range(em.k)])
            permuted_prediction = permutation[em.predict(X)]
            # iterate to update the value of P(Xi|Ci=j) and (phi)k
            self.expectation_step(X)
            # iterate to update the value of meu and sigma as the clusters shift
            self.maximisation_step(X)
        print('\n')
        print('Final Mean:- ',self.mean)
        print('\n')
        print('Final Covariance:- ',self.covar)
        print('\n')
        print('No of iterations:- ',iteration)
        print('\n')
        print('Cluster membership:-',self.weights)
        print('\n')
        print('Size:-',self.weights.shape)
        print('\nThe accuracy at iteration',iteration+1,end="")
        print(': ',np.mean(iris.target == permuted_prediction))
    
    # predicts probability of each data point wrt each cluster
    def predict_proba(self, X):
        # Creates a n*k matrix denoting probability of each point wrt each cluster 
        likelihood = np.zeros( (self.n, self.k) )
        for i in range(self.k):
            distribution = multivariate_normal(mean=self.mean[i],cov=self.covar[i])
            # pdf : probability denisty function
            likelihood[:,i] = distribution.pdf(X) 

        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights
    # predict function  
    def predict(self, X):
        weights = self.predict_proba(X)
        # datapoint belongs to cluster with maximum probability
        # returns this value
        return np.argmax(weights, axis=1)
    
def jitter(x):
    return x + np.random.uniform(low=-0.05, high=0.05, size=x.shape)


# In[43]:


np.random.seed(42)
em = EM(k, max_iter=10)
em.fit(X)
# In[ ]:




