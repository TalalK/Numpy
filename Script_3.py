#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:50:34 2019

@author: tkhodr
"""

# Least Squares with Reqularizaion

# y = ax + b 
# ax1 + b = y1
# ax2 + b = y2
# Write in a matrix form

# Using VStack will stack both matricies together
# It will result in a wide and short rectandular array
# Then Transpose the matrix
# If array is called x and you call x.T 
# You will get the transposed version of x

import numpy as np
import matplotlib.pyplot as plt
import h5py

d = h5py.File('/Users/tkhodr/Desktop/ex2.h5')
X = d['x'][:]
Y = d['y'][:]

plt.plot(X,Y,'o')

# Least Square Fitting

np.vstack((X,np.ones(X.shape)))

A = np.vstack((X,np.ones(X.shape))).T
A


np.linalg.lstsq(A,Y,rcond=None)

theta = np.linalg.lstsq(A,Y,rcond=None)[0]
theta
  
xp = np.linspace(0,6,3)
plt.plot(xp,theta[0]*xp + theta[1],'--')
plt.plot(X,Y,'*')

# Regularized least Squares
len(X),len(Y)

#Labda ||a||^2


import scipy.sparse.linalg as sla

d = h5py.File('/Users/tkhodr/Desktop/ex2.h5')
X = d['x'][:]
Y = d['y'][:]
d.close

plt.plot(X,Y,'o')

N = X.shape
A = np.vstack((X**2,np.ones(N))).T
A

A.shape

Y.shape

theta = sla.lsmr(A,Y, damp=0)[0]
theta

xp = np.linspace(-2,2,100)
def yp(xp,theta):
    return theta[0]*xp**2 + theta[1]*xp + theta[2]
plt.plot(xp, yp(xp,theta), 'r-', label ='LS fit')
plt.plot(X,Y,'ba', label = 'data')
plt.legend()

# HomeWork
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.sparse.linalg as sla

d = h5py.File('/Users/tkhodr/Desktop/poly3-data.h5')
X = d['x'][:]
Y = d['y'][:]
d.close

plt.plot(X,Y,'o')

N = X.shape
N
A = np.vstack((X**2,X,np.ones(N))).T
A

A.shape

Y.shape

theta = sla.lsmr(A,Y, damp=0)[0]
theta

xp = np.linspace(-2,2,100)
def yp(xp,theta):
    return theta[0]*xp**2 + theta[1]*xp + theta[2]

plt.plot(xp, yp(xp,theta), 'r-', label ='LS fit')
plt.plot(X,Y,'bo', label = 'data')
plt.legend()

'''
The dimensions of the arrays passed to lsmr() used to solve the
 least squares problem
 
The (a,b,c,d) values you obtain
[-0.0055848 ,  1.31850646,  0.02472752])

The plot (an image file) showing the data 
points and the polynomial, similar to the following.


The Python code

'''

