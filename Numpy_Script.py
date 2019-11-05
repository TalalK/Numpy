#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:53:34 2019

@author: tkhodr

Class notes for Numerical Computing with python
(Basics)
"""
import sys

sys.version 

import numpy as np
# Checking Library verison
np.__version__

# Creating Arrays with python list 
# Passing a list into a numpy fuction

a = np.array([1,3,5,7,9])

type(a)

# Default is int64

a.dtype

a = np.array([1,3,5,7,9], dtype=np.float32)

a.dtype

# If theres a floating point, Data type is float as default
b = np.array([1,3,5,6,9.0])

#========================================#

# Creating a 2D array (Matrix)
# To create multiple layers, use same method to nest them
a = np.array( [ [1,2,3],[4,5,6] ] )

a

a = np.array( [ [1,2,3],[4,5,6] ], dtype=np.float32 )

a

# Identity Matrix

np.identity(3)
np.identity(6)


# Creating a Zero Matrix 
np.zeros((5,))

a = np.zeros((5,4))

a.dtype

a = np.zeros((5,4), dtype = np.int16)

np.empty((3,4))


a = np.array(([1,2,3,],[3,4,5]))

a

a.shape

a.ndim

b = a[:,0]

b

c = a[0,:]

a.base

#Reshaping and Resizing a matrix

a = np.array(([1,2,3,],[3,4,5]))

a.reshape(6,1)

a.reshape(1,6)

a.reshape(3,2)

#-------------#

a.max()

a.sum(axis=1)


a.prod(axis=1)
a.prod()


%matplotlib inline
import numpy as np
from numpy.polynomial import Polynomial as P
import matplotlib.pyplot as plt

p = P([1,2,3])

p

x = np.linspace(-5,5)

x

plt.plot(x,p(x))
 
p.integ(1, k=[0])

# Differentiate the polynomial
p.deriv(1)

# Fitting a Polynomial.

x = np.linspace(0, 2*np.pi,100)
y = np.sin(x) + np.random.normal(scale=.1,size=x.shape)
plt.plot(x,y,'o')

p1 = P.fit(x,y,1)
p2 = P.fit(x,y,2)
p3 = P.fit(x,y,3)
plt.plot(x, p1(x),'--')
plt.plot(x, p2(x),'b*')
plt.plot(x, p3(x),'r-')
plt.plot(x, y , 'ko')

# Import Random numbers

np.random.rand(10)

np.random.rand(3,2)

np.random.randint(0 , 10, 10)

np.random.randint(0 , 10, 20)

# Creating a random Matrix 

np.random.randint(0,10,(4,3))

np.random.randint(0,10,12).reshape(4,3)

# Random Picking in a set of elements.

x = np.array([1,2,3,4,5,6,7,8,9,10])

np.random.choice(x,20)



# Interpolation Concept and code
# Given (x1,y1),(x2,y2),xp => What is Yp = ?
# Line Equation Between (x1,y1) and (x2,y2)
# is y = ((y2-y1)/(x2-x2))(x-x1) + y1

# yp = ((y2-y1)/(x2-x2))(xp-x1) + y1
# HW
l = np.array([1,2,3,4,5])
s = np.array([6.0,6.5,6.2,5.5,6.0])

x,y

# Step 2

id_l = np.argsort(l)
id_l

l[id_l]
s[id_l]
 
# Step 3

l = l[id_l]
s = s[id_l]

lp = 1.25

i = np.searchsorted(l,lp)

i

l1,l2 = l[i-1],l[i]
s1,s2 = s[i-1],s[i]

# Step 3 apply the interpolation

sp = (lp-l1)*(s2-s1)/(l2-l1)+ s1

sp

plt.plot(l,s,'o--')
plt.plot([lp],[sp],'r*')


%matplotlic inline 
import numpy as np

x = np.array([1.0,3.0,2.0,4.0], dtype= np.float64)
y = np.array([0,4,2,6], dtype = np.float64)

x, y

# What is the corresponding value when x = 1.25


## Step 1 Sort the data 

idx = np.argsort(x)

idx

x

x[idx]


y[idx]

## Step 2 find twp adjacent x's values surrounding xp

x = x[idx]
y = y[idx]
xp = 1.25

x

i = np.searchsorted(x,xp)
i

x1,x2 = x[i-1],x[i]
y1,y2 = y[i-1],y[i]

# Step 3 Apply the interpolation formula

yp = (xp-x1)*(y2-y1)/(x2-x1)+ y1

yp

plt.plot(x,y,'o--')
plt.plot([xp],[yp],'r*')


# Higher order interpolation

from scipy import interpolate


x = np.array([0,1,2,3,4])
y = np.array([1,0.7,0.135,0.01,0.0003])
plt.plot(x,y,'o--')
x,y

f1 = interpolate.interp1d(x,y, kind='linear')
f2 = interpolate.interp1d(x,y, kind='quadratic')
f3 = interpolate.interp1d(x,y, kind='cubic')

xp = np.linspace(0,4,100)
plt.plot(x,y,'o')
plt.plot(xp,f1(xp), 'r--', label = 'linear')
plt.plot(xp,f2(xp), 'b--', label = 'quad')
plt.plot(xp,f3(xp), 'k--', label = 'cubic')
plt.legend()

# Data outside domain 

f2 = interpolate.interp1d(x,y, kind = 'quadratic')
xp = np.linspace(-1,5,100)

x

f2 = interpolate.interp1d(x,y, kind = 'quadratic',
                          fill_value=(y[0],y[-1]),bounds_error=False)
xp = np.linspace(-1,5,100)
plt.plot(x,y,'o')
plt.plot(xp,f2(xp),'r-')

# HW Question 


# linear Regression (Least Square)

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5], dtype = np.float64)
y = np.array([1.3,1.9,3.1,4.2,4.5], dtype = np.float64)

plt.plot(x,y,'o')

# Step 1 Create matrix A

A = np.zeros((x.shape[0],2))
A

A[:,0] = x #First Column 
A[:,1] = 1 # Second Column
A


A2 = np.matmul(A.T, A) # A TRANSPOSE
A2

A2b = np.zeros((2,2))
A2b[0,0] = np.sum(x*x)
A2b[0,1] = A2b[1,0] = np.sum(x)
A2b[1,1] = x.shape[0]
A2b

b = np.matmul(A.T, y)
b

t = np.linalg.solve(A2, b)
t


def yfunc(x,t):
    return t[0]*x+t[1]
xp = np.linspace(0.5,5.5,3)
plt.plot(x,y,'ko')
plt.plot(xp,yfunc(xp,t), 'r--')
