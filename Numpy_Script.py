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