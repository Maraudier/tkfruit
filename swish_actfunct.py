# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:41:34 2020

@author: hanan
"""

from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

# Implementing the swish activation function (source: a new activation function
# described in a paper from researchers at Google)

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish': Activation(swish)})