#!/usr/bin/env python
import sys, math, logging, numpy as np
import theano, theano.tensor as T
import cPickle as pickle
import bz2

import numpy, theano, sys, math, os
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict
import activations


class Layer(object):
    def __init__(self):
        self.params = []

    def set_weights(self, weights):
        for p, w in zip(self.params, weights):
            if p.eval().shape != w.shape:
                raise Exception("Layer shape %s not compatible with weight shape %s." % (p.eval().shape, w.shape))
            p.set_value(w.astype(T.config.floatX))

    def set_lr_coefs(self, lr_coefs):

        if len(self.lr_coefs) != len(lr_coefs):
            raise Exception("Layer lr_coefs %s not compatible with given lr_coefs %s." % (self.lr_coefs, lr_coefs))
        for ii in xrange(len(self.lr_coefs)):
            self.lr_coefs[ii] = lr_coefs[ii]

    def get_weights(self):
        weights = []
        for p in self.params:
            weights.append(p.get_value())
        return weights

    def get_config(self):
        return {"name": self.__class__.__name__}

    def get_params(self):
        return self.params

    def set_name(self, name):
        for i in range(len(self.params)):
            self.params[i].name = '%s_p%d' % (name, i)

    def count_params(self):
        return sum([np.prod(p.shape.eval()) for p in self.params])

class AffineTransform(Layer):
    def __init__(self, n_in, n_out, param_stddev_factor=0.1, bias_mean=-2.0, bias_range=2.0, learn_rate_coef = 1.0,bias_learn_rate_coef=1.0, max_norm=0.0):
        
        self.n_in = n_in
        self.n_out = n_out

        W_values = np.random.standard_normal((n_in, n_out)) * param_stddev_factor 
        W_values = W_values.astype(T.config.floatX)

        b_values = bias_mean + (np.random.uniform(0, 1, n_out) -0.5) * bias_range
        b_values = b_values.astype(T.config.floatX)
        
        W = theano.shared(W_values, name='W')
        b = theano.shared(b_values, name='b')

        (self.W, self.b) = (W, b)
        (self.W_lr_coef, self.b_lr_coef) = (learn_rate_coef, bias_learn_rate_coef)

        self.params = [self.W, self.b]
        self.lr_coefs = [self.W_lr_coef, self.b_lr_coef]
        
    def link_IO(self, input):
        self.X = input
        self.Y = T.dot(self.X, self.W) + self.b

    def get_config(self):
        return {"name": self.__class__.__name__,
                "n_in": self.n_in,
                "n_out": self.n_out}

class LinearTransform(Layer):
    def __init__(self, n_in, n_out, param_stddev_factor=0.1, learn_rate_coef = 1.0, max_norm=0.0):
        
        self.n_in = n_in
        self.n_out = n_out

        W_values = np.random.standard_normal((n_in, n_out)) * param_stddev_factor 
        W_values = W_values.astype(T.config.floatX)

        W = theano.shared(W_values, name='W')

        self.W = W
        self.W_lr_coef = learn_rate_coef

        self.params = [self.W]
        self.lr_coefs = [self.W_lr_coef]
        
    def link_IO(self, input):
        self.X = input
        self.Y = T.dot(self.X, self.W)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "n_in": self.n_in,
                "n_out": self.n_out}

class Activation(Layer):
    '''
        Apply an activation function to an output.
    '''
    def __init__(self, activation):
        super(Activation, self).__init__()
        self.activation = activations.get(activation)

    def link_IO(self, input):
        self.X = input
        self.Y = self.activation(self.X)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "activation": self.activation.__name__}


