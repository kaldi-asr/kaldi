# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import theano
import theano.tensor as T
import numpy as np

from collections import OrderedDict

from .. import activations, initializations, regularizers, constraints
from ..utils.theano_utils import shared_zeros, floatX
from ..utils.generic_utils import make_tuple
from ..regularizers import ActivityRegularizer, Regularizer

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from six.moves import zip


from .core import Dense

class AffineTransform(Dense):
    '''
        Just your regular fully connected NN layer.
        Differs from Dense in initialization
    '''
    def __init__(self, input_dim, output_dim, init='kaldi_nnet1', activation='linear', weights=None, name=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, mode=""):

        super(AffineTransform, self).__init__(input_dim, output_dim, init='kaldi_nnet1', activation='linear', weights=None, name=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None)
        '''
        mode: {'input', 'bottleneck_out', 'bottleneck_inp', 'output'}
        '''

        self.mode = mode #default is hidden
        
        #set learing rate 
        (self.W_lr_coef, self.b_lr_coef) = (1.0, 1.0)

        if self.mode == 'output':
            self.b_lr_coef = 0.1
        elif self.mode == 'bottleneck_out':
            self.W_lr_coef = 0.1
            self.b_lr_coef = 0.0 # = LinearTransform
        elif self.mode == 'bottleneck_inp':
            self.W_lr_coef = 0.1
            self.b_lr_coef = 0.1 
        
        self.lr_coefs = [self.W_lr_coef, self.b_lr_coef]
        
        visible=True if self.mode == 'input' else False
        self.W = theano.shared(self.init((self.input_dim, self.output_dim), visible=visible), name='W')

        #set bias_mean, bias_range
        (bias_mean, bias_range) = (-2.000000, 4.000000)
        if self.mode == 'output':
            (bias_mean, bias_range) = (0.000000, 0.000000)

        b_values = bias_mean + (np.random.uniform(0, 1, output_dim) -0.5) * bias_range
        b_values = b_values.astype(T.config.floatX)
        self.b = theano.shared(b_values, name='n')

    def link_IO(self, input):
        self.X = input
        self.Y = T.dot(self.X, self.W) + self.b


