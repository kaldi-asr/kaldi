#!/usr/bin/env python
import sys, math, logging, numpy as np, os
import theano, theano.tensor as T
import cPickle as pickle

from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from io_funcs import kaldi_io

from layers.core import AffineTransform
from layers.core import LinearTransform
from layers.core import Activation

class NeuralNet(object):
    '''
    Neural network
    '''
    def __init__(self):
        
        # neural net input is X and output is Y
        self.X = T.matrix("X")
        self.Y = self.X 

        self.layers = []
        self.params = []
        self.lr_coefs = []
        
    def add(self, layer):

        self.params += layer.params
        self.layers.append(layer)
        if hasattr(layer, 'lr_coefs'):
            self.lr_coefs += layer.lr_coefs

        #link_IO
        layer.link_IO(self.Y)
        self.Y = layer.Y
        
    def initialize_from_proto(self, proto_file):
        '''
        Function to initialize from nnet1 style prototype file
        '''
        ##################################
        def getAffineLayerDef(line):
            """
            Parses one line of Layer definition from proto file
            """
            allowed_keys = ["<OutputDim>","<InputDim>","<ParamStddev>","<BiasMean>","<BiasRange>","<LearnRateCoef>","<BiasLearnRateCoef>","<MaxNorm>"]
            params = {}
            for k,v in zip(line.split()[1:-2:2], line.split()[2::2]):
                if k not in allowed_keys:
                    print "%s parameter is not known!" %(k)
                    return None
                params[k[1:-1]] = np.float32(v)
            return params

        def getLinearLayerDef(line):
            """
            Parses one line of Layer definition from proto file
            """
            allowed_keys = ["<OutputDim>","<InputDim>","<ParamStddev>","<LearnRateCoef>","<MaxNorm>"]
            params = {}
            for k,v in zip(line.split()[1:-2:2], line.split()[2::2]):
                if k not in allowed_keys:
                    print "%s parameter is not known!" %(k)
                    return None
                params[k[1:-1]] = np.float32(v)
            return params
        ##################################
        affine_layer_params_default = {"ParamStddev":0.1, "BiasMean":-2.0, "BiasRange": 2.0,"LearnRateCoef":1.0,"BiasLearnRateCoef":1.0,"MaxNorm":0.0}
        linear_layer_params_default = {"ParamStddev":0.1, "LearnRateCoef":1.0,"MaxNorm":0.0}
        f = kaldi_io.open_or_fd(proto_file, mode='r')
        try:
            while 1:
                row = f.readline().strip()
                if not row:
                    return
                if row == "<NnetProto>" or row == "</NnetProto>":
                    continue
                else:
                    if "<AffineTransform>" in row:
                        layer_params = getAffineLayerDef(row)
                        (n_in, n_out) = (int(row.split()[2]), int(row.split()[4]))

                        [layer_params.setdefault(a, affine_layer_params_default[a]) for a in affine_layer_params_default]
                        layer = AffineTransform(n_in, n_out,
                                                param_stddev_factor=layer_params["ParamStddev"], 
                                                bias_mean=layer_params["BiasMean"], 
                                                bias_range=layer_params["BiasRange"], 
                                                learn_rate_coef=layer_params["LearnRateCoef"],
                                                bias_learn_rate_coef=layer_params["BiasLearnRateCoef"], 
                                                max_norm=layer_params["MaxNorm"])

                    elif "<LinearTransform>" in row:
                        layer_params = getLinearLayerDef(row)
                        (n_in, n_out) = (int(row.split()[2]), int(row.split()[4]))

                        [layer_params.setdefault(a, linear_layer_params_default[a]) for a in linear_layer_params_default]
                        layer = LinearTransform(n_in, n_out,
                                                param_stddev_factor=layer_params["ParamStddev"], 
                                                learn_rate_coef=layer_params["LearnRateCoef"],
                                                max_norm=layer_params["MaxNorm"])

                    elif "<Sigmoid>" in row:
                        layer = Activation("sigmoid")
                    elif "<Softmax>" in row:
                        layer = Activation("softmax")
                    else:
                        print "Not implemented. Use sigmoid or softmax"
                        sys.exit(1)

                    self.add(layer)
        finally:
            if f is not file: f.close()


    def save_weights(self, filepath):
        import bz2
        
        f = bz2.BZ2File(filepath, "wb")

        weights = []
        for ii, l in enumerate(self.layers):
            c = l.get_config()
            if c["name"] == "AffineTransform":
                c["weights"] = l.get_weights()
                c["lr_coefs"] = l.lr_coefs
            weights.append(c)
                
        pickle.dump(weights, f)
        f.close()

    def set_weights_frm_file(self, filepath):

        import bz2
        
        f = bz2.BZ2File(filepath, "rb")
        weights = pickle.load(f)
        f.close()

        assert len(weights) == len(self.layers) #same number of layers
        for ii, (w, l) in enumerate(zip(weights, self.layers)):
            l_w = l.get_config()
            assert w["name"] == l_w["name"] #same layer type
            if w["name"] == "AffineTransform":
                l.set_weights(w["weights"])

    def load_layers_frm_file(self, filepath):
        
        import bz2
        
        f = bz2.BZ2File(filepath, "rb")
        weights = pickle.load(f)
        f.close()
        
        for ii, w in enumerate(weights):
            if w["name"] == "AffineTransform":
                layer = AffineTransform(w["n_in"], w["n_out"])
                layer.set_weights(w["weights"])
            if w["name"] == "LinearTransform":
                layer = LinearTransform(w["n_in"], w["n_out"])
                layer.set_weights(w["weights"])
            elif w["name"] == "Activation":
                layer = Activation(w["activation"])
            else:
                print "ERROR: Wrong weights"
                sys.exit(1)

            self.add(layer)

    def link_IO(self, input):
        self.X = input
        self.Y = self.X

        for ii, this_layer in enumerate(self.layers):
            this_layer.link_IO(self.Y)
            self.Y = this_layer.Y

