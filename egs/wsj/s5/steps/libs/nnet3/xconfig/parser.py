# Copyright 2016    Johns Hopkins University (Dan Povey)
#           2016    Vijayaditya Peddinti
# Apache 2.0.

""" This module contains the top level xconfig parsing functions.
"""

from __future__ import print_function

import logging
import sys
import libs.nnet3.xconfig.layers as xlayers
import libs.nnet3.xconfig.utils as xutils



# We have to modify this dictionary when adding new layers
config_to_layer = {
        'input' : xlayers.XconfigInputLayer,
        'output' : xlayers.XconfigTrivialOutputLayer,
        'output-layer' : xlayers.XconfigOutputLayer,
        'relu-layer' : xlayers.XconfigBasicLayer,
        'relu-renorm-layer' : xlayers.XconfigBasicLayer,
        'relu-batchnorm-dropout-layer' : xlayers.XconfigBasicLayer,
        'relu-dropout-layer': xlayers.XconfigBasicLayer,
        'relu-batchnorm-layer' : xlayers.XconfigBasicLayer,
        'sigmoid-layer' : xlayers.XconfigBasicLayer,
        'tanh-layer' : xlayers.XconfigBasicLayer,
        'fixed-affine-layer' : xlayers.XconfigFixedAffineLayer,
        'idct-layer' : xlayers.XconfigIdctLayer,
        'affine-layer' : xlayers.XconfigAffineLayer,
        'lstm-layer' : xlayers.XconfigLstmLayer,
        'lstmp-layer' : xlayers.XconfigLstmpLayer,
        'fast-lstm-layer' : xlayers.XconfigFastLstmLayer,
        'fast-lstmp-layer' : xlayers.XconfigFastLstmpLayer,
        'stats-layer': xlayers.XconfigStatsLayer,
        'relu-conv-layer': xlayers.XconfigConvLayer,
        'conv-layer': xlayers.XconfigConvLayer,
        'conv-relu-layer': xlayers.XconfigConvLayer,
        'relu-conv-renorm-layer': xlayers.XconfigConvLayer,
        'conv-relu-renorm-layer': xlayers.XconfigConvLayer,
        'batchnorm-conv-relu-layer': xlayers.XconfigConvLayer,
        'relu-batchnorm-conv-layer': xlayers.XconfigConvLayer,
        'conv-relu-batchnorm-layer': xlayers.XconfigConvLayer,
        'conv-relu-batchnorm-dropout-layer': xlayers.XconfigConvLayer,
        'conv-relu-dropout-layer': xlayers.XconfigConvLayer,
        'res-block': xlayers.XconfigResBlock,
        'channel-average-layer': xlayers.ChannelAverageLayer
}

# Turn a config line and a list of previous layers into
# either an object representing that line of the config file; or None
# if the line was empty after removing comments.
# 'prev_layers' is a list of objects corresponding to preceding layers of the
# config file.
def xconfig_line_to_object(config_line, prev_layers = None):
    try:
        x  = xutils.parse_config_line(config_line)
        if x is None:
            return None
        (first_token, key_to_value) = x
        if not config_to_layer.has_key(first_token):
            raise RuntimeError("No such layer type '{0}'".format(first_token))
        return config_to_layer[first_token](first_token, key_to_value, prev_layers)
    except Exception:
        logging.error(
            "***Exception caught while parsing the following xconfig line:\n"
            "*** {0}".format(config_line))
        raise

# This function reads an xconfig file and returns it as a list of layers
# (usually we use the variable name 'all_layers' elsewhere for this).
# It will die if the xconfig file is empty or if there was
# some error parsing it.
def read_xconfig_file(xconfig_filename):
    try:
        f = open(xconfig_filename, 'r')
    except Exception as e:
        sys.exit("{0}: error reading xconfig file '{1}'; error was {2}".format(
            sys.argv[0], xconfig_filename, repr(e)))
    all_layers = []
    while True:
        line = f.readline()
        if line == '':
            break
        # the next call will raise an easy-to-understand exception if
        # it fails.
        this_layer = xconfig_line_to_object(line, all_layers)
        if this_layer is None:
            continue  # line was blank after removing comments.
        all_layers.append(this_layer)
    if len(all_layers) == 0:
        raise RuntimeError("{0}: xconfig file '{1}' is empty".format(
            sys.argv[0], xconfig_filename))
    f.close()
    return all_layers
