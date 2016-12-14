# Copyright 2016    Johns Hopkins University (Dan Povey)
#           2016    Vijayaditya Peddinti
# Apache 2.0.

""" This module contains the top level xconfig parsing functions.
"""

import libs.nnet3.xconfig.layers as xlayers
import libs.nnet3.xconfig.utils as xutils
from libs.nnet3.xconfig.utils import XconfigParserError as xparser_error


# We have to modify this dictionary when adding new layers
config_to_layer = {
        'input' : xlayers.XconfigInputLayer,
        'output' : xlayers.XconfigTrivialOutputLayer,
        'output-layer' : xlayers.XconfigOutputLayer,
        'relu-layer' : xlayers.XconfigBasicLayer,
        'relu-renorm-layer' : xlayers.XconfigBasicLayer,
        'sigmoid-layer' : xlayers.XconfigBasicLayer,
        'tanh-layer' : xlayers.XconfigBasicLayer,
        'tdnn-relu-layer' : xlayers.XconfigTdnnLayer,
        'tdnn-relu-renorm-layer' : xlayers.XconfigTdnnLayer,
        'tdnn-sigmoid-layer' : xlayers.XconfigTdnnLayer,
        'tdnn-tanh-layer' : xlayers.XconfigTdnnLayer,
        'fixed-affine-layer' : xlayers.XconfigFixedAffineLayer,
        'affine-layer' : xlayers.XconfigAffineLayer,
        'lstm-layer' : xlayers.XconfigLstmLayer,
        'lstmp-layer' : xlayers.XconfigLstmpLayer,
        'lstmpc-layer' : xlayers.XconfigLstmpcLayer
        }

# Converts a line as parsed by ParseConfigLine() into a first
# token e.g. 'input-layer' and a key->value map, into
# an objet inherited from XconfigLayerBase.
# 'prev_names' is a list of previous layer names, it's needed
# to parse things like '[-1]' (meaning: the previous layer)
# when they appear in Desriptors.
def parsed_line_to_xconfig_layer(first_token, key_to_value, prev_names):

    conf_line = first_token + ' ' + ' '.join(['{0}={1}'.format(x,y) for x,y in key_to_value.items()])

    if not config_to_layer.has_key(first_token):
        raise xparser_error("No such layer type.", conf_line)

    try:
        return config_to_layer[first_token](first_token, key_to_value, prev_names)
    except xparser_error as e:
        if e.conf_line is None:
            # we want to throw informative errors which point to the xconfig line
            e.conf_line = conf_line
        raise

# Uses ParseConfigLine() to turn a config line that has been parsed into
# a first token e.g. 'affine-layer' and a key->value map like { 'dim':'1024', 'name':'affine1' },
# and then turns this into an object representing that line of the config file.
# 'prev_names' is a list of the names of preceding lines of the
# config file.
def config_line_to_object(config_line, prev_names = None):
    (first_token, key_to_value) = xutils.parse_config_line(config_line)
    return parsed_line_to_xconfig_layer(first_token, key_to_value, prev_names)

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
        x = xutils.parse_config_line(line)
        if x is None:
            continue   # line was blank or only comments.
        (first_token, key_to_value) = x
        # the next call will raise an easy-to-understand exception if
        # it fails.
        this_layer = parsed_line_to_xconfig_layer(first_token,
                                                  key_to_value,
                                                  all_layers)
        all_layers.append(this_layer)
    if len(all_layers) == 0:
        raise xparser_error("{0}: xconfig file '{1}' is empty".format(
            sys.argv[0], xconfig_filename))
    f.close()
    return all_layers


