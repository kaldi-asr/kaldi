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

import libs.common as common_lib


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
        'relu-batchnorm-so-layer' : xlayers.XconfigBasicLayer,
        'batchnorm-so-relu-layer' : xlayers.XconfigBasicLayer,
        'sigmoid-layer' : xlayers.XconfigBasicLayer,
        'tanh-layer' : xlayers.XconfigBasicLayer,
        'fixed-affine-layer' : xlayers.XconfigFixedAffineLayer,
        'idct-layer' : xlayers.XconfigIdctLayer,
        'affine-layer' : xlayers.XconfigAffineLayer,
        'lstm-layer' : xlayers.XconfigLstmLayer,
        'lstmp-layer' : xlayers.XconfigLstmpLayer,
        'lstmp-batchnorm-layer' : xlayers.XconfigLstmpLayer,
        'fast-lstm-layer' : xlayers.XconfigFastLstmLayer,
        'fast-lstm-batchnorm-layer' : xlayers.XconfigFastLstmLayer,
        'fast-lstmp-layer' : xlayers.XconfigFastLstmpLayer,
        'fast-lstmp-batchnorm-layer' : xlayers.XconfigFastLstmpLayer,
        'lstmb-layer' : xlayers.XconfigLstmbLayer,
        'stats-layer': xlayers.XconfigStatsLayer,
        'relu-conv-layer': xlayers.XconfigConvLayer,
        'conv-layer': xlayers.XconfigConvLayer,
        'conv-relu-layer': xlayers.XconfigConvLayer,
        'conv-renorm-layer': xlayers.XconfigConvLayer,
        'relu-conv-renorm-layer': xlayers.XconfigConvLayer,
        'batchnorm-conv-layer': xlayers.XconfigConvLayer,
        'conv-relu-renorm-layer': xlayers.XconfigConvLayer,
        'batchnorm-conv-relu-layer': xlayers.XconfigConvLayer,
        'relu-batchnorm-conv-layer': xlayers.XconfigConvLayer,
        'relu-batchnorm-noconv-layer': xlayers.XconfigConvLayer,
        'relu-noconv-layer': xlayers.XconfigConvLayer,
        'conv-relu-batchnorm-layer': xlayers.XconfigConvLayer,
        'conv-relu-batchnorm-so-layer': xlayers.XconfigConvLayer,
        'conv-relu-batchnorm-dropout-layer': xlayers.XconfigConvLayer,
        'conv-relu-dropout-layer': xlayers.XconfigConvLayer,
        'res-block': xlayers.XconfigResBlock,
        'res2-block': xlayers.XconfigRes2Block,
        'channel-average-layer': xlayers.ChannelAverageLayer,
        'attention-renorm-layer': xlayers.XconfigAttentionLayer,
        'attention-relu-renorm-layer': xlayers.XconfigAttentionLayer,
        'attention-relu-batchnorm-layer': xlayers.XconfigAttentionLayer,
        'relu-renorm-attention-layer': xlayers.XconfigAttentionLayer,
        'gru-layer' : xlayers.XconfigGruLayer,
        'pgru-layer' : xlayers.XconfigPgruLayer,
        'opgru-layer' : xlayers.XconfigOpgruLayer,
        'norm-pgru-layer' : xlayers.XconfigNormPgruLayer,
        'norm-opgru-layer' : xlayers.XconfigNormOpgruLayer,
        'renorm-component': xlayers.XconfigRenormComponent,
        'no-op-component': xlayers.XconfigNoOpComponent,
        'linear-component': xlayers.XconfigLinearComponent
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


def get_model_component_info(model_filename):
    """
    This function reads existing model (*.raw or *.mdl) and returns array
    of XconfigExistingLayer one per {input,output}-node or component-node
    with same 'name' used in the raw model and 'dim' equal to 'output-dim'
    for component-node and 'dim' for {input,output}-node.

    e.g. layer in *.mdl -> corresponding 'XconfigExistingLayer' layer
         'input-node name=ivector dim=100' ->
         'existing name=ivector dim=100'
         'component-node name=tdnn1.affine ... input-dim=1000 '
         'output-dim=500' ->
         'existing name=tdnn1.affine dim=500'
    """

    all_layers = []
    try:
        f = open(model_filename, 'r')
    except Exception as e:
        sys.exit("{0}: error reading model file '{1}'".format(sys.argv[0],
                                                              model_filename,
                                                              repr(e)))

    # use nnet3-info to get component names in the model.
    out = common_lib.get_command_stdout("""nnet3-info {0} | grep '\-node' """
                                        """ """.format(model_filename))

    # out contains all {output, input, component}-nodes used in model_filename
    # It can parse lines in out like:
    # i.e. input-node name=input dim=40
    #   component-node name=tdnn1.affine component=tdnn1.affine input=lda
    #   input-dim=300 output-dim=512
    layer_names = []
    key_to_value = dict()
    for line in out.split("\n"):
        parts = line.split(" ")
        dim = -1
        for  field in parts:
            key_value = field.split("=")
            if len(key_value) == 2:
                key = key_value[0]
                value = key_value[1]
                if key == "name":           # name=**
                    layer_name = value
                elif key == "dim":          # for input-node
                    dim = int(value)
                elif key == "output-dim":   # for component-node
                    dim = int(value)

        if layer_name is not None and layer_name not in layer_names:
            layer_names.append(layer_name)
            key_to_value['name'] = layer_name
            assert(dim != -1)
            key_to_value['dim'] = dim
            all_layers.append(xlayers.XconfigExistingLayer('existing', key_to_value, all_layers))
    if len(all_layers) == 0:
        raise RuntimeError("{0}: model filename '{1}' is empty.".format(
            sys.argv[0], model_filename))
    f.close()
    return all_layers


# This function reads xconfig file and returns it as a list of layers
# (usually we use the variable name 'all_layers' elsewhere for this).
# It will die if the xconfig file is empty or if there was
# some error parsing it.
# 'existing_layers' contains some layers of type 'existing' (layers which are not really
# layers but are actual component node names from an existing neural net model
# and created using get_model_component_info function).
# 'existing' layers can be used as input to component-nodes in layers of xconfig file.
def read_xconfig_file(xconfig_filename, existing_layers=[]):
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
        this_layer = xconfig_line_to_object(line, existing_layers)
        if this_layer is None:
            continue  # line was blank after removing comments.
        all_layers.append(this_layer)
        existing_layers.append(this_layer)
    if len(all_layers) == 0:
        raise RuntimeError("{0}: xconfig file '{1}' is empty".format(
            sys.argv[0], xconfig_filename))
    f.close()
    return all_layers
