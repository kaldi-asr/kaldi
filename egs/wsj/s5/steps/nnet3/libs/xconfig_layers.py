from __future__ import print_function
import subprocess
import logging
import math
import re
import sys
import traceback
import time
import argparse
from xconfig_lib import *

# Given a list of objects of type XconfigLayerBase ('all_layers'),
# including at least the layers preceding 'current_layer' (and maybe
# more layers), return the names of layers preceding 'current_layer'
# This will be used in parsing expressions like [-1] in descriptors
# (which is an alias for the previous layer).
def GetPrevNames(all_layers, current_layer):
    assert current_layer in all_layers
    prev_names = []
    for layer in all_layers:
        if layer is current_layer:
            break
        prev_names.append(layer.Name())
    return prev_names

# this converts a layer-name like 'ivector' or 'input', or a sub-layer name like
# 'lstm2.memory_cell', into a dimension.  'all_layers' is a vector of objects
# inheriting from XconfigLayerBase.  'current_layer' is provided so that the
# function can make sure not to look in layers that appear *after* this layer
# (because that's not allowed).
def GetDimFromLayerName(all_layers, current_layer, full_layer_name):
    assert isinstance(full_layer_name, str)
    split_name = full_layer_name.split('.')
    if len(split_name) == 0:
        raise Exception("Bad layer name: " + full_layer_name)
    layer_name = split_name[0]
    if len(split_name) == 1:
        qualifier = None
    else:
        # we probably expect len(split_name) == 2 in this case,
        # but no harm in allowing dots in the qualifier.
        qualifier = '.'.join(split_name[1:])

    for layer in all_layers:
        if layer is current_layer:
            break
        if layer.Name() == layer_name:
            if not qualifier in layer.Qualifiers():
                raise Exception("Layer '{0}' has no such qualifier: '{1}' ({0}.{1})".format(
                    layer_name, qualifier))
            return layer.OutputDim(qualifier)
    # No such layer was found.
    if layer_name in [ layer.Name() for layer in all_layers ]:
        raise Exception("Layer '{0}' was requested before it appeared in "
                        "the xconfig file (circular dependencies or out-of-order "
                        "layers".format(layer_name))
    else:
        raise Exception("No such layer: '{0}'".format(layer_name))


# this converts a layer-name like 'ivector' or 'input', or a sub-layer name like
# 'lstm2.memory_cell', into a descriptor (usually, but not required to be a simple
# component-node name) that can appear in the generated config file.  'all_layers' is a vector of objects
# inheriting from XconfigLayerBase.  'current_layer' is provided so that the
# function can make sure not to look in layers that appear *after* this layer
# (because that's not allowed).
def GetStringFromLayerName(all_layers, current_layer, full_layer_name):
    assert isinstance(full_layer_name, str)
    split_name = full_layer_name.split('.')
    if len(split_name) == 0:
        raise Exception("Bad layer name: " + full_layer_name)
    layer_name = split_name[0]
    if len(split_name) == 1:
        qualifier = None
    else:
        # we probably expect len(split_name) == 2 in this case,
        # but no harm in allowing dots in the qualifier.
        qualifier = '.'.join(split_name[1:])

    for layer in all_layers:
        if layer is current_layer:
            break
        if layer.Name() == layer_name:
            if not qualifier in layer.Qualifiers():
                raise Exception("Layer '{0}' has no such qualifier: '{1}' ({0}.{1})".format(
                    layer_name, qualifier))
            return layer.OutputName(qualifier)
    # No such layer was found.
    if layer_name in [ layer.Name() for layer in all_layers ]:
        raise Exception("Layer '{0}' was requested before it appeared in "
                        "the xconfig file (circular dependencies or out-of-order "
                        "layers".format(layer_name))
    else:
        raise Exception("No such layer: '{0}'".format(layer_name))



# A base-class for classes representing layers of xconfig files.
# This mainly just sets self.layer_type, self.name and self.config,
class XconfigLayerBase(object):
    # Constructor.
    # first_token is the first token on the xconfig line, e.g. 'affine-layer'.f
    # key_to_value is a dict like:
    # { 'name':'affine1', 'input':'Append(0, 1, 2, ReplaceIndex(ivector, t, 0))', 'dim=1024' }.
    # The only required and 'special' values that are dealt with directly at this level, are
    # 'name' and 'input'.
    # The rest are put in self.config and are dealt with by the child classes' init functions.
    # prev_names is an array of the names (xxx in 'name=xxx') of previous
    # lines of the config file.

    def __init__(self, first_token, key_to_value, prev_names = None):
        self.layer_type = first_token
        if not 'name' in key_to_value:
            raise Exception("Expected 'name' to be specified.")
        self.name = key_to_value['name']
        if not IsValidLineName(self.name):
            raise Exception("Invalid value: name={0}".format(key_to_value['name']))

        # the following, which should be overridden in the child class, sets
        # default config parameters in self.config.
        self.SetDefaultConfigs()
        # The following is not to be reimplemented in child classes;
        # sets the config files to those specified by the user.
        self._SetConfigs(key_to_value)
        # the following, which should be overridden in the child class, checks
        # that the config parameters that have been set are reasonable.
        self.CheckConfigs()


    # We broke this code out of __init__ for clarity.
    def _SetConfigs(self, key_to_value):
        # the child-class constructor will deal with the configuration values
        # in a more specific way.
        for key,value in key_to_value.items():
            if key != 'name' and key != 'input':
                if not key in self.config:
                    raise Exception("Configuration value {0}={1} was not expected in "
                                    "layer of type {2}".format(key, value, self.layer_type))
                self.config[key] = ConvertValueToType(key, type(self.config[key]), value)


    # This function converts 'this' to a string which could be printed to an
    # xconfig file; in xconfig_to_configs.py we actually expand all the lines to
    # strings and write it as xconfig.expanded as a reference (so users can
    # see any defaults).
    def str(self):
        ans = '{0} name={1}'.format(self.layer_type, self.name)
        ans += ' ' + ' '.join([ '{0}={1}'.format(key, self.config[key])
                                for key in sorted(self.config.keys())])
        return ans

    def __str__(self):
        return self.str()

    # This function, which is a convenience function intended to be called from
    # child classes, converts a string representing a descriptor
    # ('descriptor_string') into an object of type Descriptor, and returns it.
    # It needs 'self' and 'all_layers' (where 'all_layers' is a list of objects
    # of type XconfigLayerBase) so that it can work out a list of the names of
    # other layers, and get dimensions from them.
    def ConvertToDescriptor(self, descriptor_string, all_layers):
        prev_names = GetPrevNames(all_layers, self)
        tokens = TokenizeDescriptor(descriptor_string, prev_names)
        pos = 0
        (self.input, pos) = ParseNewDescriptor(tokens, pos, prev_names)
        # note: 'pos' should point to the 'end of string' marker
        # that terminates 'tokens'.
        if pos != len(tokens) - 1:
            raise Exception("Parsing Descriptor, saw junk at end: " +
                            ' '.join(tokens[pos:-1]))

    # Returns the dimension of a Descriptor object.
    # This is a convenience function provided for use in child classes;
    def GetDimForDescriptor(self, descriptor, all_layers):
        layer_to_dim_func = lambda name: GetDimFromLayerName(all_layers, self, name)
        return descriptor.Dim(layer_to_dim_func)

    # Returns the 'final' string form of a Descriptor object, as could be used
    # in config files.
    # This is a convenience function provided for use in child classes;
    def GetStringForDescriptor(self, descriptor, all_layers):
        layer_to_string_func = lambda name: GetStringFromLayerName(all_layers, self, name)
        return descriptor.ConfigString(layer_to_string_func)

    # Name() returns the name of this layer, e.g. 'affine1'.  It does not
    # necessarily correspond to a component name.
    def Name():
        return self.name

    ######  Functions that should be overridden by the child class: #####

    # child classes should override this.
    def SetDefaultConfigs():
        raise Exception("Child classes must override SetDefaultConfigs().")

    # child classes should override this.
    def CheckConfigs():
        pass

    # Returns a list of all qualifiers (meaning auxiliary outputs) that this
    # layer supports.  These are either 'None' for the regular output, or a
    # string (e.g. 'projection' or 'memory_cell') for any auxiliary outputs that
    # the layer might provide.  Most layer types will not need to override this.
    def Qualifiers():
        return [ None ]

    # Called with qualifier == None, this returns the component-node name of the
    # principal output of the layer (or if you prefer, the text form of a
    # descriptor that gives you such an output; such as Append(some_node,
    # some_other_node)).
    # The 'qualifier' argument is a text value that is designed for extensions
    # to layers that have additional auxiliary outputs.  For example, to implement
    # a highway LSTM you need the memory-cell of a layer, so you might allow
    # qualifier='memory_cell' for such a layer type, and it would return the
    # component node or a suitable Descriptor: something like 'lstm3.c_t'
    def OutputName(qualifier = None):
        raise Exception("Child classes must override OutputName()")

    # The dimension that this layer outputs.  The 'qualifier' parameter is for
    # layer types which support auxiliary outputs.
    def OutputDim(qualifier = None):
        raise Exception("Child classes must override OutputDim()")

    # This function returns lines destined for the 'full' config format, as
    # would be read by the C++ programs.
    # Since the program xconfig_to_configs.py writes several config files, this
    # function returns a list of pairs of the form (config_file_basename, line),
    # e.g. something like
    # [ ('init', 'input-node name=input dim=40'),
    #   ('ref', 'input-node name=input dim=40') ]
    # which would be written to config_dir/init.config and config_dir/ref.config.
    #
    # 'all_layers' is a vector of objects inheriting from XconfigLayerBase,
    # which is used to get the component names and dimensions at the input.
    def GetFullConfig(self, all_layers):
        raise Exception("Child classes must override GetFullConfig()")


# This class is for lines like
# 'input name=input dim=40'
# or
# 'input name=ivector dim=100'
# in the config file.
class XconfigInputLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == 'input'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)


    def SetDefaultConfigs(self):
        self.config = { 'dim':-1 }

    def CheckConfigs(self):
        if self.config['dim'] <= 0:
            raise Exception("Dimension of input-layer '{0}' is not set".format(self.name))

    def OutputName(qualifier = None):
        assert qualifier is None
        return self.name

    def OutputDim(qualifier = None):
        assert qualifier is None
        return self.config['dim']

    def GetFullConfig(self, all_layers):
        # the input layers need to be printed in 'init.config' (which
        # initializes the neural network prior to the LDA), in 'ref.config',
        # which is a version of the config file used for getting left and right
        # context (it doesn't read anything for the LDA-like transform and/or
        # presoftmax-prior-scale components)
        # In 'full.config' we write everything, this is just for reference,
        # and also for cases where we don't use the LDA-like transform.
        ans = []
        for config_name in [ 'init', 'ref', 'full' ]:
            ans.append( (config_name,
                         'input-node name={0} dim={1}'.format(self.name,
                                                              self.config['dim'])))
        return ans


# Converts a line as parsed by ParseConfigLine() into a first
# token e.g. 'input-layer' and a key->value map, into
# an objet inherited from XconfigLayerBase.
# 'prev_names' is a list of previous layer names, it's needed
# to parse things like '[-1]' (meaning: the previous layer)
# when they appear in Desriptors.
def ParsedLineToXconfigLayer(first_token, key_to_value, prev_names):
    if first_token == 'input':
        return XconfigInputLayer(first_token, key_to_value, prev_names)
    else:
        raise Exception("Error parsing xconfig line (no such layer type): " +
                        first_token + ' ' +
                        ' '.join(['{0} {1}'.format(x,y) for x,y in key_to_value.items()]))


# Uses ParseConfigLine() to turn a config line that has been parsed into
# a first token e.g. 'affine-layer' and a key->value map like { 'dim':'1024', 'name':'affine1' },
# and then turns this into an object representing that line of the config file.
# 'prev_names' is a list of the names of preceding lines of the
# config file.
def ConfigLineToObject(config_line, prev_names = None):
    (first_token, key_to_value) = ParseConfigLine(config_line)
    return ParsedLineToXconfigLayer(first_token, key_to_value, prev_names)



# This function reads an xconfig file and returns it as a list of layers
# (usually we use the variable name 'all_layers' elsewhere for this).
# It will die if the xconfig file is empty or if there was
# some error parsing it.
def ReadXconfigFile(xconfig_filename):
    try:
        f = open(xconfig_filename, 'r')
    except Exception as e:
        sys.exit("{0}: error reading xconfig file '{1}'; error was {2}".format(
            sys.argv[0], xconfig_filename, repr(e)))
    prev_names = []
    all_layers = []
    while True:
        line = f.readline()
        if line == '':
            break
        x = ParseConfigLine(config_line)
        if x is None:
            continue   # line was blank or only comments.
        (first_token, key_to_value) = x
        # the next call will raise an easy-to-understand exception if
        # it fails.
        this_layer = ParsedLineToXconfigLayer(first_token,
                                              key_to_value,
                                              prev_names)
        prev_names.append(this_layer.Name())
        all_layers.append(this_layer)
    if len(all_layers) == 0:
        raise Exception("{0}: xconfig file '{1}' is empty".format(
            sys.argv[0], xconfig_filename))
    f.close()
    return all_layers


def TestLayers():
    # for some config lines that should be printed the same way as they
    # are read, check that this is the case.
    for x in [ 'input name=input dim=30' ]:
        assert str(ConfigLineToObject(x, [])) == x
