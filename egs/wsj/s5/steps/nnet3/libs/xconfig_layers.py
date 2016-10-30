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

# This class represents a line that starts with 'input', e.g.
# 'input name=ivector dim=100', or 'input name=input dim=40'
class XconfigInputLine:
    # Constructor.
    # first_token must be the string 'input'.
    # key_to_value is a dict like { 'name':'ivector', 'dim':'100' }.
    # 'prev_names' is a list of the names of preceding lines of the
    # config file; it's not used here but is part of the common
    # interface for xconfig input line constructors.
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == 'input'
        if not 'name' in key_to_value:
            raise Exception("Config line for input does not specify name.")
        self.name = key_to_value['name']
        if not IsValidLineName(self.name):
            raise Exception("Name '{0}' is not a valid node name.".format(self.name))
        if not 'dim' in key_to_value:
            raise Exception("Config line for input does not specify dimension.")
        try:
            self.dim = int(key_to_value['dim'])
            assert self.dim > 0
        if len(key_to_value) > 2:
            raise Exception("Unused name=value pairs in config line")
        except:
            raise Exception("Dimension '{0}' is not valid.".format(key_to_value['dim']))


    # This returns the name of the layer, e.g. 'input' or 'ivector'.
    def Name():
        return self.name

    # This returns the component-node name of the principal output of the layer.  For
    # the input layer this is the same as the name.  For an affine layer
    # 'affine1' it might be e.g. 'affine1.renorm'.
    # The 'qualifier' parameter is for compatibility with other layer
    # types, which support auxiliary outputs.
    def OutputName(qualifier = None):
        assert qualifier == None
        return self.name

    # The dimension that this layer outputs.
    # OutputDim().
    # The 'qualifier' parameter is for compatibility with other layer
    # types, which support auxiliary outputs.
    def OutputDim(qualifier = None):
        assert qualifier == None
        return self.dim

    # Returns a list of all qualifiers (meaning auxiliary outputs) that this
    # layer supports (these are either 'None' for the regular output, or a
    # string such as 'projection' or something like that, for auxiliary outputs.
    def Qualifiers():
        return [ None ]

    # This function writes the 'full' config format, as would be read
    # by the C++ programs.  It writes the config lines to 'file'.
    # 'all_layers' is a vector of objects (of type XConfigInputLine or
    # inheriting from XconfigLayerBase), which is used to get
    # the component names and
    def GetFullConfig(self, file, all_layers):
        print("input-node name={0} dim={0}".format(self.name, self.dim)

    def str(self):
        return 'input name={0} dim={1}'.format(self.name, self.dim)

    def __str__(self):
        return self.str()



# A base-class for classes representing layers of xconfig files (but not input
# nodes).  This handles parsing the Descriptors and other common tasks.
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
        if not 'name' in key_to_value
            raise Exception("Expected 'name' to be specified.")
        self.name = key_to_value['name']
        if not IsValidLineName(self.name):
            raise Exception("Invalid value: name={0}".format(key_to_value['name']))

        if not 'input' in key_to_value
            raise Exception("Expected 'name' to be specified.")
        input_descriptor_str = key_to_value[input]
        tokens = TokenizeDescriptor(input_descriptor_str, prev_names)
        pos = 0
        (self.input, pos) = ParseNewDescriptor(tokens, pos, prev_names)
        # note: 'pos' should point to the 'end of string' marker
        # that terminates 'tokens'.
        if pos != len(tokens) - 1:
            raise Exception("Parsing Descriptor, saw junk at end: " +
                            ' '.join(tokens[pos:-1]))
        # the following, which should be overridden in the child class, sets
        # default config parameters in self.config.
        self.SetDefaultConfigs()
        self._OverrideConfigs()
        # the following, which should be overridden in the child class, checks
        # that the config parameters that have been set are reasonable.
        self.CheckConfigs()


    # We broke this code out of __init__ for clarity.
    def _OverrideConfigs(key_to_value):
        # the child-class constructor will deal with the configuration values
        # in a more specific way.
        for key,value in key_to_value.items():
            if key != 'name' and key != 'input':
                if not key in self.config:
                    raise Exception("Configuration value {0}={1} was not expected in "
                                    "layer of type {2}".format(key, value, self.layer_type))
                if isinstance(value, bool):
                self.config[key] = ConvertValueToType(key, type(self.config[key]),
                                                      value)

    def GetDefaultConfigs():
        raise Exception("Child classes must override GetDefaultConfigs().")


    # child classes may override this but do not have to.
    def CheckConfigs():
        pass


    # Returns a list of all qualifiers (meaning auxiliary outputs) that this
    # layer supports (these are either 'None' for the regular output, or a
    # string such as 'projection' or something like that, for auxiliary outputs.
    # This is a default implementation of the function.
    def Qualifiers():
        return [ None ]

    # This returns the component-node name of the principal output of the layer.  For
    # the input layer this is the same as the name.  For an affine layer
    # 'affine1' it might be e.g. 'affine1.renorm'.
    # The 'qualifier' parameter is for compatibility with other layer
    # types, which support auxiliary outputs.
    def OutputName(qualifier = None):
        raise Exception("Child classes must override OutputName()")

    # The dimension that this layer outputs.
    # The 'qualifier' parameter is to support
    # types, which support auxiliary outputs.
    def OutputDim(qualifier = None):
        raise Exception("Child classes must override OutputDim()")


    # This function writes the 'full' config format, as would be read
    # by the C++ programs.  It writes the config lines to 'file'.
    # 'all_layers' is a vector of objects (of type XConfigInputLine or
    # inheriting from XconfigLayerBase), which is used to get
    # the component names and dimensions at the input.
    def GetFullConfig(self, file, all_layers):
        raise Exception("Child classes must override GetFullConfig()")

    # Name() returns the name of this layer, e.g. 'affine1'.  It does not
    # necessarily correspond to a component name.
    def Name():
        return self.name

    def str(self):
        ans = '{0} name={1}'.format(self.layer_type, self.name)
        ans += ' ' + ' '.join([ '{0}={1}'.format(key, self.config[key])
                                for key in sorted(self.config.keys())])
        return ans

    def __str__(self):
        return self.str()



# Uses ParseConfigLine() to turn a config line that has been parsed into
# a first token e.g. 'affine-layer' and a key->value map like { 'dim':'1024', 'name':'affine1' },
# and then turns this into an object representing that line of the config file.
# 'prev_names' is a list of the names of preceding lines of the
# config file.
def ConfigLineToObject(config_line, prev_names = None):
    (first_token, key_to_value) = ParseConfigLine(config_line)

    if first_token == 'input':
        return XconfigInputLine(key_to_value)


def TestLayers():
    # for some config lines that should be printed the same way as they
    # are read, check that this is the case.
    for x in [ 'input name=input dim=30' ]:
        assert str(ConfigLineToObject(x, [])) == x
