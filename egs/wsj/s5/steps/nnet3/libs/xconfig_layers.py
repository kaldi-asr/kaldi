from __future__ import print_function
import subprocess
import logging
import math
import re
import sys
import traceback
import time
import argparse
import xconfig_utils


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
    # all_layers is an array of objects inheriting XconfigLayerBase for all previously
    # parsed layers.

    def __init__(self, first_token, key_to_value, all_layers):
        self.layer_type = first_token
        if not 'name' in key_to_value:
            raise RuntimeError("Expected 'name' to be specified.")
        self.name = key_to_value['name']
        if not xconfig_utils.IsValidLineName(self.name):
            raise RuntimeError("Invalid value: name={0}".format(key_to_value['name']))

        # the following, which should be overridden in the child class, sets
        # default config parameters in self.config.
        self.SetDefaultConfigs()
        # The following is not to be reimplemented in child classes;
        # it sets the config values to those specified by the user, and
        # parses any Descriptors.
        self.SetConfigs(key_to_value, all_layers)
        # the following, which should be overridden in the child class, checks
        # that the config parameters that have been set are reasonable.
        self.CheckConfigs()


    # We broke this code out of __init__ for clarity.
    def SetConfigs(self, key_to_value, all_layers):
        # the child-class constructor will deal with the configuration values
        # in a more specific way.
        for key,value in key_to_value.items():
            if key != 'name':
                if not key in self.config:
                    raise RuntimeError("Configuration value {0}={1} was not expected in "
                                    "layer of type {2}".format(key, value, self.layer_type))
                self.config[key] = xconfig_utils.ConvertValueToType(key, type(self.config[key]), value)


        self.descriptors = dict()
        self.descriptor_dims = dict()
        # Parse Descriptors and get their dims and their 'final' string form.
        # Put them as 4-tuples (descriptor, string, normalized-string, final-string)
        # in self.descriptors[key]
        for key in self.GetDescriptorConfigs():
            if not key in self.config:
                raise RuntimeError("{0}: object of type {1} needs to override "
                                   "GetDescriptorConfigs()".format(sys.argv[0],
                                                                   str(type(self))))
            descriptor_string = self.config[key]  # input string.
            assert isinstance(descriptor_string, str)
            desc = self.ConvertToDescriptor(descriptor_string, all_layers)
            desc_dim = self.GetDimForDescriptor(desc, all_layers)
            desc_norm_str = desc.str()
            # desc_output_str contains the "final" component names, those that
            # appear in the actual config file (i.e. not names like
            # 'layer.qualifier'); that's how it differs from desc_norm_str.
            # Note: it's possible that the two strings might be the same in
            # many, even most, cases-- it depends whether OutputName(self, qualifier)
            # returns self.Name() + '.' + qualifier when qualifier is not None.
            # That's up to the designer of the layer type.
            desc_output_str = self.GetStringForDescriptor(desc, all_layers)
            self.descriptors[key] = (desc, desc_dim, desc_norm_str, desc_output_str)
            # the following helps to check the code by parsing it again.
            desc2 = self.ConvertToDescriptor(desc_norm_str, all_layers)
            desc_norm_str2 = desc2.str()
            # if the following ever fails we'll have to do some debugging.
            if desc_norm_str != desc_norm_str2:
                raise RuntimeError("Likely code error: '{0}' != '{1}'".format(
                        desc_norm_str, desc_norm_str2))

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


    # This function converts any config variables in self.config which
    # correspond to Descriptors, into a 'normalized form' derived from parsing
    # them as Descriptors, replacing things like [-1] with the actual layer
    # names, and regenerating them as strings.  We stored this when the
    # object was initialized, in self.descriptors; this function just copies them
    # back to the config.
    def NormalizeDescriptors(self):
        for key,tuple in self.descriptors.items():
            self.config[key] = tuple[2]  # desc_norm_str

    # This function, which is a convenience function intended to be called from
    # child classes, converts a string representing a descriptor
    # ('descriptor_string') into an object of type Descriptor, and returns it.
    # It needs 'self' and 'all_layers' (where 'all_layers' is a list of objects
    # of type XconfigLayerBase) so that it can work out a list of the names of
    # other layers, and get dimensions from them.
    def ConvertToDescriptor(self, descriptor_string, all_layers):
        prev_names = xconfig_utils.GetPrevNames(all_layers, self)
        tokens = xconfig_utils.TokenizeDescriptor(descriptor_string, prev_names)
        pos = 0
        (descriptor, pos) = xconfig_utils.ParseNewDescriptor(tokens, pos, prev_names)
        # note: 'pos' should point to the 'end of string' marker
        # that terminates 'tokens'.
        if pos != len(tokens) - 1:
            raise RuntimeError("Parsing Descriptor, saw junk at end: " +
                            ' '.join(tokens[pos:-1]))
        return descriptor

    # Returns the dimension of a Descriptor object.
    # This is a convenience function used in SetConfigs.
    def GetDimForDescriptor(self, descriptor, all_layers):
        layer_to_dim_func = lambda name: xconfig_utils.GetDimFromLayerName(all_layers, self, name)
        return descriptor.Dim(layer_to_dim_func)

    # Returns the 'final' string form of a Descriptor object, as could be used
    # in config files.
    # This is a convenience function provided for use in child classes;
    def GetStringForDescriptor(self, descriptor, all_layers):
        layer_to_string_func = lambda name: xconfig_utils.GetStringFromLayerName(all_layers, self, name)
        return descriptor.ConfigString(layer_to_string_func)

    # Name() returns the name of this layer, e.g. 'affine1'.  It does not
    # necessarily correspond to a component name.
    def Name(self):
        return self.name

    ######  Functions that might be overridden by the child class: #####

    # child classes should override this.
    def SetDefaultConfigs(self):
        raise RuntimeError("Child classes must override SetDefaultConfigs().")

    # child classes should override this.
    def CheckConfigs(self):
        pass

    # This function, which may be (but usually will not have to be) overrideden
    # by child classes, returns a list of keys/names of config variables that
    # will be interpreted as Descriptors.  It is used in the function
    # 'NormalizeDescriptors()'.  This implementation will work
    # layer types whose only Descriptor-valued config is 'input'.

    # If a child class adds more config variables that are interpreted as
    # descriptors (e.g. to read auxiliary inputs), or does not have an input
    # (e.g. the XconfigInputLayer), it should override this function's
    # implementation to something like: `return ['input', 'input2']`
    def GetDescriptorConfigs(self):
        return [ 'input' ]

    # Returns a list of all qualifiers (meaning auxiliary outputs) that this
    # layer supports.  These are either 'None' for the regular output, or a
    # string (e.g. 'projection' or 'memory_cell') for any auxiliary outputs that
    # the layer might provide.  Most layer types will not need to override this.
    def Qualifiers(self):
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
    def OutputName(self, qualifier = None):
        raise RuntimeError("Child classes must override OutputName()")

    # The dimension that this layer outputs.  The 'qualifier' parameter is for
    # layer types which support auxiliary outputs.
    def OutputDim(self, qualifier = None):
        raise RuntimeError("Child classes must override OutputDim()")

    # This function returns lines destined for the 'full' config format, as
    # would be read by the C++ programs.
    # Since the program xconfig_to_configs.py writes several config files, this
    # function returns a list of pairs of the form (config_file_basename, line),
    # e.g. something like
    # [ ('init', 'input-node name=input dim=40'),
    #   ('ref', 'input-node name=input dim=40') ]
    # which would be written to config_dir/init.config and config_dir/ref.config.
    def GetFullConfig(self):
        raise RuntimeError("Child classes must override GetFullConfig()")


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
            raise RuntimeError("Dimension of input-layer '{0}' is not set".format(self.name))

    def GetDescriptorConfigs(self):
        return []  # there is no 'input' field in self.config.

    def OutputName(self, qualifier = None):
        assert qualifier is None
        return self.name

    def OutputDim(self, qualifier = None):
        assert qualifier is None
        return self.config['dim']

    def GetFullConfig(self):
        # the input layers need to be printed in 'init.config' (which
        # initializes the neural network prior to the LDA), in 'ref.config',
        # which is a version of the config file used for getting left and right
        # context (it doesn't read anything for the LDA-like transform and/or
        # presoftmax-prior-scale components)
        # In 'full.config' we write everything, this is just for reference,
        # and also for cases where we don't use the LDA-like transform.
        ans = []
        for config_name in [ 'init', 'ref', 'all' ]:
            ans.append( (config_name,
                         'input-node name={0} dim={1}'.format(self.name,
                                                              self.config['dim'])))
        return ans



# This class is for lines like
# 'output name=output input=Append(input@-1, input@0, input@1, ReplaceIndex(ivector, t, 0))'
# This is for outputs that are not really output "layers" (there is no affine transform or
# nonlinearity), they just directly map to an output-node in nnet3.
class XconfigTrivialOutputLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == 'output'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def SetDefaultConfigs(self):
        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = { 'input':'[-1]' }

    def CheckConfigs(self):
        pass  # nothing to check; descriptor-parsing can't happen in this function.

    def OutputName(self, qualifier = None):
        assert qualifier is None
        return self.name

    def OutputDim(self, qualifier = None):
        assert qualifier is None
        # note: each value of self.descriptors is (descriptor, dim, normalized-string, output-string).
        return self.descriptors['input'][1]

    def GetFullConfig(self):
        # the input layers need to be printed in 'init.config' (which
        # initializes the neural network prior to the LDA), in 'ref.config',
        # which is a version of the config file used for getting left and right
        # context (it doesn't read anything for the LDA-like transform and/or
        # presoftmax-prior-scale components)
        # In 'full.config' we write everything, this is just for reference,
        # and also for cases where we don't use the LDA-like transform.
        ans = []

        # note: each value of self.descriptors is (descriptor, dim,
        # normalized-string, output-string).
        # by 'output-string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        descriptor_final_str = self.descriptors['input'][3]

        for config_name in [ 'ref', 'all' ]:
            ans.append( (config_name,
                         'output-node name={0} input={1}'.format(
                        self.name, descriptor_final_str)))
        return ans


# This class is for lines like
#  'output-layer name=output dim=4257 input=Append(input@-1, input@0, input@1, ReplaceIndex(ivector, t, 0))'
# By default this includes a log-softmax component.  The parameters are initialized to zero, as
# this is best for output layers.
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   dim=-1                   [Output dimension of layer, will normally equal the number of pdfs.]
#   include-log-softmax=true [setting it to false will omit the log-softmax component- useful for chain
#                              models.]
#   objective-type=linear    [the only other choice currently is 'quadratic', for use in regression
#                             problems]

#   learning-rate-factor=1.0 [Learning rate factor for the final affine component, multiplies the
#                              standard learning rate. normally you'll leave this as-is, but for
#                              xent regularization output layers for chain models you'll want to set
#                              learning-rate-factor=(0.5/xent_regularize), normally
#                              learning-rate-factor=5.0 since xent_regularize is normally 0.1.
#   presoftmax-scale-file=''  [If set, a filename for a vector that will be used to scale the output
#                              of the affine component before the log-softmax (if
#                              include-log-softmax=true), or before the output (if not).  This is
#                              helpful to avoid instability in training due to some classes having
#                              much more data than others.  The way we normally create this vector
#                              is to take the priors of the classes to the power -0.25 and rescale
#                              them so the average is 1.0.  This factor -0.25 is referred to
#                              as presoftmax_prior_scale_power in scripts.]
#                              In the scripts this would normally be set to config_dir/presoftmax_prior_scale.vec
class XconfigOutputLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == 'output-layer'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def SetDefaultConfigs(self):
        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = { 'input':'[-1]', 'dim':-1, 'include-log-softmax':True,
                        'objective-type':'linear', 'learning-rate-factor':1.0,
                        'include-log-softmax':True, 'presoftmax-scale-file':'' }

    def CheckConfigs(self):
        if self.config['dim'] <= 0:
            raise RuntimeError("In output-layer, dim has invalid value {0}".format(self.config['dim']))
        if self.config['objective-type'] != 'linear' and self.config['objective_type'] != 'quadratic':
            raise RuntimeError("In output-layer, objective-type has invalid value {0}".format(
                    self.config['objective-type']))
        if self.config['learning-rate-factor'] <= 0.0:
            raise RuntimeError("In output-layer, learning-rate-factor has invalid value {0}".format(
                    self.config['learning-rate-factor']))

        pass  # nothing to check; descriptor-parsing can't happen in this function.


    # you cannot access the output of this layer from other layers... see
    # comment in OutputName for the reason why.
    def Qualifiers(self):
        return []

    def OutputName(self, qualifier = None):
        # Note: nodes of type output-node in nnet3 may not be accessed in Descriptors,
        # so calling this with qualifier=None doesn't make sense.  But it might make
        # sense to make the output of the softmax layer and/or the output of the
        # affine layer available as inputs to other layers, in some circumstances.
        # we'll implement that when it's needed.
        raise RuntimeError("Outputs of output-layer may not be used by other layers")

    def OutputDim(self, qualifier = None):
        # see comment in OutputName().
        raise RuntimeError("Outputs of output-layer may not be used by other layers")

    def GetFullConfig(self):
        # the input layers need to be printed in 'init.config' (which
        # initializes the neural network prior to the LDA), in 'ref.config',
        # which is a version of the config file used for getting left and right
        # context (it doesn't read anything for the LDA-like transform and/or
        # presoftmax-prior-scale components)
        # In 'full.config' we write everything, this is just for reference,
        # and also for cases where we don't use the LDA-like transform.
        ans = []

        # note: each value of self.descriptors is (descriptor, dim,
        # normalized-string, output-string).
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        descriptor_final_string = self.descriptors['input'][3]
        input_dim = self.descriptors['input'][1]
        output_dim = self.config['dim']
        objective_type = self.config['objective-type']
        learning_rate_factor = self.config['learning-rate-factor']
        include_log_softmax = self.config['include-log-softmax']
        presoftmax_scale_file = self.config['presoftmax-scale-file']

        for config_name in [ 'ref', 'all' ]:
            # First the affine node.
            line = ('component name={0}.affine type=NaturalGradientAffineComponent input-dim={1} '
                    'output-dim={2} param-stddev=0 bias-stddev=0 '.format(
                    self.name, input_dim, output_dim) +
                    ('learning-rate-factor={0} '.format(learning_rate_factor)
                     if learning_rate_factor != 1.0 else ''))
            ans.append((config_name, line))
            line = ('component-node name={0}.affine component={0}.affine input={1}'.format(
                    self.name, descriptor_final_string))
            ans.append((config_name, line))
            cur_node = '{0}.affine'.format(descriptor_final_string)
            if presoftmax_scale_file != '' and config_name == 'all':
                # don't use the presoftmax-scale in 'ref.config' since that file won't exist at the
                # time we evaluate it.  (ref.config is used to find the left/right context).
                line = ('component name={0}.fixed-scale type=FixedScaleComponent scales={1}'.format(
                        self.name, presoftmax_scale_file))
                ans.append((config_name, line))
                line = ('component-node name={0}.fixed-scale component={0}.fixed-scale input={1}'.format(
                        self.name, cur_node))
                ans.append((config_name, line))
                cur_node = '{0}.fixed-scale'.format(self.name)
            if include_log_softmax:
                line = ('component name={0}.log-softmax type=LogSoftmaxComponent dim={1}'.format(
                        self.name, output_dim))
                ans.append((config_name, line))
                line = ('component-node name={0}.log-softmax component={0}.log-softmax input={1}'.format(
                        self.name, cur_node))
                ans.append((config_name, line))
                cur_node = '{0}.log-softmax'.format(self.name)
            line = ('output-node name={0} input={0}.log-softmax'.format(self.name, cur_node))
            ans.append((config_name, line))
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
    elif first_token == 'output':
        return XconfigTrivialOutputLayer(first_token, key_to_value, prev_names)
    elif first_token == 'output-layer':
        return XconfigOutputLayer(first_token, key_to_value, prev_names)
    else:
        raise RuntimeError("Error parsing xconfig line (no such layer type): " +
                        first_token + ' ' +
                        ' '.join(['{0} {1}'.format(x,y) for x,y in key_to_value.items()]))


# Uses ParseConfigLine() to turn a config line that has been parsed into
# a first token e.g. 'affine-layer' and a key->value map like { 'dim':'1024', 'name':'affine1' },
# and then turns this into an object representing that line of the config file.
# 'prev_names' is a list of the names of preceding lines of the
# config file.
def ConfigLineToObject(config_line, prev_names = None):
    (first_token, key_to_value) = xconfig_utils.ParseConfigLine(config_line)
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
    all_layers = []
    while True:
        line = f.readline()
        if line == '':
            break
        x = xconfig_utils.ParseConfigLine(line)
        if x is None:
            continue   # line was blank or only comments.
        (first_token, key_to_value) = x
        # the next call will raise an easy-to-understand exception if
        # it fails.
        this_layer = ParsedLineToXconfigLayer(first_token,
                                              key_to_value,
                                              all_layers)
        all_layers.append(this_layer)
    if len(all_layers) == 0:
        raise RuntimeError("{0}: xconfig file '{1}' is empty".format(
            sys.argv[0], xconfig_filename))
    f.close()
    return all_layers


def TestLayers():
    # for some config lines that should be printed the same way as they
    # are read, check that this is the case.
    for x in [ 'input name=input dim=30' ]:
        assert str(ConfigLineToObject(x, [])) == x
