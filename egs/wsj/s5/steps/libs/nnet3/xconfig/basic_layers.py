# Copyright 2016    Johns Hopkins University (Dan Povey)
#           2016    Vijayaditya Peddinti
#           2017    Google Inc. (vpeddinti@google.com)
#           2017    Vimal Manohar
# Apache 2.0.

""" This module contains the parent class from which all layers are inherited
and some basic layer definitions.
"""

from __future__ import print_function
import math
import re
import sys
import libs.nnet3.xconfig.utils as xutils
import libs.common as common_lib


class XconfigLayerBase(object):
    """ A base-class for classes representing layers of xconfig files.
    """

    def __init__(self, first_token, key_to_value, all_layers):
        """
         first_token: first token on the xconfig line, e.g. 'affine-layer'.f
         key_to_value: dictionary with parameter values
             { 'name':'affine1',
               'input':'Append(0, 1, 2, ReplaceIndex(ivector, t, 0))',
               'dim=1024' }.
             The only required and 'special' values that are dealt with directly
             at this level, are 'name' and 'input'. The rest are put in
             self.config and are dealt with by the child classes' init functions.
         all_layers: An array of objects inheriting XconfigLayerBase for all
                    previously parsed layers.
        """

        self.layer_type = first_token
        if not 'name' in key_to_value:
            raise RuntimeError("Expected 'name' to be specified.")
        self.name = key_to_value['name']
        if not xutils.is_valid_line_name(self.name):
            raise RuntimeError("Invalid value: name={0}".format(
                key_to_value['name']))
        for prev_layer in all_layers:
            if self.name == prev_layer.name:
                raise RuntimeError("Name '{0}' is used for more than one "
                                    "layer.".format(self.name))

        # the following, which should be overridden in the child class, sets
        # default config parameters in self.config.
        self.set_default_configs()
        # The following is not to be reimplemented in child classes;
        # it sets the config values to those specified by the user, and
        # parses any Descriptors.
        self.set_configs(key_to_value, all_layers)
        # This method, sets the derived default config values
        # i.e., config values when not specified can be derived from
        # other values. It can be overridden in the child class.
        self.set_derived_configs()
        # the following, which should be overridden in the child class, checks
        # that the config parameters that have been set are reasonable.
        self.check_configs()


    def set_configs(self, key_to_value, all_layers):
        """ Sets the config variables.
            We broke this code out of __init__ for clarity.
            the child-class constructor will deal with the configuration values
            in a more specific way.
        """

        # First check that there are no keys that don't correspond to any config
        # parameter of this layer, and if so, raise an exception with an
        # informative message saying what configs are allowed.
        for key,value in key_to_value.items():
            if key != 'name':
                if not key in self.config:
                    configs = ' '.join([ ('{0}->"{1}"'.format(x,y) if isinstance(y, str)
                                          else '{0}->{1}'.format(x,y))
                                         for x,y in self.config.items() ])
                    raise RuntimeError("Configuration value {0}={1} was not "
                                        "expected in layer of type {2}; allowed "
                                        "configs with their defaults: {3}"
                                        .format(key, value, self.layer_type, configs))

        for key,value in key_to_value.items():
            if key != 'name':
                assert key in self.config  # we checked above.
                self.config[key] = xutils.convert_value_to_type(key,
                                                                type(self.config[key]),
                                                                value)
        self.descriptors = dict()
        self.descriptor_dims = dict()
        # Parse Descriptors and get their dims and their 'final' string form.
        # in self.descriptors[key]
        for key in self.get_input_descriptor_names():
            if not key in self.config:
                raise RuntimeError("{0}: object of type {1} needs to override"
                                   " get_input_descriptor_names()."
                                   "".format(sys.argv[0], str(type(self))))

            descriptor_string = self.config[key]  # input string.
            assert isinstance(descriptor_string, str)
            desc = self.convert_to_descriptor(descriptor_string, all_layers)
            desc_dim = self.get_dim_for_descriptor(desc, all_layers)
            desc_norm_str = desc.str()

            # desc_output_str contains the "final" component names, those that
            # appear in the actual config file (i.e. not names like
            # 'layer.auxiliary_output'); that's how it differs from desc_norm_str.
            # Note: it's possible that the two strings might be the same in
            # many, even most, cases-- it depends whether
            # output_name(self, auxiliary_output)
            # returns self.get_name() + '.' + auxiliary_output
            # when auxiliary_output is not None.
            # That's up to the designer of the layer type.
            desc_output_str = self.get_string_for_descriptor(desc, all_layers)
            self.descriptors[key] = {'string':desc,
                                     'normalized-string':desc_norm_str,
                                     'final-string':desc_output_str,
                                     'dim':desc_dim}

            # the following helps to check the code by parsing it again.
            desc2 = self.convert_to_descriptor(desc_norm_str, all_layers)
            desc_norm_str2 = desc2.str()
            # if the following ever fails we'll have to do some debugging.
            if desc_norm_str != desc_norm_str2:
                raise RuntimeError("Likely code error: '{0}' != '{1}'"
                                    "".format(desc_norm_str, desc_norm_str2))

    def str(self):
        """Converts 'this' to a string which could be printed to
        an xconfig file; in xconfig_to_configs.py we actually expand all the
        lines to strings and write it as xconfig.expanded as a reference
        (so users can see any defaults).
        """

        list_of_entries = [ '{0} name={1}'.format(self.layer_type, self.name) ]
        for key, value in sorted(self.config.items()):
            if isinstance(value, str) and re.search('=', value):
                # the value is a string that contains an '=' sign, so we need to
                # enclose it in double-quotes, otherwise we woudldn't be able to
                # parse from that output.
                if re.search('"', value):
                    print("Warning: config '{0}={1}' contains both double-quotes "
                          "and equals sign; it will not be possible to parse it "
                          "from the file.".format(key, value), file=sys.stderr)
                list_of_entries.append('{0}="{1}"'.format(key, value))
            else:
                list_of_entries.append('{0}={1}'.format(key, value))

        return ' '.join(list_of_entries)


    def __str__(self):
        return self.str()


    def normalize_descriptors(self):
        """Converts any config variables in self.config which correspond to
        Descriptors, into a 'normalized form' derived from parsing them as
        Descriptors, replacing things like [-1] with the actual layer names,
        and regenerating them as strings.  We stored this when the object was
        initialized, in self.descriptors; this function just copies them back
        to the config.
        """

        for key, desc_str_dict in self.descriptors.items():
            self.config[key] = desc_str_dict['normalized-string']

    def convert_to_descriptor(self, descriptor_string, all_layers):
        """Convenience function intended to be called from child classes,
        converts a string representing a descriptor ('descriptor_string')
        into an object of type Descriptor, and returns it. It needs 'self' and
        'all_layers' (where 'all_layers' is a list of objects of type
        XconfigLayerBase) so that it can work out a list of the names of other
        layers, and get dimensions from them.
        """

        prev_names = xutils.get_prev_names(all_layers, self)
        tokens = xutils.tokenize_descriptor(descriptor_string, prev_names)
        pos = 0
        (descriptor, pos) = xutils.parse_new_descriptor(tokens, pos, prev_names)
        # note: 'pos' should point to the 'end of string' marker
        # that terminates 'tokens'.
        if pos != len(tokens) - 1:
            raise RuntimeError("Parsing Descriptor, saw junk at end: " +
                            ' '.join(tokens[pos:-1]))
        return descriptor

    def get_dim_for_descriptor(self, descriptor, all_layers):
        """Returns the dimension of a Descriptor object. This is a convenience
        function used in set_configs.
        """

        layer_to_dim_func = \
                lambda name: xutils.get_dim_from_layer_name(all_layers, self,
                                                            name)
        return descriptor.dim(layer_to_dim_func)

    def get_string_for_descriptor(self, descriptor, all_layers):
        """Returns the 'final' string form of a Descriptor object,
        as could be used in config files. This is a convenience function
        provided for use in child classes;
        """

        layer_to_string_func = \
                lambda name: xutils.get_string_from_layer_name(all_layers,
                                                               self, name)
        return descriptor.config_string(layer_to_string_func)

    def get_name(self):
        """Returns the name of this layer, e.g. 'affine1'.  It does not
        necessarily correspond to a component name.
        """

        return self.name

    ######  Functions that might be overridden by the child class: #####

    def set_default_configs(self):
        """Child classes should override this.
        """

        raise Exception("Child classes must override set_default_configs().")

    def set_derived_configs(self):
        """This is expected to be called after set_configs and before
        check_configs().
        """

        if self.config['dim'] <= 0:
            self.config['dim'] = self.descriptors['input']['dim']

    def check_configs(self):
        """child classes should override this.
        """

        pass

    def get_input_descriptor_names(self):
        """This function, which may be (but usually will not have to be)
        overridden by child classes, returns a list of names of the input
        descriptors expected by this component. Typically this would just
        return ['input'] as most layers just have one 'input'. However some
        layers might require more inputs (e.g. cell state of previous LSTM layer
        in Highway LSTMs). It is used in the function 'normalize_descriptors()'.
        This implementation will work for layer types whose only
        Descriptor-valued config is 'input'.
        If a child class adds more inputs, or does not have an input
        (e.g. the XconfigInputLayer), it should override this function's
        implementation to something like: `return ['input', 'input2']`
        """

        return [ 'input' ]

    def auxiliary_outputs(self):
        """Returns a list of all auxiliary outputs that this layer supports.
        These are either 'None' for the regular output, or a string
        (e.g. 'projection' or 'memory_cell') for any auxiliary outputs that
        the layer might provide.  Most layer types will not need to override
        this.
        """

        return [ None ]

    def output_name(self, auxiliary_output = None):
        """Called with auxiliary_output == None, this returns the component-node
        name of the principal output of the layer (or if you prefer, the text
        form of a descriptor that gives you such an output; such as
        Append(some_node, some_other_node)).
        The 'auxiliary_output' argument is a text value that is designed for
        extensions to layers that have additional auxiliary outputs.
        For example, to implement a highway LSTM you need the memory-cell of a
        layer, so you might allow auxiliary_output='memory_cell' for such a
        layer type, and it would return the component node or a suitable
        Descriptor: something like 'lstm3.c_t'
        """

        raise Exception("Child classes must override output_name()")

    def output_dim(self, auxiliary_output = None):
        """The dimension that this layer outputs.  The 'auxiliary_output'
        parameter is for layer types which support auxiliary outputs.
        """

        raise Exception("Child classes must override output_dim()")

    def get_full_config(self):
        """This function returns lines destined for the 'full' config format, as
        would be read by the C++ programs. Since the program
        xconfig_to_configs.py writes several config files, this function returns
        a list of pairs of the form (config_file_basename, line),
        e.g. something like
         [  ('init', 'input-node name=input dim=40'),
            ('ref', 'input-node name=input dim=40') ]
        which would be written to config_dir/init.config and config_dir/ref.config.
        """

        raise Exception("Child classes must override get_full_config()")


class XconfigInputLayer(XconfigLayerBase):
    """This class is for lines like
    'input name=input dim=40'
    or
    'input name=ivector dim=100'
    in the config file.
    """


    def __init__(self, first_token, key_to_value, prev_names = None):

        assert first_token == 'input'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)


    def set_default_configs(self):

        self.config = { 'dim': -1}

    def check_configs(self):

        if self.config['dim'] <= 0:
            raise RuntimeError("Dimension of input-layer '{0}'"
                                "should be positive.".format(self.name))

    def get_input_descriptor_names(self):

        return []  # there is no 'input' field in self.config.

    def output_name(self, auxiliary_outputs = None):

        # there are no auxiliary outputs as this layer will just pass the input
        assert auxiliary_outputs is None
        return self.name

    def output_dim(self, auxiliary_outputs = None):

        # there are no auxiliary outputs as this layer will just pass the input
        assert auxiliary_outputs is None
        return self.config['dim']

    def get_full_config(self):

        # unlike other layers the input layers need to be printed in
        # 'init.config' (which initializes the neural network prior to the LDA)
        ans = []
        for config_name in [ 'init', 'ref', 'final' ]:
            ans.append( (config_name,
                         'input-node name={0} dim={1}'.format(self.name,
                                                              self.config['dim'])))
        return ans



class XconfigTrivialOutputLayer(XconfigLayerBase):
    """This class is for lines like
    'output name=output input=Append(input@-1, input@0, input@1, ReplaceIndex(ivector, t, 0))'
    This is for outputs that are not really output "layers"
    (there is no affine transform or nonlinearity), they just directly map to an
    output-node in nnet3.
    """

    def __init__(self, first_token, key_to_value, prev_names = None):

        assert first_token == 'output'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):

        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = { 'input':'[-1]', 'dim':-1 }

    def check_configs(self):

        pass  # nothing to check; descriptor-parsing can't happen in this function.

    def output_name(self, auxiliary_outputs = None):

        # there are no auxiliary outputs as this layer will just pass the output
        # of the previous layer
        assert auxiliary_outputs is None
        return self.name

    def output_dim(self, auxiliary_outputs = None):

        assert auxiliary_outputs is None
        # note: each value of self.descriptors is (descriptor, dim, normalized-string, output-string).
        return self.descriptors['input']['dim']

    def get_full_config(self):

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
        descriptor_final_str = self.descriptors['input']['final-string']

        for config_name in ['init', 'ref', 'final' ]:
            ans.append( (config_name,
                         'output-node name={0} input={1}'.format(
                        self.name, descriptor_final_str)))
        return ans


class XconfigOutputLayer(XconfigLayerBase):
    """This class is for lines like
    'output-layer name=output dim=4257 input=Append(input@-1, input@0, input@1, ReplaceIndex(ivector, t, 0))'
    By default this includes a log-softmax component.  The parameters are
    initialized to zero, asthis is best for output layers.

    Parameters of the class, and their defaults:
        input='[-1]'    :   Descriptor giving the input of the layer.
        dim=None    :   Output dimension of layer, will normally equal the number of pdfs.
        include-log-softmax=true    :   setting it to false will omit the
            log-softmax component- useful for chain models.
        objective-type=linear   :   the only other choice currently is
            'quadratic', for use in regression problems
        learning-rate-factor=1.0    :   Learning rate factor for the final
            affine component, multiplies the standard learning rate. normally
            you'll leave this as-is, but for xent regularization output layers
            for chain models you'll want to set
            learning-rate-factor=(0.5/xent_regularize),
            normally learning-rate-factor=5.0 since xent_regularize is
            normally 0.1.
        presoftmax-scale-file=None  :   If set, a filename for a vector that
            will be used to scale the output of the affine component before the
            log-softmax (if include-log-softmax=true), or before the output
            (if not).  This is helpful to avoid instability in training due to
            some classes having much more data than others.  The way we normally
            create this vector is to take the priors of the classes to the
            power -0.25 and rescale them so the average is 1.0.  This factor
            -0.25 is referred to as presoftmax_prior_scale_power in scripts. In
            the scripts this would normally be set to
            config_dir/presoftmax_prior_scale.vec
        output-delay=0    :  Can be used to shift the frames on the output, equivalent
             to delaying labels by this many frames (positive value increases latency
             in online decoding but may help if you're using unidirectional LSTMs.
        ng-affine-options=''  :   Can be used supply non-default options to the affine
             layer (intended for the natural gradient but can be an arbitrary string
             to be added to the config line.  e.g. 'update-period=2'.).
    """

    def __init__(self, first_token, key_to_value, prev_names = None):

        assert first_token == 'output-layer'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):

        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = {'input' : '[-1]',
                       'dim' : -1,
                       'include-log-softmax' : True,
                            # this would be false for chain models
                       'objective-type' : 'linear',
                            # see Nnet::ProcessOutputNodeConfigLine in
                            # nnet-nnet.cc for other options
                       'learning-rate-factor' : 1.0,
                       'presoftmax-scale-file' : '',
                            # used in DNN (not RNN) training when using
                            # frame-level objfns,
                       'max-change' : 1.5,
                       'param-stddev' : 0.0,
                       'bias-stddev' : 0.0,
                       'output-delay' : 0,
                       'ng-affine-options' : ''
                      }

    def check_configs(self):

        if self.config['dim'] <= -1:
            raise RuntimeError("In output-layer, dim has invalid value {0}"
                                "".format(self.config['dim']))

        if self.config['objective-type'] != 'linear' and \
                self.config['objective_type'] != 'quadratic':
            raise RuntimeError("In output-layer, objective-type has"
                                " invalid value {0}"
                                "".format(self.config['objective-type']))

        if self.config['learning-rate-factor'] <= 0.0:
            raise RuntimeError("In output-layer, learning-rate-factor has"
                                " invalid value {0}"
                                "".format(self.config['learning-rate-factor']))


    # you cannot access the output of this layer from other layers... see
    # comment in output_name for the reason why.
    def auxiliary_outputs(self):

        return []

    def output_name(self, auxiliary_outputs = None):

        # Note: nodes of type output-node in nnet3 may not be accessed in
        # Descriptors, so calling this with auxiliary_outputs=None doesn't
        # make sense.  But it might make sense to make the output of the softmax
        # layer and/or the output of the affine layer available as inputs to
        # other layers, in some circumstances.
        # we'll implement that when it's needed.
        raise RuntimeError("Outputs of output-layer may not be used by other"
                            " layers")

    def output_dim(self, auxiliary_output = None):

        # see comment in output_name().
        raise RuntimeError("Outputs of output-layer may not be used by other"
                            " layers")

    def get_full_config(self):

        ans = []

        # note: each value of self.descriptors is (descriptor, dim,
        # normalized-string, output-string).
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        descriptor_final_string = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        output_dim = self.config['dim']
        objective_type = self.config['objective-type']
        learning_rate_factor = self.config['learning-rate-factor']
        include_log_softmax = self.config['include-log-softmax']
        presoftmax_scale_file = self.config['presoftmax-scale-file']
        param_stddev = self.config['param-stddev']
        bias_stddev = self.config['bias-stddev']
        output_delay = self.config['output-delay']
        max_change = self.config['max-change']
        ng_affine_options = self.config['ng-affine-options']

        # note: ref.config is used only for getting the left-context and
        # right-context of the network;
        # final.config is where we put the actual network definition.
        for config_name in [ 'ref', 'final' ]:
            # First the affine node.
            line = ('component name={0}.affine'
                    ' type=NaturalGradientAffineComponent'
                    ' input-dim={1}'
                    ' output-dim={2}'
                    ' param-stddev={3}'
                    ' bias-stddev={4}'
                    ' max-change={5} {6} '
                    ''.format(self.name, input_dim, output_dim,
                              param_stddev, bias_stddev, max_change, ng_affine_options) +
                    ('learning-rate-factor={0} '.format(learning_rate_factor)
                     if learning_rate_factor != 1.0 else ''))
            ans.append((config_name, line))

            line = ('component-node name={0}.affine'
                    ' component={0}.affine input={1}'
                    ''.format(self.name, descriptor_final_string))
            ans.append((config_name, line))
            cur_node = '{0}.affine'.format(self.name)

            if presoftmax_scale_file is not '' and config_name == 'final':
                # don't use the presoftmax-scale in 'ref.config' since that
                # file won't exist at the time we evaluate it.
                # (ref.config is used to find the left/right context).
                line = ('component name={0}.fixed-scale'
                        ' type=FixedScaleComponent scales={1}'
                        ''.format(self.name, presoftmax_scale_file))
                ans.append((config_name, line))

                line = ('component-node name={0}.fixed-scale'
                        ' component={0}.fixed-scale input={1}'
                        ''.format(self.name, cur_node))
                ans.append((config_name, line))
                cur_node = '{0}.fixed-scale'.format(self.name)

            if include_log_softmax:
                line = ('component name={0}.log-softmax'
                        ' type=LogSoftmaxComponent dim={1}'
                        ''.format(self.name, output_dim))
                ans.append((config_name, line))

                line = ('component-node name={0}.log-softmax'
                        ' component={0}.log-softmax input={1}'
                        ''.format(self.name, cur_node))
                ans.append((config_name, line))
                cur_node = '{0}.log-softmax'.format(self.name)

            if output_delay != 0:
                cur_node = 'Offset({0}, {1})'.format(cur_node, output_delay)

            line = ('output-node name={0} input={1}'.format(self.name, cur_node))
            ans.append((config_name, line))
        return ans


# This class is for parsing lines like
#  'relu-renorm-layer name=layer1 dim=1024 input=Append(-3,0,3)'
# or:
#  'sigmoid-layer name=layer1 dim=1024 input=Append(-3,0,3)'
# which specify addition of an affine component and a sequence of non-linearities.
# Here, the name of the layer itself dictates the sequence of nonlinearities
# that are applied after the affine component; the name should contain some
# combination of 'relu', 'renorm', 'sigmoid' and 'tanh',
# and these nonlinearities will be added along with the affine component.
#
# The dimension specified is the output dim; the input dim is worked out from the input descriptor.
# This class supports only nonlinearity types that do not change the dimension; we can create
# another layer type to enable the use p-norm and similar dimension-reducing nonlinearities.
#
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   dim=None                   [Output dimension of layer, e.g. 1024]
#   self-repair-scale=1.0e-05  [Affects relu, sigmoid and tanh layers.]
#   learning-rate-factor=1.0   [This can be used to make the affine component
#                               train faster or slower].
#
class XconfigBasicLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        # Here we just list some likely combinations.. you can just add any
        # combinations you want to use, to this list.
        assert first_token in [ 'relu-layer', 'relu-renorm-layer', 'sigmoid-layer',
                                'tanh-layer', 'relu-batchnorm-layer', 'relu-dropout-layer',
                                'relu-batchnorm-dropout-layer' ]
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):

        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = { 'input':'[-1]',
                        'dim':-1,
                        'max-change' : 0.75,
                        'self-repair-scale' : 1.0e-05,
                        'target-rms' : 1.0,
                        'learning-rate-factor' : 1.0,
                        'ng-affine-options' : '',
                        'dropout-proportion': 0.5}  # dropout-proportion only
                                                    # affects layers with
                                                    # 'dropout' in the name.


    def check_configs(self):
        if self.config['dim'] < 0:
            raise RuntimeError("dim has invalid value {0}".format(self.config['dim']))
        if self.config['self-repair-scale'] < 0.0 or self.config['self-repair-scale'] > 1.0:
            raise RuntimeError("self-repair-scale has invalid value {0}"
                               .format(self.config['self-repair-scale']))
        if self.config['target-rms'] < 0.0:
            raise RuntimeError("target-rms has invalid value {0}"
                               .format(self.config['target-rms']))
        if self.config['learning-rate-factor'] <= 0.0:
            raise RuntimeError("learning-rate-factor has invalid value {0}"
                               .format(self.config['learning-rate-factor']))

    def output_name(self, auxiliary_output=None):
        # at a later stage we might want to expose even the pre-nonlinearity
        # vectors
        assert auxiliary_output == None

        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        last_nonlinearity = split_layer_name[-2]
        # return something like: layer3.renorm
        return '{0}.{1}'.format(self.name, last_nonlinearity)

    def output_dim(self, auxiliary_output = None):
        output_dim = self.config['dim']
        # If not set, the output-dim defaults to the input-dim.
        if output_dim <= 0:
            output_dim = self.descriptors['input']['dim']
        return output_dim


    def get_full_config(self):
        ans = []
        config_lines = self._generate_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in this layer
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans


    def _generate_config(self):
        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        nonlinearities = split_layer_name[:-1]

        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']

        # the child classes e.g. tdnn might want to process the input
        # before adding the other components

        return self._add_components(input_desc, input_dim, nonlinearities)

    def _add_components(self, input_desc, input_dim, nonlinearities):
        output_dim = self.output_dim()
        self_repair_scale = self.config['self-repair-scale']
        target_rms = self.config['target-rms']
        max_change = self.config['max-change']
        ng_affine_options = self.config['ng-affine-options']
        learning_rate_factor=self.config['learning-rate-factor']
        learning_rate_option=('learning-rate-factor={0}'.format(learning_rate_factor)
                              if learning_rate_factor != 1.0 else '')

        configs = []
        # First the affine node.
        line = ('component name={0}.affine'
                ' type=NaturalGradientAffineComponent'
                ' input-dim={1}'
                ' output-dim={2}'
                ' max-change={3}'
                ' {4} {5} '
                ''.format(self.name, input_dim, output_dim,
                          max_change, ng_affine_options,
                          learning_rate_option))
        configs.append(line)

        line = ('component-node name={0}.affine'
                ' component={0}.affine input={1}'
                ''.format(self.name, input_desc))
        configs.append(line)
        cur_node = '{0}.affine'.format(self.name)

        for nonlinearity in nonlinearities:
            if nonlinearity == 'relu':
                line = ('component name={0}.{1}'
                        ' type=RectifiedLinearComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                            self_repair_scale))

            elif nonlinearity == 'sigmoid':
                line = ('component name={0}.{1}'
                        ' type=SigmoidComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                            self_repair_scale))

            elif nonlinearity == 'tanh':
                line = ('component name={0}.{1}'
                        ' type=TanhComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                            self_repair_scale))

            elif nonlinearity == 'renorm':
                line = ('component name={0}.{1}'
                        ' type=NormalizeComponent dim={2}'
                        ' target-rms={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                            target_rms))

            elif nonlinearity == 'batchnorm':
                line = ('component name={0}.{1}'
                        ' type=BatchNormComponent dim={2}'
                        ' target-rms={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                            target_rms))

            elif nonlinearity == 'dropout':
                line = ('component name={0}.{1} type=DropoutComponent '
                           'dim={2} dropout-proportion={3}'.format(
                               self.name, nonlinearity, output_dim,
                               self.config['dropout-proportion']))

            else:
                raise RuntimeError("Unknown nonlinearity type: {0}"
                                   .format(nonlinearity))

            configs.append(line)
            line = ('component-node name={0}.{1}'
                    ' component={0}.{1} input={2}'
                    ''.format(self.name, nonlinearity, cur_node))

            configs.append(line)
            cur_node = '{0}.{1}'.format(self.name, nonlinearity)
        return configs


# This class is for lines like
#  'fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=foo/bar/lda.mat'
#
# The output dimension of the layer may be specified via 'dim=xxx', but if not specified,
# the dimension defaults to the same as the input.  Note: we don't attempt to read that
# file at the time the config is created, because in the recipes, that file is created
# after the config files.
#
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   dim=None                   [Output dimension of layer; defaults to the same as the input dim.]
#   affine-transform-file='' [Must be specified.]
#   delay=0                  [Optional delay for the output-node in init.config]
class XconfigFixedAffineLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == 'fixed-affine-layer'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = { 'input':'[-1]',
                        'dim':-1,
                        'affine-transform-file':'',
                        'delay':0,
                        'write-init-config':True}

    def check_configs(self):
        if self.config['affine-transform-file'] is None:
            raise RuntimeError("affine-transform-file must be set.")

    def output_name(self, auxiliary_output = None):
        # Fixed affine layer computes only one vector, there are no intermediate
        # vectors.
        assert auxiliary_output == None
        return self.name

    def output_dim(self, auxiliary_output = None):
        output_dim = self.config['dim']
        # If not set, the output-dim defaults to the input-dim.
        if output_dim <= 0:
            output_dim = self.descriptors['input']['dim']
        return output_dim

    def get_full_config(self):
        ans = []

        # note: each value of self.descriptors is (descriptor, dim,
        # normalized-string, output-string).
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        descriptor_final_string = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        output_dim = self.output_dim()
        transform_file = self.config['affine-transform-file']

        if self.config['write-init-config']:
            if self.config['delay'] != 0:
                line = 'component name={0}.delayed type=NoOpComponent dim={1}'.format(self.name, input_dim)
                ans.append(('init', line))
                line = 'component-node name={0}.delayed component={0}.delayed input={1}'.format(self.name, descriptor_final_string)
                ans.append(('init', line))
                line = 'output-node name=output input=Offset({0}.delayed, {1})'.format(self.name, self.config['delay'])
                ans.append(('init', line))
            else:
                # to init.config we write an output-node with the name 'output' and
                # with a Descriptor equal to the descriptor that's the input to this
                # layer.  This will be used to accumulate stats to learn the LDA transform.
                line = 'output-node name=output input={0}'.format(descriptor_final_string)
                ans.append(('init', line))

        # write the 'real' component to final.config
        line = 'component name={0} type=FixedAffineComponent matrix={1}'.format(
            self.name, transform_file)
        ans.append(('final', line))
        # write a random version of the component, with the same dims, to ref.config
        line = 'component name={0} type=FixedAffineComponent input-dim={1} output-dim={2}'.format(
            self.name, input_dim, output_dim)
        ans.append(('ref', line))
        # the component-node gets written to final.config and ref.config.
        line = 'component-node name={0} component={0} input={1}'.format(
            self.name, descriptor_final_string)
        ans.append(('final', line))
        ans.append(('ref', line))
        return ans

# This class is for lines like
#  'affine-layer name=affine input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0))'
#
# The output dimension of the layer may be specified via 'dim=xxx', but if not specified,
# the dimension defaults to the same as the input.  Note: we don't attempt to read that
# file at the time the config is created, because in the recipes, that file is created
# after the config files.
#
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   dim=None                   [Output dimension of layer; defaults to the same as the input dim.]
#
class XconfigAffineLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == 'affine-layer'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        # use None for optional parameters as we want to default to the C++ defaults
        # C++ component provides more options but I will just expose these for now
        # Note : The type of the parameter is determined based on the value assigned
        #        so please use decimal point if your parameter is a float
        self.config = { 'input' : '[-1]',
                        'dim' : -1,
                        'param-stddev' : -1.0, # this has to be initialized to 1/sqrt(input_dim)
                        'bias-stddev' : 1.0,
                        'bias-mean' : 0.0,
                        'max-change' : 0.75,
                        'learning-rate-factor' : 1.0,
                        'ng-affine-options' : ''}

    def set_derived_configs(self):
        super(XconfigAffineLayer, self).set_derived_configs()
        if self.config['param-stddev'] < 0:
            self.config['param-stddev'] = 1.0 / math.sqrt(self.descriptors['input']['dim'])

    def check_configs(self):
        if self.config['dim'] <= 0:
            raise RuntimeError("dim specified is invalid")

    def output_name(self, auxiliary_output = None):
        # affine layer computes only one vector, there are no intermediate
        # vectors.
        assert auxiliary_output == None
        return self.name

    def output_dim(self, auxiliary_output = None):
        output_dim = self.config['dim']
        # If not set, the output-dim defaults to the input-dim.
        if output_dim <= 0:
            output_dim = self.descriptors['input']['dim']

        return output_dim

    def get_full_config(self):
        ans = []

        # note: each value of self.descriptors is (descriptor, dim,
        # normalized-string, output-string).
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        descriptor_final_string = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        output_dim = self.output_dim()

        option_string=''
        for key in ['param-stddev', 'bias-stddev', 'bias-mean', 'max-change']:
            option_string += ' {0}={1}'.format(key, self.config[key])
        option_string += self.config['ng-affine-options']

        conf_lines = []
        # write the 'real' component to final.config
        conf_lines.append('component name={n} type=NaturalGradientAffineComponent '
                          'input-dim={i} output-dim={o} {opts}'.format(n = self.name,
                                                                       i = input_dim,
                                                                       o = output_dim,
                                                                       opts = option_string))
        # the component-node gets written to final.config and ref.config.
        conf_lines.append('component-node name={0} component={0} input={1}'.format(self.name,
                                                                                   descriptor_final_string))

        # the config is same for both final and ref configs
        for conf_name in ['final', 'ref']:
            for line in conf_lines:
                ans.append((conf_name, line))
        return ans


class XconfigIdctLayer(XconfigLayerBase):
    """
    This class is for lines like
     'idct-layer name=idct dim=40 cepstral-lifter=22 affine-transform-file=foo/bar/idct.mat'

    This is used to convert input MFCC-features to Filterbank featurs. The
    affine transformation is written out to the file specified via
    'affine-transform-file=xxx'.
    The output dimension of the layer may be specified via 'dim=xxx', but if not specified,
    the dimension defaults to the same as the input.

    See other configuration values below.

    Parameters of the class, and their defaults:
      input='[-1]'             [Descriptor giving the input of the layer.]
      dim=None                   [Output dimension of layer; defaults to the same as the input dim.]
      cepstral-lifter=22       [Apply liftering co-efficient.]
      affine-transform-file='' [Must be specified.]

    """
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == 'idct-layer'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = {'input': '[-1]',
                       'dim': -1,
                       'cepstral-lifter': 22.0,
                       'affine-transform-file': ''}

    def check_configs(self):
        if self.config['affine-transform-file'] is None:
            raise RuntimeError("affine-transform-file must be set.")


    def output_name(self, auxiliary_output = None):
        # Fixed affine layer computes only one vector, there are no intermediate
        # vectors.
        assert auxiliary_output == None
        return self.name

    def output_dim(self, auxiliary_output = None):
        output_dim = self.config['dim']
        # If not set, the output-dim defaults to the input-dim.
        if output_dim <= 0:
            output_dim = self.descriptors['input']['dim']
        return output_dim

    def get_full_config(self):
        ans = []

        # note: each value of self.descriptors is (descriptor, dim,
        # normalized-string, output-string).
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        descriptor_final_string = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        output_dim = self.output_dim()
        transform_file = self.config['affine-transform-file']

        idct_mat = common_lib.compute_idct_matrix(
            input_dim, output_dim, self.config['cepstral-lifter'])
        # append a zero column to the matrix, this is the bias of the fixed
        # affine component
        for n in range(0, output_dim):
            idct_mat[n].append(0)
        common_lib.write_kaldi_matrix(transform_file, idct_mat)

        # write the 'real' component to final.config
        line = 'component name={0} type=FixedAffineComponent matrix={1}'.format(
            self.name, transform_file)
        ans.append(('final', line))
        # write a random version of the component, with the same dims, to ref.config
        line = 'component name={0} type=FixedAffineComponent input-dim={1} output-dim={2}'.format(
            self.name, input_dim, output_dim)
        ans.append(('ref', line))
        # the component-node gets written to final.config and ref.config.
        line = 'component-node name={0} component={0} input={1}'.format(
            self.name, descriptor_final_string)
        ans.append(('final', line))
        ans.append(('ref', line))
        return ans


def test_layers():
    # for some config lines that should be printed the same way as they
    # are read, check that this is the case.
    for x in [ 'input name=input dim=30' ]:
        assert str(config_line_to_object(x, [])) == x
