# Copyright 2018    Johns Hopkins University (Author: Dan Povey)
#           2016    Vijayaditya Peddinti
# Apache 2.0.



""" This module has the implementation of convolutional layers.
"""
from __future__ import print_function
import math
import re
import sys
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase


# This class is for lines like the following:
#

#  conv-batchnorm-layer name=conv2 height-in=40 height-out=40 \
#      num-filters-out=64 height-offsets=-1,0,1 time-offsets=-1,0,1 \
#      required-time-offsets=0
#  or (with NormalizeLayer instead of batch-norm, and with subsampling on the height axis):
#  conv-renorm-layer name=conv3 height-in=40 height-out=20 \
#      height-subsample-out=2 num-filters-out=128 height-offsets=-1,0,1 \
#       time-offsets=-1,0,1 required-time-offsets=0
#
# You don't specify subsampling on the time axis explicitly, it's implicit
# in the 'time-offsets' which are the same as the splicing indexes in a TDNN,
# and which, unlike the height offsets, operate relative to a fixed clock,
# so that after subsampling by a factor of 2, we'd expect all time-offsets
# of subsequent layers to be a factor of 2.  You don't specify the input
# num-filters either; it's worked out from the input height and the input dim.
#
# The layer-name encodes the use (or not) of batch normalization, so that if you
# want to skip batch normalization you could just call it 'conv-layer'.
#
# If batch-normalization is used, it's *spatial* batch-normalization, meaning
# that the offset and scale is specific to the output filter, but shared across
# all time and height offsets.
#
# Most of the configuration values mirror same-named values in class
# TimeHeightConvolutionComponent, and for a deeper understanding of what's going
# on you should look at the comment by its declaration, in
# src/nnet3/nnet-convolutional-component.h.
#
# Parameters of the class, and their defaults if they have defaults:
#
#   input='[-1]'             Descriptor giving the input of the layer.
#   height-in                The height of the input image, e.g. 40 if the input
#                            is MFCCs.  The num-filters-in is worked out as
#                            (dimension of input) / height-in.  If the preceding
#                            layer is a convolutional layer, height-in should be
#                            the same as the height-out of the preceding layer.
#   height-subsample-out=1   The height subsampling factor, will be e.g. 2 if you
#                            want to subsample by a factor of 2 on the height
#                            axis.
#   height-out               The height of the output image.  This will normally
#                            be <= (height-in / height-subsample-out).
#                            Zero-padding on the height axis may be implied by a
#                            combination of this and height-offsets-in, e.g. if
#                            height-out==height-in and height-subsample-out=1
#                            and height-offsets=-2,-1,0,1 then we'd be padding
#                            by 2 pixels on the bottom and 1 on the top; see
#                            comments in nnet-convolutional-layers.h for more
#                            details.
#   height-offsets           The offsets on the height axis that define what
#                            inputs require for each output pixel; will
#                            often be something like -1,0,1 (if zero-padding
#                            on height axis) or 0,1,2 otherwise.  These are
#                            comparable to TDNN splicing offsets; e.g. if
#                            height-offsets=-1,0,1 then height 10 at the output
#                            would take input from heights 9,10,11 at the input.
#   num-filters-out          The number of output filters.  The output dimension
#                            of this layer is num-filters-out * height-out; the
#                            filter dim varies the fastest (filter-stride == 1).
#   time-offsets             The input offsets on the time axis; these are
#                            interpreted just like the splicing indexes in TDNNs.
#                            E.g. if time-offsets=-2,0,2 then time 100 at the
#                            output would require times 98,100,102 at the input.
#   required-time-offsets    The subset of 'time-offsets' that are required in
#                            order to produce an output; if the set has fewer
#                            elements than 'time-offsets' then it implies some
#                            kind of zero-padding on the time axis is allowed.
#                            Defaults to the same as 'time-offsets'.  For speech
#                            tasks we recommend not to set this, as the normal
#                            padding approach is to pad with copies of the
#                            first/last frame, which is handled automatically in
#                            the calling code.
#   target-rms=1.0           Only applicable if the layer type is
#                            conv-batchnorm-layer or
#                            conv-normalize-layer.  This will affect the
#                            scaling of the output features (larger -> larger),
#                            and sometimes we set target-rms=0.5 for the layer
#                            prior to the final layer to make the final layer
#                            train more slowly.
#   self-repair-scale=2.0e-05  This affects the ReLu's.  It is a scale on the
#                            'self-repair' mechanism that nudges the inputs to the
#                            ReLUs into the appropriate range in cases where
#                            the unit is active either too little of the time
#                            (<10%) or too much of the time (>90%).
#
# The following initialization and natural-gradient related options are, if
# provided, passed through to the config file; if not, they are left at the
# defaults in the code.  See nnet-convolutional-component.h for more information.
#
#  param-stddev, bias-stddev, max-change, learning-rate-factor (float)
#  use-natural-gradient (bool)
#  rank-in, rank-out    (int)
#  num-minibatches-history (float)
#  alpha-in, alpha-out (float)
# the following is also passed into the convolution components, if specified:
#  l2-regularize (float)

class XconfigConvLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        for operation in first_token.split('-')[:-1]:
            assert operation in ['conv', 'renorm', 'batchnorm', 'relu',
                                 'noconv', 'dropout', 'so']
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input':'[-1]',
                       'height-in':-1,
                       'height-subsample-out':1,
                       'height-out':-1,
                       'height-offsets':'',
                       'num-filters-out':-1,
                       'time-offsets':'',
                       'required-time-offsets':'',
                       'target-rms':1.0,
                       'self-repair-scale': 2.0e-05,
                       'self-repair-lower-threshold': 0.05,
                       # the following are not really inspected by this level of
                       # code, just passed through (but not if left at '').
                       'param-stddev':'', 'bias-stddev':'',
                       'max-change': 0.75, 'learning-rate-factor':'',
                       'use-natural-gradient':'',
                       'rank-in':'', 'rank-out':'', 'num-minibatches-history':'',
                       'alpha-in':'', 'alpha-out':'', 'l2-regularize':'',
                       'dropout-proportion': 0.5}

    def set_derived_configs(self):
        # sets 'num-filters-in'.
        input_dim = self.descriptors['input']['dim']
        height_in = self.config['height-in']
        if height_in <= 0:
            raise RuntimeError("height-in must be specified");
        if input_dim % height_in != 0:
            raise RuntimeError("Input dimension {0} is not a multiple of height-in={1}".format(
                input_dim, height_in))
        self.config['num-filters-in'] = input_dim / height_in


    # Check whether 'str' is a sorted, unique, nonempty list of integers, like -1,0,1.,
    # returns true if so.
    def check_offsets_var(self, str):
        try:
            a = [ int(x) for x in str.split(",") ]
            if len(a) == 0:
                return False
            for i in range(len(a) - 1):
                if a[i] >= a[i+1]:
                    return False
            return True
        except:
            return False

    def check_configs(self):
        # Do some basic checking of the configs.  The component-level code does
        # some more thorough checking, but if you set the height-out too small it
        # prints it as a warning, which the user may not see, so at a minimum we
        # want to check for that here.
        height_subsample_out = self.config['height-subsample-out']
        height_in = self.config['height-in']
        height_out = self.config['height-out']
        if height_subsample_out <= 0:
            raise RuntimeError("height-subsample-out has invalid value {0}.".format(
                height_subsample_out))
        # we already checked height-in in set_derived_configs.
        if height_out <= 0:
            raise RuntimeError("height-out has invalid value {0}.".format(
                height_out))
        if height_out * height_subsample_out > height_in:
            raise RuntimeError("The combination height-in={0}, height-out={1} and "
                               "height-subsample-out={2} does not look right "
                               "(height-out too large).".format(
                                   height_in, height_out, height_subsample_out))
        height_offsets = self.config['height-offsets']
        time_offsets = self.config['time-offsets']
        required_time_offsets = self.config['required-time-offsets']

        if not 'noconv' in self.layer_type.split('-'):
            # only check height-offsets, time-offsets and required-time-offsets if there
            # is actually a convolution in this layer.
            if not self.check_offsets_var(height_offsets):
                raise RuntimeError("height-offsets={0} is not valid".format(height_offsets))
            if not self.check_offsets_var(time_offsets):
                raise RuntimeError("time-offsets={0} is not valid".format(time_offsets))
            if required_time_offsets != "" and not self.check_offsets_var(required_time_offsets):
                raise RuntimeError("required-time-offsets={0} is not valid".format(
                    required_time_offsets))

        if height_out * height_subsample_out < \
           height_in - len(height_offsets.split(',')):
            raise RuntimeError("The combination height-in={0}, height-out={1} and "
                               "height-subsample-out={2} and height-offsets={3} "
                               "does not look right (height-out too small).")

        if self.config['target-rms'] <= 0.0:
            raise RuntimeError("Config value target-rms={0} is not valid".format(
                self.config['target_rms']))

    def auxiliary_outputs(self):
        return []

    def output_name(self, auxiliary_output = None):
        assert auxiliary_output is None
        # note: the [:-1] is to remove the '-layer'.
        operations = self.layer_type.split('-')[:-1]
        if operations[-1] == 'noconv':
            operations = operations[:-1]
        assert len(operations) >= 1
        last_operation = operations[-1]
        assert last_operation in ['relu', 'conv', 'renorm', 'batchnorm', 'dropout', 'so']
        # we'll return something like 'layer1.batchnorm'.
        return '{0}.{1}'.format(self.name, last_operation)

    def output_dim(self, auxiliary_output = None):
        assert auxiliary_output is None
        return self.config['num-filters-out'] * self.config['height-out']

    def get_full_config(self):
        ans = []
        config_lines = self._generate_cnn_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in CNN initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # convenience function to generate the CNN config
    def _generate_cnn_config(self):
        configs = []

        name = self.name

        # These 3 variables will be updated as we add components.
        cur_num_filters = self.config['num-filters-in']
        cur_height = self.config['height-in']
        cur_descriptor = self.descriptors['input']['final-string']

        # note: the [:-1] is to remove the '-layer'.
        operations = self.layer_type.split('-')[:-1]
        if operations[-1] == 'noconv':
            operations = operations[:-1]
        # e.g.:
        # operations = [ 'conv', 'relu', 'batchnorm' ]
        # or:
        # operations = [ 'relu', 'conv', 'renorm' ]

        for operation in operations:
            if operation == 'conv':
                a = []
                for opt_name in [
                        'param-stddev', 'bias-stddev', 'use-natural-gradient',
                        'max-change', 'rank-in', 'rank-out', 'num-minibatches-history',
                        'alpha-in', 'alpha-out', 'num-filters-in', 'num-filters-out',
                        'height-in','height-out', 'height-subsample-out',
                        'height-offsets', 'time-offsets', 'required-time-offsets',
                        'learning-rate-factor', 'l2-regularize' ]:
                    value = self.config[opt_name]
                    if value != '':
                        a.append('{0}={1}'.format(opt_name, value))
                conv_opts = ' '.join(a)

                configs.append('component name={0}.conv type=TimeHeightConvolutionComponent '
                               '{1}'.format(name, conv_opts))
                configs.append('component-node name={0}.conv component={0}.conv '
                               'input={1}'.format(name, cur_descriptor))
                cur_num_filters = self.config['num-filters-out']
                cur_height = self.config['height-out']
            elif operation == 'batchnorm':
                configs.append('component name={0}.batchnorm  type=BatchNormComponent dim={1} '
                               'block-dim={2} target-rms={3}'.format(
                                   name, cur_num_filters * cur_height, cur_num_filters,
                                   self.config['target-rms']))
                configs.append('component-node name={0}.batchnorm component={0}.batchnorm '
                               'input={1}'.format(name, cur_descriptor))
            elif operation == 'renorm':
                configs.append('component name={0}.renorm type=NormalizeComponent '
                           'dim={1} target-rms={2}'.format(
                               name, cur_num_filters * cur_height,
                               self.config['target-rms']))
                configs.append('component-node name={0}.renorm component={0}.renorm '
                               'input={1}'.format(name, cur_descriptor))
            elif operation == 'relu':
                configs.append('component name={0}.relu type=RectifiedLinearComponent '
                               'dim={1} block-dim={2} self-repair-scale={3} '
                               'self-repair-lower-threshold={4}'.format(
                                   name, cur_num_filters * cur_height, cur_num_filters,
                                   self.config['self-repair-scale'],
                                   self.config['self-repair-lower-threshold']))
                configs.append('component-node name={0}.relu component={0}.relu '
                               'input={1}'.format(name, cur_descriptor))
            elif operation == 'dropout':
                configs.append('component name={0}.dropout type=DropoutComponent '
                           'dim={1} dropout-proportion={2}'.format(
                               name, cur_num_filters * cur_height,
                               self.config['dropout-proportion']))
                configs.append('component-node name={0}.dropout component={0}.dropout '
                               'input={1}'.format(name, cur_descriptor))
            elif operation == 'so':
                configs.append('component name={0}.so type=ScaleAndOffsetComponent '
                           'dim={1} block-dim={2}'.format(
                               name, cur_num_filters * cur_height, cur_num_filters))
                configs.append('component-node name={0}.so component={0}.so '
                               'input={1}'.format(name, cur_descriptor))
            else:
                raise RuntimeError("Un-handled operation type: " + operation)

            cur_descriptor = '{0}.{1}'.format(name, operation)

        return configs


# This class is for lines like the following:
#
# res-block name=res1 num-filters=64 height=32 time-period=1
#
# It implements a residual block as in ResNets, with pre-activation, and with
# some small differences-- basically, instead of adding the input to the output,
# we put a convolutional layer in there but initialize it to the unit matrix and
# if you want you can give it a relatively small (or even zero) learning rate
# and max-change.  And there is batch-norm in that path also.
#
# The number of filters is the same on the input and output; it is actually
# redundant to write it in the config file, because given that we know the
# height, we can work it out from the dimension of the input (as dimension =
# height * num-filters).  But we allow it to be specified anyway, for clarity.
#
# Note: the res-block does not support subsampling or changing the number of
# filters.  If you want to do that, we recommend that you should do it with a
# single relu-batchnorm-conv-layer.
#
# Here are the most important configuration values, with defaults shown if
# defaults exist:
#
# input='[-1]'    Descriptor giving the input of the layer.
# height          The input and output height of the image, e.g. 40.  Note: the width
#                 is associated with the time dimension and is dealt with
#                 implicitly, so it's not specified here.
# num-filters     The number of filters on the input and output, e.g. 64.
#                 It does not have to be specified; if it is not specified,
#                 we work it out from the input dimension.
# num-bottleneck-filters   If specified then this will be a 'bottleneck'
#                 ResBlock, in which there is a 1x1 convolution from
#                 num-filters->num-bottleneck-filters, a 3x3 convolution
#                 from num-bottleneck-filters->num-bottleneck-filters, and
#                 a 1x1 convolution from num-bottleneck-filters->num-filters.
#
# time-period=1   Think of this as the stride in the time dimension.  At the
#                 input of the network will always have time-period=1; then
#                 after subsampling once in time we'd have time-period=2; then
#                 after subsampling again we'd have time-period=4.  Because of
#                 the way nnet3 works, subsampling on the time axis is an
#                 implicit, not explicit, operation.
# height-period=1  This will almost always be left at the default (1).  It is
#                 analogous to time-period, but because the height, unlike the
#                 time, is explicitly subsampled, in normal topologies this should
#                 be left at 1.
#
# bypass-source=noop
#                       The output of this component is Sum(convolution, x), and
#                       this option controls what 'x' is.  There are 3 options
#                       here: 'noop', 'input', 'relu' or 'batchnorm'.  'noop' is
#                       equivalent to 'input' in what it computes; it just
#                       inserts a 'noop' component in order to make the
#                       computation more efficient.  For both 'noop' and
#                       'input', x is the input to this component.  If
#                       bypass-source=relu then we use the relu of the
#                       input; if 'batchnorm', then we use the relu+batchnorm of
#                       the input.
# allow-zero-padding=true By default this will allow zero-padding in the time
#                       dimension, meaning that you don't need extra frames at
#                       the input to compute the output.  There may be ASR
#                       applications where you want to pad in the time dimension
#                       with repeats of the first or last frame (as we do for
#                       TDNNs), where it would be appropriate to write
#                       allow-zero-padding=false.  Note: the way we have
#                       set it up, it does zero-padding on the height axis
#                       regardless
#
# Less important config variables:
#  self-repair-scale=2.0e-05  This affects the ReLu's.  It is a scale on the
#                            'self-repair' mechanism that nudges the inputs to the
#                            ReLUs into the appropriate range in cases where
#                            the unit is active either too little of the time
#                            (<10%) or too much of the time (>90%).
#  max-change=0.75           Max-parameter-change constant (per minibatch)
#                            used for convolutional components.
#
#
# The following natural-gradient-related configuration variables are passed in
# to the convolution components, if specified:
#  use-natural-gradient (bool)
#  rank-in, rank-out    (int)
#  num-minibatches-history (float)
#  alpha-in, alpha-out (float)
# the following is also passed into the convolution components, if specified:
#  l2-regularize (float)
#

class XconfigResBlock(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == 'res-block'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input':'[-1]',
                       'height':-1,
                       'num-filters':-1,
                       'num-bottleneck-filters':-1,
                       'time-period':1,
                       'height-period':1,
                       'self-repair-scale': 2.0e-05,
                       'self-repair-lower-threshold1': 0.05,
                       'self-repair-lower-threshold2': 0.05,
                       'self-repair-lower-threshold3': 0.05,
                       'max-change': 0.75,
                       'allow-zero-padding': True,
                       'bypass-source' : 'noop',
                       # the following are not really inspected by this level of
                       # code, just passed through (but not if left at '').
                       'param-stddev':'', 'bias-stddev':'',
                       'use-natural-gradient':'',
                       'rank-in':'', 'rank-out':'',
                       'num-minibatches-history':'',
                       'alpha-in':'', 'alpha-out':'', 'l2-regularize':'' }

    def set_derived_configs(self):
        # set 'num-filters' or check it..
        input_dim = self.descriptors['input']['dim']
        height = self.config['height']

        cur_num_filters = self.config['num-filters']
        if cur_num_filters == -1:
            if input_dim % height != 0:
                raise RuntimeError("Specified image height {0} does not "
                                   "divide the input dim {1}".format(
                                       height, input_dim))
            self.config['num-filters'] = input_dim / height
        elif input_dim != cur_num_filters * height:
            raise RuntimeError("Expected the input-dim to equal "
                               "height={0} * num-filters={1} = {2}, but "
                               "it is {3}".format(
                                   height, cur_num_filters,
                                   height * cur_num_filters,
                                   input_dim));

    def check_configs(self):
        # we checked the dimensions in set_derived_configs.
        if not self.config['bypass-source'] in [
                'input', 'noop', 'relu', 'batchnorm' ]:
            raise RuntimeError("Expected direct-convolution-source to "
                               "be input, relu or batchnorm, got: {1}".format(
                                   self.config['direct-convolution-source']))

    def auxiliary_outputs(self):
        return []

    def output_name(self, auxiliary_output = None):
        bypass_source = self.config['bypass-source']
        b = self.config['num-bottleneck-filters']
        conv = ('{0}.conv2' if b <= 0 else '{0}.conv3').format(self.name)
        if bypass_source == 'input':
            residual = self.descriptors['input']['final-string']
        elif bypass_source == 'noop':
            # we let the noop be the sum of the convolutional part and the
            # input, so just return the output of the no-op component.
            return '{0}.noop'.format(self.name)
        elif bypass_source == 'relu':
            residual = '{0}.relu1'.format(self.name)
        else:
            assert bypass_source == 'batchnorm'
            residual = '{0}.batchnorm1'.format(self.name)

        return 'Sum({0}, {1})'.format(conv, residual)

    def output_dim(self, auxiliary_output = None):
        assert auxiliary_output is None
        input_dim = self.descriptors['input']['dim']
        return input_dim

    def get_full_config(self):
        ans = []
        b = self.config['num-bottleneck-filters']
        if b <= 0:
            config_lines = self._generate_normal_resblock_config()
        else:
            config_lines = self._generate_bottleneck_resblock_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in CNN initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # _generate_normal_resblock_config is a convenience function to generate the
    # res-block config (the non-bottleneck version).
    #
    # The main path inside the res-block in the non-bottleneck case is as
    # follows:
    #
    # input -> relu1 -> batchnorm1 -> conv1 -> relu2 -> batchnorm2 -> conv2
    #
    # We put the relu before the batchnorm because we think it makes more sense;
    # because the Torch people seemed to find that this works better
    # (https://github.com/gcr/torch-residual-networks/issues/5);
    # and because in our batchnorm component we haven't implemented the beta and
    # gamma; these would be essential to having it work before relu, but
    # when before a convolution or linear component, they add no extra modeling
    # power.
    #
    # The output of the res-block can be the sum of the last convolutional
    # component (conv2), with the input.  However, the option ('bypass-source')
    # controls whether we sum with the raw input, or its relu or relu+batchnorm.
    # If the term is going to be the raw input, we give the option ('noop') and
    # to cache the output sum via a NoOpComponent)-- because due to how nnet3
    # works, if we didn't do this, redundant summing operations would take
    # place.
    def _generate_normal_resblock_config(self):
        configs = []

        name = self.name
        num_filters = self.config['num-filters']
        assert self.config['num-bottleneck-filters'] == -1
        height = self.config['height']
        input_descriptor = self.descriptors['input']['final-string']
        allow_zero_padding = self.config['allow-zero-padding']
        height_period = self.config['height-period']
        time_period = self.config['time-period']

        # input -> relu1 -> batchnorm1 -> conv1 -> relu2 -> batchnorm2 -> conv2
        cur_descriptor = input_descriptor
        for n in [1, 2]:
            # the ReLU
            configs.append('component name={0}.relu{1} type=RectifiedLinearComponent '
                           'dim={2} block-dim={3} self-repair-scale={4} '
                           'self-repair-lower-threshold={5}'.format(
                               name, n, num_filters * height, num_filters,
                               self.config['self-repair-scale'],
                               self.config['self-repair-lower-threshold{0}'.format(n)]))
            configs.append('component-node name={0}.relu{1} component={0}.relu{1} '
                           'input={2}'.format(name, n, cur_descriptor))

            cur_descriptor = '{0}.relu{1}'.format(name, n)

            # the batch-norm
            configs.append('component name={0}.batchnorm{1}  type=BatchNormComponent dim={2} '
                               'block-dim={3}'.format(
                                   name, n, num_filters * height,
                                   num_filters))
            configs.append('component-node name={0}.batchnorm{1} component={0}.batchnorm{1} '
                           'input={2}'.format(name, n, cur_descriptor))
            cur_descriptor = '{0}.batchnorm{1}'.format(name, n)


            # the convolution.
            a = []
            for opt_name in [
                    'param-stddev', 'bias-stddev', 'use-natural-gradient',
                    'max-change', 'rank-in', 'rank-out', 'num-minibatches-history',
                    'alpha-in', 'alpha-out', 'l2-regularize' ]:
                value = self.config[opt_name]
                if value != '':
                        a.append('{0}={1}'.format(opt_name, value))
            conv_opts = ('height-in={h} height-out={h} height-offsets=-{hp},0,{hp} '
                         'time-offsets=-{p},0,{p} '
                         'num-filters-in={f} num-filters-out={f} {r} {o}'.format(
                             h=height, hp=height_period, p=time_period, f=num_filters,
                             r=('required-time-offsets=0' if allow_zero_padding else ''),
                             o=' '.join(a)))

            configs.append('component name={0}.conv{1} type=TimeHeightConvolutionComponent '
                           '{2}'.format(name, n, conv_opts))
            configs.append('component-node name={0}.conv{1} component={0}.conv{1} '
                           'input={2}'.format(name, n, cur_descriptor))
            cur_descriptor = '{0}.conv{1}'.format(name, n)



        if self.config['bypass-source'] == 'noop':
            dim = self.descriptors['input']['dim']
            configs.append('component name={0}.noop dim={1} type=NoOpComponent'.format(
                name, dim))
            configs.append('component-node name={0}.noop component={0}.noop '
                           'input=Sum({1}, {0}.conv2)'.format(name,
                                                              input_descriptor))

        # Note: the function 'output_name' is responsible for returning the
        # descriptor corresponding to the output of the network.
        return configs



    # _generate_bottleneck_resblock_config is a convenience function to generate the
    # res-block config (this is the bottleneck version, where there is
    # a 3x3 kernel with a smaller number of filters than at the input and output,
    # sandwiched between two 1x1 kernels.
    #
    # The main path inside the res-block in the bottleneck case is as follows:
    #
    # input -> relu1 -> batchnorm1 -> conv1 -> relu2 -> batchnorm2 -> conv2 ->
    #   relu3 -> batchnorm3 -> conv3
    #
    # power.
    #
    # The output of the res-block can be the sum of the last convolutional
    # component (conv3), with the input.  However we give the option
    # ('bypass-source') to sum with the raw input, or its relu or
    # relu+batchnorm.  If the term is going to be the raw input, we give the
    # option ('noop') and to cache the output sum via a NoOpComponent)-- because
    # due to how nnet3 works, if we didn't do this, redundant summing operations
    # would take place.
    def _generate_bottleneck_resblock_config(self):
        configs = []

        name = self.name
        num_filters = self.config['num-filters']
        num_bottleneck_filters = self.config['num-bottleneck-filters']
        assert num_bottleneck_filters > 0
        height = self.config['height']
        input_descriptor = self.descriptors['input']['final-string']
        allow_zero_padding = self.config['allow-zero-padding']
        height_period = self.config['height-period']
        time_period = self.config['time-period']

        # input -> relu1 -> batchnorm1 -> conv1 -> relu2 -> batchnorm2 -> conv2
        cur_descriptor = input_descriptor
        cur_num_filters = num_filters

        for n in [1, 2, 3]:
            # the ReLU
            configs.append('component name={0}.relu{1} type=RectifiedLinearComponent '
                           'dim={2} block-dim={3} self-repair-scale={4} '
                           'self-repair-lower-threshold={5}'.format(
                               name, n, cur_num_filters * height, cur_num_filters,
                               self.config['self-repair-scale'],
                               self.config['self-repair-lower-threshold{0}'.format(n)]))
            configs.append('component-node name={0}.relu{1} component={0}.relu{1} '
                           'input={2}'.format(name, n, cur_descriptor))

            cur_descriptor = '{0}.relu{1}'.format(name, n)

            # the batch-norm
            configs.append('component name={0}.batchnorm{1}  type=BatchNormComponent dim={2} '
                               'block-dim={3}'.format(
                                   name, n, cur_num_filters * height,
                                   cur_num_filters))
            configs.append('component-node name={0}.batchnorm{1} component={0}.batchnorm{1} '
                           'input={2}'.format(name, n, cur_descriptor))
            cur_descriptor = '{0}.batchnorm{1}'.format(name, n)


            # the convolution.
            a = []
            for opt_name in [
                    'param-stddev', 'bias-stddev', 'use-natural-gradient',
                    'max-change', 'rank-in', 'rank-out', 'num-minibatches-history',
                    'alpha-in', 'alpha-out', 'l2-regularize' ]:
                value = self.config[opt_name]
                if value != '':
                        a.append('{0}={1}'.format(opt_name, value))

            height_offsets = ('-{hp},0,{hp}'.format(hp=height_period) if n == 2 else '0')
            time_offsets = ('-{t},0,{t}'.format(t=time_period) if n == 2 else '0')
            next_num_filters = (num_filters if n == 3 else num_bottleneck_filters)
            conv_opts = ('height-in={h} height-out={h} height-offsets={ho} time-offsets={to} '
                         'num-filters-in={fi} num-filters-out={fo} {r} {o}'.format(
                             h=height, ho=height_offsets, to=time_offsets,
                             fi=cur_num_filters, fo=next_num_filters,
                             r=('required-time-offsets=0' if allow_zero_padding else ''),
                             o=' '.join(a)))

            configs.append('component name={0}.conv{1} type=TimeHeightConvolutionComponent '
                           '{2}'.format(name, n, conv_opts))
            configs.append('component-node name={0}.conv{1} component={0}.conv{1} '
                           'input={2}'.format(name, n, cur_descriptor))
            cur_descriptor = '{0}.conv{1}'.format(name, n)
            cur_num_filters = next_num_filters


        if self.config['bypass-source'] == 'noop':
            dim = self.descriptors['input']['dim']
            configs.append('component name={0}.noop dim={1} type=NoOpComponent'.format(
                name, dim))
            configs.append('component-node name={0}.noop component={0}.noop '
                           'input=Sum({1}, {0}.conv3)'.format(name,
                                                              input_descriptor))

        # Note: the function 'output_name' is responsible for returning the
        # descriptor corresponding to the output of the network.
        return configs


# This class is for lines like the following:
#
# res2-block name=res1 num-filters=64 height=32 time-period=1
#
# It is a residual block with post-activations, which does not support
# downsampling (strided convolution) or changing the number of filters;
# for that, see res2-downsample-block.
# It's a pretty standard res-block, more standard than "res-block" (XconfigResBlock).
#
# The number of filters is the same on the input and output; it is actually
# redundant to write it in the config file, because given that we know the
# height, we can work it out from the dimension of the input (as dimension =
# height * num-filters).  But we allow it to be specified anyway, for clarity.
#

# Here are the most important configuration values, with defaults shown if
# defaults exist:
#
# input='[-1]'    Descriptor giving the input of the layer.
# height          The input and output height of the image, e.g. 40.  Note: the width
#                 is associated with the time dimension and is dealt with
#                 implicitly, so it's not specified here.
# num-filters     The number of filters on the input and output, e.g. 64.
#                 It does not have to be specified; if it is not specified,
#                 we work it out from the input dimension.
# num-bottleneck-filters   If specified then this will be a 'bottleneck'
#                 ResBlock, in which there is a 1x1 convolution from
#                 num-filters->num-bottleneck-filters, a 3x3 convolution
#                 from num-bottleneck-filters->num-bottleneck-filters, and
#                 a 1x1 convolution from num-bottleneck-filters->num-filters.
# time-period=1   Think of this as the stride in the time dimension.  At the
#                 input of the network will always have time-period=1; then
#                 after subsampling once in time we'd have time-period=2; then
#                 after subsampling again we'd have time-period=4.  Because of
#                 the way nnet3 works, subsampling on the time axis is an
#                 implicit, not explicit, operation.
# allow-zero-padding=true By default this will allow zero-padding in the time
#                       dimension, meaning that you don't need extra frames at
#                       the input to compute the output.  There may be ASR
#                       applications where you want to pad in the time dimension
#                       with repeats of the first or last frame (as we do for
#                       TDNNs), where it would be appropriate to write
#                       allow-zero-padding=false.  Note: the way we have
#                       set it up, it does zero-padding on the height axis
#                       regardless
#
# Less important config variables:
#  self-repair-scale=2.0e-05  This affects the ReLu's.  It is a scale on the
#                            'self-repair' mechanism that nudges the inputs to the
#                            ReLUs into the appropriate range in cases where
#                            the unit is active either too little of the time
#                            (<10%) or too much of the time (>90%).
#  max-change=0.75           Max-parameter-change constant (per minibatch)
#                            used for convolutional components.
#
#
# The following natural-gradient-related configuration variables are passed in
# to the convolution components, if specified:
#  use-natural-gradient (bool)
#  rank-in, rank-out    (int)
#  num-minibatches-history (float)
#  alpha-in, alpha-out (float)
# the following is also passed into the convolution components, if specified:
#  l2-regularize (float)

class XconfigRes2Block(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == 'res2-block'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input':'[-1]',
                       'height':-1,  # sets height-in and height-out
                       'height-in':-1,
                       'height-out':-1,
                       'num-filters':-1, # interpreted as num-filters-out.
                       'num-bottleneck-filters':-1,
                       'time-period':1,
                       'self-repair-scale': 2.0e-05,
                       'self-repair-lower-threshold1': 0.05,
                       'self-repair-lower-threshold2': 0.05,
                       'self-repair-lower-threshold3': 0.05,
                       'max-change': 0.75,
                       'allow-zero-padding': True,
                       # the following are not really inspected by this level of
                       # code, just passed through (but not if left at '').
                       'param-stddev':'', 'bias-stddev':'',
                       'use-natural-gradient':'',
                       'rank-in':'', 'rank-out':'',
                       'num-minibatches-history':'',
                       'alpha-in':'', 'alpha-out':'',
                       'l2-regularize':'' }

    def set_derived_configs(self):
        input_dim = self.descriptors['input']['dim']

        if not ((self.config['height'] > 0  and self.config['height-in'] == -1 and
                 self.config['height-out'] == -1) or
                (self.config['height-out'] > 0 and self.config['height-in'] > 0)):
            raise RuntimeError("You must specify height, or height-in and height-out, for res2-block.")

        if not (self.config['height-in'] > 0 and self.config['height-out'] > 0):
            height = self.config['height']
            if not height > 0:
                raise RuntimeError("You must specify either height, or height-in and height-out, for "
                                   "res2-block.")
            self.config['height-in'] = height
            self.config['height-out'] = height

        height_in = self.config['height-in']
        if input_dim % height_in != 0:
            raise RuntimeError("Specified input image height {0} does not "
                                   "divide the input dim {1}".format(
                                       height_in, input_dim))
            self.config['num-filters'] = input_dim / height

    def check_configs(self):
        if self.config['num-filters'] == -1:
            raise RuntimeError("You must specify num-filters for res2-block.")

    def auxiliary_outputs(self):
        return []

    def output_name(self, auxiliary_output = None):
        b = self.config['num-bottleneck-filters']
        return ('{0}.relu2' if b <= 0 else '{0}.relu3').format(self.name)

    def output_dim(self, auxiliary_output = None):
        assert auxiliary_output is None
        return self.config['height-out'] * self.config['num-filters']

    def get_full_config(self):
        ans = []
        b = self.config['num-bottleneck-filters']
        if b <= 0:
            config_lines = self._generate_normal_resblock_config()
        else:
            config_lines = self._generate_bottleneck_resblock_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in CNN initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # _generate_normal_resblock_config is a convenience function to generate the
    # res-block config (the non-bottleneck version).
    #
    # The main path inside the res-block in the non-bottleneck case is as
    # follows:
    #
    # input -> conv1 -> batchnorm1 -> scaleoffset1 -> relu1 -> conv2 -> batchnorm2 -> scaleoffset2 -> relu2
    #
    # where the 'scaleoffsetN' are ScaleAndOffsetComponent, which conventionally would be
    # considered part of the BatchNorm.
    #
    # The relu2 actually sees the sum of the input and  'scaleoffset2'-- which gives us the bypass
    # connection.
    def _generate_normal_resblock_config(self):
        configs = []
        name = self.name
        assert self.config['num-bottleneck-filters'] == -1
        input_dim = self.descriptors['input']['dim']
        height_in = self.config['height-in']
        height_out = self.config['height-out']
        time_period_out = self.config['time-period']
        if not input_dim % height_in == 0:
            raise RuntimeError("input-dim {0} does not divide height-in {1}".format(
                input_dim, height_in))
        num_filters_in = input_dim / height_in
        num_filters_out = self.config['num-filters']

        if height_out != height_in:
            if height_out < height_in / 2 - 1 or height_out > height_in /  2 + 1:
                raise RuntimeError("Expected height-out to be about half height-in, or the same: "
                                   "height-in={0} height-out={1}".format(height_in, height_out))
            if not time_period_out % 2 == 0:
                raise RuntimeError("Expected time-period to be a multiple of 2 if you are subsampling "
                                   "on height.")
            time_period_in = time_period_out / 2
            height_subsample = 2
        else:
            time_period_in = time_period_out
            height_subsample = 1


        cur_time_period = time_period_in
        cur_num_filters = num_filters_in
        cur_height = height_in

        input_descriptor = self.descriptors['input']['final-string']
        allow_zero_padding = self.config['allow-zero-padding']
        if height_subsample == 1 and num_filters_in == num_filters_out:
            bypass_descriptor = input_descriptor
        else:
            bypass_descriptor = '{0}.conv_bypass'.format(name)

        cur_descriptor = input_descriptor

        # get miscellaneous convolution options passed in from the xconfig line
        a = []
        for opt_name in [
                'param-stddev', 'bias-stddev', 'use-natural-gradient',
                'max-change', 'rank-in', 'rank-out', 'num-minibatches-history',
                'alpha-in', 'alpha-out', 'l2-regularize' ]:
            value = self.config[opt_name]
            if value != '':
                a.append('{0}={1}'.format(opt_name, value))
        misc_conv_opts = ' '.join(a)

        for n in [1, 2]:
            # the convolution.
            conv_opts = ('height-in={hi} height-out={ho} height-offsets=-1,0,1 '
                         'height-subsample-out={hs} '
                         'time-offsets=-{p},0,{p} '
                         'num-filters-in={fi} num-filters-out={fo} {r} {o}'.format(
                             hi=cur_height, ho=height_out,
                             p=cur_time_period,
                             hs=(height_subsample if n == 1 else 1),
                             fi=cur_num_filters,
                             fo=num_filters_out,
                             r=('required-time-offsets=0' if allow_zero_padding else ''),
                             o=misc_conv_opts))

            configs.append('component name={0}.conv{1} type=TimeHeightConvolutionComponent '
                           '{2}'.format(name, n, conv_opts))
            configs.append('component-node name={0}.conv{1} component={0}.conv{1} '
                           'input={2}'.format(name, n, cur_descriptor))
            cur_descriptor = '{0}.conv{1}'.format(name, n)

            cur_num_filters = num_filters_out
            cur_height = height_out
            cur_time_period = time_period_out

            # the batch-norm
            configs.append('component name={0}.batchnorm{1}  type=BatchNormComponent dim={2} '
                               'block-dim={3}'.format(
                                   name, n, cur_num_filters * cur_height,
                                   cur_num_filters))
            configs.append('component-node name={0}.batchnorm{1} component={0}.batchnorm{1} '
                           'input={2}'.format(name, n, cur_descriptor))
            cur_descriptor = '{0}.batchnorm{1}'.format(name, n)

            # the scale-and-offset
            configs.append('component name={0}.scaleoffset{1}  type=ScaleAndOffsetComponent dim={2} '
                               'block-dim={3}'.format(
                                   name, n, cur_num_filters * cur_height,
                                   cur_num_filters))
            configs.append('component-node name={0}.scaleoffset{1} component={0}.scaleoffset{1} '
                           'input={2}'.format(name, n, cur_descriptor))
            cur_descriptor = '{0}.scaleoffset{1}'.format(name, n)


            if n == 2:
                # the bypass connection
                cur_descriptor = 'Sum({0}, {1})'.format(cur_descriptor, bypass_descriptor)


            # the ReLU
            configs.append('component name={0}.relu{1} type=RectifiedLinearComponent '
                           'dim={2} block-dim={3} self-repair-scale={4} '
                           'self-repair-lower-threshold={5}'.format(
                               name, n, cur_num_filters * cur_height, cur_num_filters,
                               self.config['self-repair-scale'],
                               self.config['self-repair-lower-threshold{0}'.format(n)]))
            configs.append('component-node name={0}.relu{1} component={0}.relu{1} '
                           'input={2}'.format(name, n, cur_descriptor))

            cur_descriptor = '{0}.relu{1}'.format(name, n)

        if bypass_descriptor != input_descriptor:
            # We need to add the 1x1 bypass convolution because we're either doing height
            # subsampling or changing the number of filters.
            conv_opts = ('height-in={hi} height-out={ho} height-offsets=0 '
                         'time-offsets=0 height-subsample-out={hs} '
                         'num-filters-in={fi} num-filters-out={fo} {o}'.format(
                             hi=height_in, ho=height_out, hs=height_subsample,
                             fi=num_filters_in, fo=num_filters_out, o=misc_conv_opts))
            configs.append('component name={0}.conv_bypass type=TimeHeightConvolutionComponent '
                           '{1}'.format(name, conv_opts))
            configs.append('component-node name={0}.conv_bypass component={0}.conv_bypass '
                           'input={1}'.format(name, input_descriptor))



        # Note: the function 'output_name' is responsible for returning the
        # descriptor corresponding to the output of the network, which in
        # this case would be '{0}.relu2'.format(name).
        return configs


    # _generate_bottleneck_resblock_config is a convenience function to generate the
    # res-block config (this is the bottleneck version, where there is
    # a 3x3 kernel with a smaller number of filters than at the input and output,
    # sandwiched between two 1x1 kernels.
    #
    # The main path inside the res-block in the bottleneck case is as follows:
    #
    # input -> conv1 -> batchnorm1 -> scaleoffset1 -> relu1 ->
    #          conv2 -> batchnorm2 -> scaleoffset2 -> relu2 ->
    #          conv3 -> batchnorm3 -> scaleoffset3 -> relu3
    #
    #  but the relu3 takes as its input the sum of 'input' and 'scaleoffset3'.
    #
    def _generate_bottleneck_resblock_config(self):
        configs = []

        name = self.name
        num_bottleneck_filters = self.config['num-bottleneck-filters']
        assert num_bottleneck_filters > 0
        input_dim = self.descriptors['input']['dim']
        height_in = self.config['height-in']
        height_out = self.config['height-out']
        input_descriptor = self.descriptors['input']['final-string']
        allow_zero_padding = self.config['allow-zero-padding']
        time_period_out = self.config['time-period']
        if not input_dim % height_in == 0:
            raise RuntimeError("input-dim={0} does not divide height-in={1}".format(
                input_dim, height_in))
        num_filters_in = input_dim / height_in
        num_filters_out = self.config['num-filters']

        if height_out != height_in:
            if height_out < height_in / 2 - 1 or height_out > height_in /  2 + 1:
                raise RuntimeError("Expected height-out to be about half height-in, or the same: "
                                   "height-in={0} height-out={1}".format(height_in, height_out))
            height_subsample = 2
        else:
            height_subsample = 1

        cur_descriptor = input_descriptor
        cur_num_filters = num_filters_in
        cur_height = height_in
        if height_subsample == 1 and num_filters_in == num_filters_out:
            bypass_descriptor = input_descriptor
        else:
            bypass_descriptor = '{0}.conv_bypass'.format(name)

        # get miscellaneous convolution options passed in from the xconfig line
        a = []
        for opt_name in [
                'param-stddev', 'bias-stddev', 'use-natural-gradient',
                'max-change', 'rank-in', 'rank-out', 'num-minibatches-history',
                'alpha-in', 'alpha-out', 'l2-regularize' ]:
            value = self.config[opt_name]
            if value != '':
                a.append('{0}={1}'.format(opt_name, value))
        misc_conv_opts = ' '.join(a)


        for n in [1, 2, 3]:
            # the convolution.
            height_offsets = ('-1,0,1' if n == 2 else '0')
            this_height_subsample = height_subsample if n == 1 else 1
            time_offsets = ('-{t},0,{t}'.format(t=time_period_out) if n == 2 else '0')
            next_num_filters = (num_filters_out if n == 3 else num_bottleneck_filters)

            conv_opts = ('height-in={h_in} height-out={h_out} height-offsets={ho} time-offsets={to} '
                         'num-filters-in={fi} num-filters-out={fo} height-subsample-out={hs} '
                         '{r} {o}'.format(
                             h_in=cur_height, h_out=height_out,
                             to=time_offsets, ho=height_offsets,
                             hs=this_height_subsample,
                             fi=cur_num_filters, fo=next_num_filters,
                             r=('required-time-offsets=0' if allow_zero_padding else ''),
                             o=misc_conv_opts))

            configs.append('component name={0}.conv{1} type=TimeHeightConvolutionComponent '
                           '{2}'.format(name, n, conv_opts))
            configs.append('component-node name={0}.conv{1} component={0}.conv{1} '
                           'input={2}'.format(name, n, cur_descriptor))

            cur_num_filters = next_num_filters
            cur_height = height_out
            cur_descriptor = '{0}.conv{1}'.format(name, n)

            # the batch-norm
            configs.append('component name={0}.batchnorm{1}  type=BatchNormComponent dim={2} '
                               'block-dim={3}'.format(
                                   name, n, cur_num_filters * cur_height,
                                   cur_num_filters))
            configs.append('component-node name={0}.batchnorm{1} component={0}.batchnorm{1} '
                           'input={2}'.format(name, n, cur_descriptor))
            cur_descriptor = '{0}.batchnorm{1}'.format(name, n)

            # the scale and offset
            configs.append('component name={0}.scaleoffset{1}  type=ScaleAndOffsetComponent dim={2} '
                               'block-dim={3}'.format(
                                   name, n, cur_num_filters * cur_height,
                                   cur_num_filters))
            configs.append('component-node name={0}.scaleoffset{1} component={0}.scaleoffset{1} '
                           'input={2}'.format(name, n, cur_descriptor))
            cur_descriptor = '{0}.scaleoffset{1}'.format(name, n)

            if n == 3:
                # the bypass connection
                cur_descriptor = 'Sum({0}, {1})'.format(cur_descriptor, bypass_descriptor)

            # the ReLU
            configs.append('component name={0}.relu{1} type=RectifiedLinearComponent '
                           'dim={2} block-dim={3} self-repair-scale={4} '
                           'self-repair-lower-threshold={5}'.format(
                               name, n, cur_num_filters * cur_height, cur_num_filters,
                               self.config['self-repair-scale'],
                               self.config['self-repair-lower-threshold{0}'.format(n)]))
            configs.append('component-node name={0}.relu{1} component={0}.relu{1} '
                           'input={2}'.format(name, n, cur_descriptor))

            cur_descriptor = '{0}.relu{1}'.format(name, n)

        if bypass_descriptor != input_descriptor:
            # We need to add the 1x1 bypass convolution because we're either doing height
            # subsampling or changing the number of filters.
            conv_opts = ('height-in={hi} height-out={ho} height-offsets=0 '
                         'time-offsets=0 height-subsample-out={hs} '
                         'num-filters-in={fi} num-filters-out={fo} {o}'.format(
                             hi=height_in, ho=height_out, hs=height_subsample,
                             fi=num_filters_in, fo=num_filters_out, o=misc_conv_opts))
            configs.append('component name={0}.conv_bypass type=TimeHeightConvolutionComponent '
                           '{1}'.format(name, conv_opts))
            configs.append('component-node name={0}.conv_bypass component={0}.conv_bypass '
                           'input={1}'.format(name, input_descriptor))

        # Note: the function 'output_name' is responsible for returning the
        # descriptor corresponding to the output of the network, which
        # in this case will be '{0}.relu3'.format(name).
        return configs


# This layer just maps to a single component, a SumBlockComponent.  It's for
# doing channel averaging at the end of neural networks.  See scripts for
# examples of how to use it.
# An example line using this layer is:
# channel-average-layer name=channel-average input=Append(2, 4, 6, 8) dim=64

# the configuration value 'dim' is the output dimension of this layer.
# The input dimension is expected to be a multiple of 'dim'.  The output
# will be the average of 'dim'-sized blocks of the input.
class ChannelAverageLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "channel-average-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input':'[-1]',
                       'dim': -1 }

    def set_derived_configs(self):
        pass

    def check_configs(self):
        input_dim = self.descriptors['input']['dim']
        dim = self.config['dim']
        if dim <= 0:
            raise RuntimeError("dim must be specified and > 0.")
        if input_dim % dim != 0:
            raise RuntimeError("input-dim={0} is not a multiple of dim={1}".format(
                input_dim, dim))

    def auxiliary_outputs(self):
        return []

    def output_name(self, auxiliary_output = None):
        assert auxiliary_output is None
        return self.name

    def output_dim(self, auxiliary_output = None):
        assert auxiliary_output is None
        return self.config['dim']


    def get_full_config(self):
        ans = []
        config_lines = self._generate_channel_average_config()
        for line in config_lines:
            for config_name in ['ref', 'final']:
                ans.append((config_name, line))
        return ans

    def _generate_channel_average_config(self):
        configs = []
        name = self.name
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        dim = self.config['dim']
        # choose the scale that makes it an average rather than a sum.
        scale = dim * 1.0 / input_dim
        configs.append('component name={0} type=SumBlockComponent input-dim={1} '
                       'output-dim={2} scale={3}'.format(name, input_dim,
                                                         dim, scale))
        configs.append('component-node name={0} component={0} input={1}'.format(
            name, input_descriptor))
        return configs
