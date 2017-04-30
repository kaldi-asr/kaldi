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
#  conv-normalize-layer name=conv3 height-in=40 height-out=20 \
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

class XconfigConvLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        for operation in first_token.split('-')[:-1]:
            assert operation in ['conv', 'renorm', 'batchnorm', 'relu', 'dropout']
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
                       # the following are not really inspected by this level of
                       # code, just passed through (but not if left at '').
                       'param-stddev':'', 'bias-stddev':'',
                       'max-change': 0.75, 'init-orthogonal': 'false',
                       'learning-rate-factor':'',
                       'use-natural-gradient':'',
                       'rank-in':'', 'rank-out':'', 'num-minibatches-history':'',
                       'alpha-in':'', 'alpha-out':'',
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
        assert len(operations) > 1
        last_operation = operations[-1]
        assert last_operation in ['relu', 'conv',
                                  'renorm', 'batchnorm', 'dropout']
        # we'll return something like 'layer1.batchnorm'.
        return '{0}.{1}'.format(self.name, last_operation)

    def output_dim(self, auxiliary_output = None):
        assert auxiliary_output is None
        return self.config['num-filters-out'] * self.config['height-out']

    def get_full_config(self):
        ans = []
        config_lines = self.generate_cnn_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in CNN initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # convenience function to generate the CNN config
    def generate_cnn_config(self):
        configs = []

        name = self.name

        # These 3 variables will be updated as we add components.
        cur_num_filters = self.config['num-filters-in']
        cur_height = self.config['height-in']
        cur_descriptor = self.descriptors['input']['final-string']

        # note: the [:-1] is to remove the '-layer'.
        operations = self.layer_type.split('-')[:-1]
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
                        'learning-rate-factor', 'init-orthogonal']:
                    value = self.config[opt_name]
                    if value != '':
                        a.append('{0}={1}'.format(opt_name, value))
                conv_opts = ' '.join(a)

                configs.append("### Begin convolutional layer '{0}'".format(name))
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
                           'dim={1} self-repair-scale={2}'.format(
                               name, cur_num_filters * cur_height,
                               self.config['self-repair-scale']))
                configs.append('component-node name={0}.relu component={0}.relu '
                               'input={1}'.format(name, cur_descriptor))
            elif operation == 'dropout':
                configs.append('component name={0}.dropout type=DropoutComponent '
                           'dim={1} dropout-proportion={2}'.format(
                               name, cur_num_filters * cur_height,
                               self.config['dropout-proportion']))
                configs.append('component-node name={0}.dropout component={0}.dropout '
                               'input={1}'.format(name, cur_descriptor))
            else:
                raise RuntimeError("Un-handled operation type: " + operation)

            cur_descriptor = '{0}.{1}'.format(name, operation)

        return configs
