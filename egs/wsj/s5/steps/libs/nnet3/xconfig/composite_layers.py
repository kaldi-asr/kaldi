# Copyright 2018    Johns Hopkins University (Dan Povey)
# Apache 2.0.

""" This module contains some composite layers, which is basically a catch-all
    term for things like TDNN-F that contain several affine or linear comopnents.
"""
from __future__ import print_function
import math
import re
import sys
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase

# This class is intended to implement an extension of the factorized TDNN
# (TDNN-F) that supports resnet-type 'bypass' connections.  It is for lines like
# the following:
#
# tdnnf-layer name=tdnnf2 dim=1024 bottleneck-dim=128 dropout-proportion=0.0 time-stride=3
#
# The line above would be roughly equivalent to the following four lines (except
# for different naming, and the use of TdnnComponent, for efficiency, in place
# of AffineComponent).  Assume that the previous layer (the default input) was tdnnf1:
#
#  linear-component name=tdnnf2.linear dim=128 orthonormal-constraint=-1.0 input=Append(Offset(-3, tdnnf1), tdnnf1)
#  relu-batchnorm-dropout-layer name=tdnnf2.affine dim=1024 dropout-proportion=0.0 \
#    dropout-per-dim-continuous=true input=Append(0,3)
#  no-op-component name=tdnnf2 input=Sum(Scale(0.66,tdnnf1), tdnn2.affine)

#  Documentation of some of the important options:
#
#   - dropout-proportion
# This gets passed through to the dropout component.  If you don't set
# 'dropout-proportion', no dropout component will be included; it would be like
# using a relu-batchnorm-layer in place of a relu-batchnorm-dropout-layer.  You
# should only set 'dropout-proportion' if you intend to use dropout (it would
# usually be combined with the --dropout-schedule option to train.py).  If you
# use the --dropout-schedule option, the value doesn't really matter since it
# will be changed during training, and 0 is recommended.
#
#  - time-stride
# Controls the time offsets in the splicing, e.g. if you set time-stride to
# 1 instead of the 3 in the example, the time-offsets would be -1 and 1 instead
# of 1 and 3.
# If you set time-stride=0, as a special case no splicing over time will be
# performed (so no Append() expressions) and the second linear component (named
# tdnnf2l in the example) would be omitted, since it would add no modeling
# power.
# You can set time-stride to a negative number which will negate all the
# time indexes; it might potentially be useful to alternate negative and positive
# time-stride if you wanted to force the overall network to have symmetric
# context, since with positive time stride, this layer has more negative
# than positive time context (i.e. more left than right).
#
#  - bypass-scale

# A scale on the previous layer's output, used in bypass (resnet-type)
# connections.  Should not exceed 1.0.  The default is 0.66.  If you set it to
# zero, the layer will lack the bypass (but we don't recommend this).  won't use
# a bypass connection at all, so it would be like conventional TDNN-F Note: the
# layer outputs are added together after the batchnorm so the model cannot
# control their relative magnitudes and this does actually affect what it can
# model.  When we experimented with having this scale trainable it did not seem
# to give an advantage.
#
#  - l2-regularize
# This is passed through to the linear and affine components.  You'll normally
# want this to be set to a nonzero value, e.g. 0.004.

class XconfigTdnnfLayer(XconfigLayerBase):

    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "tdnnf-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input':'[-1]',
                       'dim':-1,
                       'bottleneck-dim':-1,
                       'bypass-scale':0.66,
                       'dropout-proportion':-1.0,
                       'time-stride':1,
                       'l2-regularize':0.0,
                       'max-change': 0.75,
                       'self-repair-scale': 1.0e-05,
                       'context': 'default'}

    def set_derived_configs(self):
        pass

    def check_configs(self):
        if self.config['bottleneck-dim'] <= 0:
            raise RuntimeError("bottleneck-dim must be set and >0.")
        if self.config['dim'] <= self.config['bottleneck-dim']:
            raise RuntimeError("dim must be greater than bottleneck-dim")

        dropout = self.config['dropout-proportion']
        if dropout != -1.0 and not (dropout >= 0.0 and dropout < 1.0):
            raise RuntimeError("invalid value for dropout-proportion")

        if abs(self.config['bypass-scale']) > 1.0:
            raise RuntimeError("bypass-scale has invalid value")

        input_dim = self.descriptors['input']['dim']
        output_dim = self.config['dim']
        if output_dim != input_dim and self.config['bypass-scale'] != 0.0:
            raise RuntimeError('bypass-scale is nonzero but output-dim != input-dim: {0} != {1}'
                               ''.format(output_dim, input_dim))

        if not self.config['context'] in ['default', 'left-only', 'shift-left', 'none']:
            raise RuntimeError('context must be default, left-only shift-left or none, got {}'.format(
                self.config['context']))


    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        output_component = ''
        if self.config['bypass-scale'] != 0.0:
            # the no-op component is used to cache something that we don't want
            # to have to recompute.
            output_component = 'noop'
        elif self.config['dropout-proportion'] != -1.0:
            output_component = 'dropout'
        else:
            output_component = 'batchnorm'
        return '{0}.{1}'.format(self.name, output_component)


    def output_dim(self, auxiliary_output=None):
        return self.config['dim']

    def get_full_config(self):
        ans = []
        config_lines = self._generate_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                ans.append((config_name, line))
        return ans


    def _generate_config(self):
        configs = []
        name = self.name
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        output_dim = self.config['dim']
        bottleneck_dim = self.config['bottleneck-dim']
        bypass_scale = self.config['bypass-scale']
        dropout_proportion = self.config['dropout-proportion']
        time_stride = self.config['time-stride']
        context = self.config['context']
        if time_stride != 0 and context != 'none':
            time_offsets1 = '{0},0'.format(-time_stride)
            if context == 'default':
                time_offsets2 = '0,{0}'.format(time_stride)
            elif context == 'shift-left':
                time_offsets2 = '{0},0'.format(-time_stride)
            else:
                assert context == 'left-only'
                time_offsets2 = '0'
        else:
            time_offsets1 = '0'
            time_offsets2 = '0'
        l2_regularize = self.config['l2-regularize']
        max_change = self.config['max-change']
        self_repair_scale = self.config['self-repair-scale']

        # The first linear layer, from input-dim (spliced x2) to bottleneck-dim
        configs.append('component name={0}.linear type=TdnnComponent input-dim={1} '
                       'output-dim={2} l2-regularize={3} max-change={4} use-bias=false '
                       'time-offsets={5} orthonormal-constraint=-1.0'.format(
                           name, input_dim, bottleneck_dim, l2_regularize,
                           max_change, time_offsets1))
        configs.append('component-node name={0}.linear component={0}.linear '
                       'input={1}'.format(name, input_descriptor))

        # The affine layer, from bottleneck-dim (spliced x2) to output-dim
        configs.append('component name={0}.affine type=TdnnComponent '
                       'input-dim={1} output-dim={2} l2-regularize={3} max-change={4} '
                       'time-offsets={5}'.format(
                           name, bottleneck_dim, output_dim, l2_regularize,
                           max_change, time_offsets2))
        configs.append('component-node name={0}.affine component={0}.affine '
                       'input={0}.linear'.format(name))

        # The ReLU layer
        configs.append('component name={0}.relu type=RectifiedLinearComponent dim={1} '
                       'self-repair-scale={2}'.format(
                           name, output_dim, self_repair_scale))
        configs.append('component-node name={0}.relu component={0}.relu '
                       'input={0}.affine'.format(name))

        # The BatchNorm layer
        configs.append('component name={0}.batchnorm type=BatchNormComponent '
                       'dim={1}'.format(name, output_dim))
        configs.append('component-node name={0}.batchnorm component={0}.batchnorm '
                       'input={0}.relu'.format(name))

        if dropout_proportion != -1:
            # This is not normal dropout.  It's dropout where the mask is shared
            # across time, and (thanks to continuous=true), instead of a
            # zero-or-one scale, it's a continuously varying scale whose
            # expected value is 1, drawn from a uniform distribution over an
            # interval of a size that varies with dropout-proportion.
            configs.append('component name={0}.dropout type=GeneralDropoutComponent '
                           'dim={1} dropout-proportion={2} continuous=true'.format(
                               name, output_dim, dropout_proportion))
            configs.append('component-node name={0}.dropout component={0}.dropout '
                           'input={0}.batchnorm'.format(name))
            cur_component_type = 'dropout'
        else:
            cur_component_type = 'batchnorm'

        if bypass_scale != 0.0:
            # Add a NoOpComponent to cache the weighted sum of the input and the
            # output.  We could easily have the output of the component be a
            # Descriptor like 'Append(Scale(0.66, tdnn1.batchnorm), tdnn2.batchnorm)',
            # but if we did that and you used many of this component in sequence,
            # the weighted sums would have more and more terms as you went deeper
            # in the network.
            configs.append('component name={0}.noop type=NoOpComponent '
                           'dim={1}'.format(name, output_dim))
            configs.append('component-node name={0}.noop component={0}.noop '
                           'input=Sum(Scale({1}, {2}), {0}.{3})'.format(
                               name, bypass_scale, input_descriptor,
                               cur_component_type))

        return configs

# This is for lines like the following:
#  prefinal-layer name=prefinal-chain input=prefinal-l l2-regularize=0.02 big-dim=1024 small-dim=256
#
# which is equivalent to the following sequence of components (except for
# name differences):
#  relu-batchnorm-layer name=prefinal-chain input=prefinal-l l2-regularize=0.02 dim=1024
#  linear-comonent name=prefinal-chain-l dim=256 l2-regularize=0.02 orthonormal-constraint=-1.0
#  batchnorm-component name=prefinal-chain-batchnorm
#
# This layer is really just for convenience in writing config files: it doesn't
# do anything that's particular hard or unusual, but it encapsulates a commonly
# repeated pattern.
class XconfigPrefinalLayer(XconfigLayerBase):

    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "prefinal-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input':'[-1]',
                       'big-dim':-1,
                       'small-dim':-1,
                       'l2-regularize':0.0,
                       'max-change': 0.75,
                       'self-repair-scale': 1.0e-05}

    def set_derived_configs(self):
        pass

    def check_configs(self):
        if self.config['small-dim'] <= 0:
            raise RuntimeError("small-dim must be set and >0.")
        if self.config['big-dim'] <= self.config['small-dim']:
            raise RuntimeError("big-dim must be greater than small-dim")

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        return '{0}.batchnorm2'.format(self.name)

    def output_dim(self, auxiliary_output=None):
        return self.config['small-dim']

    def get_full_config(self):
        ans = []
        config_lines = self._generate_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                ans.append((config_name, line))
        return ans


    def _generate_config(self):
        configs = []
        name = self.name

        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        small_dim = self.config['small-dim']
        big_dim = self.config['big-dim']
        l2_regularize = self.config['l2-regularize']
        max_change = self.config['max-change']
        self_repair_scale = self.config['self-repair-scale']

        # The affine layer, from input-dim to big-dim.
        configs.append('component name={0}.affine type=NaturalGradientAffineComponent '
                       'input-dim={1} output-dim={2} l2-regularize={3} max-change={4}'.format(
                           name, input_dim, big_dim, l2_regularize, max_change))
        configs.append('component-node name={0}.affine component={0}.affine '
                       'input={1}'.format(name, input_descriptor))

        # The ReLU layer
        configs.append('component name={0}.relu type=RectifiedLinearComponent dim={1} '
                       'self-repair-scale={2}'.format(
                           name, big_dim, self_repair_scale))
        configs.append('component-node name={0}.relu component={0}.relu '
                       'input={0}.affine'.format(name))

        # The first BatchNorm layer
        configs.append('component name={0}.batchnorm1 type=BatchNormComponent '
                       'dim={1}'.format(name, big_dim))
        configs.append('component-node name={0}.batchnorm1 component={0}.batchnorm1 '
                       'input={0}.relu'.format(name))

        # The linear layer, from big-dim to small-dim, with orthonormal-constraint=-1
        # ("floating" orthonormal constraint).
        configs.append('component name={0}.linear type=LinearComponent '
                       'input-dim={1} output-dim={2} l2-regularize={3} max-change={4} '
                       'orthonormal-constraint=-1 '.format(
                           name, big_dim, small_dim,
                           l2_regularize, max_change))
        configs.append('component-node name={0}.linear component={0}.linear '
                       'input={0}.batchnorm1'.format(name))

        # The second BatchNorm layer
        configs.append('component name={0}.batchnorm2 type=BatchNormComponent '
                       'dim={1}'.format(name, small_dim))
        configs.append('component-node name={0}.batchnorm2 component={0}.batchnorm2 '
                       'input={0}.linear'.format(name))

        return configs
