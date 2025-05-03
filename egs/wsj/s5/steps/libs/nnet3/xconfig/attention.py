# Copyright 2017    Johns Hopkins University (Dan Povey)
#           2017    Hossein Hadian
# Apache 2.0.

""" This module has the implementation of attention layers.
"""

from __future__ import print_function
from __future__ import division
import math
import re
import sys
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase

# This class is for parsing lines like
#  'attention-renorm-layer num-heads=10 value-dim=50 key-dim=50 time-stride=3 num-left-inputs=5 num-right-inputs=2.'
#
# Parameters of the class, and their defaults:
#   input='[-1]'               [Descriptor giving the input of the layer.]
#   self-repair-scale=1.0e-05  [Affects relu, sigmoid and tanh layers.]
#   learning-rate-factor=1.0   [This can be used to make the affine component
#                               train faster or slower].
#   Documentation for the rest of the parameters (related to the
#   attention component) can be found in nnet-attention-component.h


class XconfigAttentionLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        # Here we just list some likely combinations.. you can just add any
        # combinations you want to use, to this list.
        assert first_token in ['attention-renorm-layer',
                               'attention-relu-renorm-layer',
                               'attention-relu-batchnorm-layer',
                               'relu-renorm-attention-layer']
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = { 'input':'[-1]',
                        'max-change' : 0.75,
                        'self-repair-scale' : 1.0e-05,
                        'target-rms' : 1.0,
                        'learning-rate-factor' : 1.0,
                        'ng-affine-options' : '',
                        'l2-regularize': 0.0,
                        'num-left-inputs-required': -1,
                        'num-right-inputs-required': -1,
                        'output-context': True,
                        'time-stride': 1,
                        'num-heads': 1,
                        'key-dim': -1,
                        'key-scale': 0.0,
                        'value-dim': -1,
                        'num-left-inputs': -1,
                        'num-right-inputs': -1,
                        'dropout-proportion': 0.5}  # dropout-proportion only
                                                    # affects layers with
                                                    # 'dropout' in the name.

    def check_configs(self):
        if self.config['self-repair-scale'] < 0.0 or self.config['self-repair-scale'] > 1.0:
            raise RuntimeError("self-repair-scale has invalid value {0}"
                               .format(self.config['self-repair-scale']))
        if self.config['target-rms'] < 0.0:
            raise RuntimeError("target-rms has invalid value {0}"
                               .format(self.config['target-rms']))
        if self.config['learning-rate-factor'] <= 0.0:
            raise RuntimeError("learning-rate-factor has invalid value {0}"
                               .format(self.config['learning-rate-factor']))
        for conf in ['value-dim', 'key-dim',
                     'num-left-inputs', 'num-right-inputs']:
            if self.config[conf] < 0:
                raise RuntimeError("{0} has invalid value {1}"
                                   .format(conf, self.config[conf]))
        if self.config['key-scale'] == 0.0:
            self.config['key-scale'] = 1.0 / math.sqrt(self.config['key-dim'])

    def output_name(self, auxiliary_output=None):
        # at a later stage we might want to expose even the pre-nonlinearity
        # vectors
        assert auxiliary_output == None

        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        last_nonlinearity = split_layer_name[-2]
        # return something like: layer3.renorm
        return '{0}.{1}'.format(self.name, last_nonlinearity)

    def attention_input_dim(self):
        context_dim = (self.config['num-left-inputs'] +
                       self.config['num-right-inputs'] + 1)
        num_heads = self.config['num-heads']
        key_dim = self.config['key-dim']
        value_dim = self.config['value-dim']
        query_dim = key_dim + context_dim;
        return num_heads * (key_dim + value_dim + query_dim)

    def attention_output_dim(self):
        context_dim = (self.config['num-left-inputs'] +
                       self.config['num-right-inputs'] + 1)
        num_heads = self.config['num-heads']
        value_dim = self.config['value-dim']
        return (num_heads *
                (value_dim +
                 (context_dim if self.config['output-context'] else 0)))

    def output_dim(self, auxiliary_output = None):
      return self.attention_output_dim()

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
        dim = self.attention_input_dim()
        self_repair_scale = self.config['self-repair-scale']
        target_rms = self.config['target-rms']
        max_change = self.config['max-change']
        ng_affine_options = self.config['ng-affine-options']
        l2_regularize = self.config['l2-regularize']
        learning_rate_factor=self.config['learning-rate-factor']
        learning_rate_option=('learning-rate-factor={0}'.format(learning_rate_factor)
                              if learning_rate_factor != 1.0 else '')
        l2_regularize_option = ('l2-regularize={0} '.format(l2_regularize)
                                if l2_regularize != 0.0 else '')
        configs = []
        # First the affine node.
        line = ('component name={0}.affine'
                ' type=NaturalGradientAffineComponent'
                ' input-dim={1}'
                ' output-dim={2}'
                ' max-change={3}'
                ' {4} {5} {6}'
                ''.format(self.name, input_dim, dim,
                          max_change, ng_affine_options,
                          learning_rate_option, l2_regularize_option))
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
                        ''.format(self.name, nonlinearity, dim,
                            self_repair_scale))

            elif nonlinearity == 'attention':
                line = ('component name={0}.{1}'
                        ' type=RestrictedAttentionComponent'
                        ' value-dim={2}'
                        ' key-dim={3}'
                        ' num-left-inputs={4}'
                        ' num-right-inputs={5}'
                        ' num-left-inputs-required={6}'
                        ' num-right-inputs-required={7}'
                        ' output-context={8}'
                        ' time-stride={9}'
                        ' num-heads={10}'
                        ' key-scale={11}'
                        ''.format(self.name, nonlinearity,
                                  self.config['value-dim'],
                                  self.config['key-dim'],
                                  self.config['num-left-inputs'],
                                  self.config['num-right-inputs'],
                                  self.config['num-left-inputs-required'],
                                  self.config['num-right-inputs-required'],
                                  self.config['output-context'],
                                  self.config['time-stride'],
                                  self.config['num-heads'],
                                  self.config['key-scale']))
                dim = self.attention_output_dim()

            elif nonlinearity == 'sigmoid':
                line = ('component name={0}.{1}'
                        ' type=SigmoidComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, nonlinearity, dim,
                            self_repair_scale))

            elif nonlinearity == 'tanh':
                line = ('component name={0}.{1}'
                        ' type=TanhComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, nonlinearity, dim,
                            self_repair_scale))

            elif nonlinearity == 'renorm':
                line = ('component name={0}.{1}'
                        ' type=NormalizeComponent dim={2}'
                        ' target-rms={3}'
                        ''.format(self.name, nonlinearity, dim,
                            target_rms))

            elif nonlinearity == 'batchnorm':
                line = ('component name={0}.{1}'
                        ' type=BatchNormComponent dim={2}'
                        ' target-rms={3}'
                        ''.format(self.name, nonlinearity, dim,
                            target_rms))

            elif nonlinearity == 'dropout':
                line = ('component name={0}.{1} type=DropoutComponent '
                           'dim={2} dropout-proportion={3}'.format(
                               self.name, nonlinearity, dim,
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

class XconfigResidualAttentionLayer(XconfigLayerBase):
    # This is just multi-head attention followed by batch-norm and then
    # projection back to the input dim, another batchorm, then a bypass
    # connection with the input.
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == 'residual-attention-layer'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = { 'input':'[-1]',
                        'max-change' : 0.75,
                        'self-repair-scale' : 1.0e-05,
                        'learning-rate-factor' : 1.0,
                        'l2-regularize': 0.0,
                        'num-left-inputs-required': -1,
                        'num-right-inputs-required': -1,
                        'output-context': True,
                        'time-stride': 1,
                        'input-splicing': 0, # splicing at input to attention, e.g. -1,0,1.
                        'num-heads': 8,
                        'key-dim': 64,
                        'key-scale': 0.0,
                        'value-dim': 64,
                        'bypass-scale': 0.66,
                        'num-left-inputs': -1,
                        'num-right-inputs': -1,
                        'dropout-proportion': -1.0}  # If >= 0, will use dropout.  (Note: you prob. want to use a dropout schedule.)


    def check_configs(self):
        if self.config['self-repair-scale'] < 0.0 or self.config['self-repair-scale'] > 1.0:
            raise RuntimeError("self-repair-scale has invalid value {0}"
                               .format(self.config['self-repair-scale']))
        if self.config['learning-rate-factor'] <= 0.0:
            raise RuntimeError("learning-rate-factor has invalid value {0}"
                               .format(self.config['learning-rate-factor']))
        for conf in ['value-dim', 'key-dim',
                     'num-left-inputs', 'num-right-inputs',
                     'bypass-scale']:
            if self.config[conf] < 0:
                raise RuntimeError("{0} has invalid value {1}"
                                   .format(conf, self.config[conf]))
        if self.config['key-scale'] == 0.0:
            self.config['key-scale'] = 1.0 / math.sqrt(self.config['key-dim'])

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output == None
        return '{0}.noop'.format(self.name)

    def attention_input_dim(self):
        context_dim = (self.config['num-left-inputs'] +
                       self.config['num-right-inputs'] + 1)
        num_heads = self.config['num-heads']
        key_dim = self.config['key-dim']
        value_dim = self.config['value-dim']
        query_dim = key_dim + context_dim;
        return num_heads * (key_dim + value_dim + query_dim)

    def attention_output_dim(self):
        context_dim = (self.config['num-left-inputs'] +
                       self.config['num-right-inputs'] + 1)
        num_heads = self.config['num-heads']
        value_dim = self.config['value-dim']
        return (num_heads *
                (value_dim +
                 (context_dim if self.config['output-context'] else 0)))

    def output_dim(self, auxiliary_output = None):
        input_dim = self.descriptors['input']['dim']
        return input_dim

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
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        output_dim = input_dim
        attention_input_dim = self.attention_input_dim()
        attention_output_dim = self.attention_output_dim()
        self_repair_scale = self.config['self-repair-scale']
        bypass_scale = self.config['bypass-scale']
        max_change = self.config['max-change']
        l2_regularize = self.config['l2-regularize']
        dropout_proportion = self.config['dropout-proportion']
        input_splicing = self.config['input-splicing']
        learning_rate_factor=self.config['learning-rate-factor']
        learning_rate_option=('learning-rate-factor={0}'.format(learning_rate_factor)
                              if learning_rate_factor != 1.0 else '')
        l2_regularize_option = ('l2-regularize={0} '.format(l2_regularize)
                                if l2_regularize != 0.0 else '')
        configs = []
        # First the affine or TDNN layer... you can consider this as the
        # parameters of the attention component
        line = ('component name={0}.affine1 type=TdnnComponent input-dim={1} output-dim={2}'
                ' max-change={3} time-offsets={4} {5} {6}'
                ''.format(self.name, input_dim, attention_input_dim,
                          max_change, input_splicing,
                          learning_rate_option, l2_regularize_option))
        configs.append(line)

        line = ('component-node name={0}.affine1 component={0}.affine1 input={1}'
                ''.format(self.name, input_desc))
        configs.append(line)

        line = ('component name={0}.batchnorm1 type=BatchNormComponent dim={1}'
                ''.format(self.name, attention_input_dim))
        configs.append(line)
        line = ('component-node name={0}.batchnorm1 component={0}.batchnorm1 '
                'input={0}.affine1'.format(self.name, input_desc))
        configs.append(line)
        # We have batchnorm AND layer norm.  Batchnorm ensures each dim has
        # the same dynamic range; layer norm ensures each vector (each time step)
        # does.
        line = ('component name={0}.layernorm1 type=NormalizeComponent dim={1}'
                ''.format(self.name, attention_input_dim))
        configs.append(line)
        line = ('component-node name={0}.layernorm1 component={0}.layernorm1 '
                'input={0}.batchnorm1'.format(self.name, input_desc))
        configs.append(line)

        # ... then a scale-and-offset component for generality.
        line = ('component name={0}.scale_offset1 type=ScaleAndOffsetComponent dim={1} '
                ''.format(self.name, attention_input_dim))
        configs.append(line)
        line = ('component-node name={0}.scale_offset1 component={0}.scale_offset1 '
                'input={0}.layernorm1'.format(self.name, input_desc))
        configs.append(line)

        line = ('component name={0}.attention type=RestrictedAttentionComponent '
                ' value-dim={1} key-dim={2} num-left-inputs={3}'
                ' num-right-inputs={4} num-left-inputs-required={5}'
                ' num-right-inputs-required={6} output-context={7}'
                ' time-stride={8} num-heads={9} key-scale={10}'
                ''.format(self.name,
                          self.config['value-dim'],
                          self.config['key-dim'],
                          self.config['num-left-inputs'],
                          self.config['num-right-inputs'],
                          self.config['num-left-inputs-required'],
                          self.config['num-right-inputs-required'],
                          self.config['output-context'],
                          self.config['time-stride'],
                          self.config['num-heads'],
                          self.config['key-scale']))
        configs.append(line)
        line = ('component-node name={0}.attention component={0}.attention '
                'input={0}.scale_offset1'.format(self.name, input_desc))
        configs.append(line)

        line = ('component name={0}.affine2 type=TdnnComponent input-dim={1} output-dim={2}'
                ' max-change={3} time-offsets=0 {4} {5}'
                ''.format(self.name, attention_output_dim, output_dim,
                          max_change, learning_rate_option, l2_regularize_option))
        configs.append(line)
        line = ('component-node name={0}.affine2 component={0}.affine2 '
                'input={0}.attention'.format(self.name))
        configs.append(line)

        line = ('component name={0}.batchnorm2 type=BatchNormComponent dim={1}'
                ''.format(self.name, output_dim))
        configs.append(line)
        line = ('component-node name={0}.batchnorm2 component={0}.batchnorm2 '
                'input={0}.affine2'.format(self.name))
        configs.append(line)

        if dropout_proportion >= 0:
            # NOTE: this may not be the optimal form of dropout (shared across
            # time and continuous)... might have to experiment with this.
            configs.append('component name={0}.dropout type=GeneralDropoutComponent '
                           'dim={1} dropout-proportion={2} continuous=true'.format(
                    self.name, output_dim, dropout_proportion))
            configs.append('component-node name={0}.dropout component={0}.dropout '
                           'input={0}.batchnorm2'.format(self.name))
            cur_node = 'dropout'
        else:
            cur_node = 'batchnorm2'


        line = ('component name={0}.noop type=NoOpComponent dim={1}'
                ''.format(self.name, output_dim))
        configs.append(line)
        line = ('component-node name={0}.noop component={0}.noop '
                'input=Sum({0}.{1}, Scale({2}, {3}))'.format(
                self.name, cur_node, bypass_scale, input_desc))
        configs.append(line)

        return configs
