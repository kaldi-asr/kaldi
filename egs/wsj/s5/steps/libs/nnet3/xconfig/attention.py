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
                        'dim': -1,
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


# This class is for parsing lines like
#  'attention-block dim=768  bottleneck-dim=128 num-heads=8 value-dim=50 key-dim=50 time-stride=3 num-left-inputs=30 num-right-inputs=10 bypass-scale=0.66'
#
#  It is a little like a TDNNF-layer, but with attention in the middle and no
#  ReLU.  Note: as of now, there is no nonlinearity other than what comes from
#  the attention component itself (it has a softmax).  Imagine the input and
#  output dim of the layer is largish, like 768.
#
#  So we go, 768 --(linear with orthonormal)--> 128 --(affine)--> attention-input-dim  --(attention)--> (50+context-dim)*8  \
#            --(linear with orthonormal)-->128 -->(linear) 768 -> batchnorm, then add residual connection from original 768-dim input.
#
#  ... where attention-input-dim equals value-dim + 2*key-dim + context-dim
#  and context-dim = (num-left-inputs + 1 + num-right-inputs + 1)
#     in this case it's 50 + 2*50 + (30+10+1) = 191.
#
#
# Parameters of the class, and their defaults:
#   input='[-1]'               [Descriptor giving the input of the layer.]
#   bottleneck-dim=-1              [bottleneck dimension, e.g. 128.]
#   num-heads=-1               [Number of attention heads, e.g. 8]
#   value-dim=-1               [Dimension of values (the things which get weighted-averaged
#                               and then output. E.g. 50]
#   key-dim=-1                 [Dimension of the keys, e.g. 50.  Affects the query
#                               dimension, but that's larger by context_dim,
#                               where context_dim == num-left-inputs+1+num-right-inputs.
#                               That's for the encoding of the position of the input frame.]
#   dim=-1                     [Dimension of the output of this layer (after the bottleneck;
#                               e.g. 768].  Defaults to the dimension of the input.]
#   time-stride=1              [Time stride, dictates the spacing of the inputs to this
#                               layer.  E.g. might be 3 in typical TDNN-F setups.]
#   num-left-inputs=-1         [Number of inputs to the left that we use.  Must be specified.]
#   num-right-inputs=-1         [Number of inputs to the right that we use.  Must be specified.]
#   num-left-inputs-required: -1   [This affects the left/right context that the network will
#                                have, i.e. how many frames of input it will insist on having.
#                                It affects the behavior at chunk boundaries; larger will tend
#                                to be slower but more accurate.  Note: the default of -1 means:
#                                use the same as num-left-inputs].
#   num-right-inputs-required: -1  [See comment for num-left-inputs-required]
#   output-context:  True        [If true, the softmax weights will be an additional
#                                output of the attention heads.]
#   key-scale: 0.0               [If >0.0, becomes a scaling factor on the keys.  Otherwise, we
#                                 use the default value of 1.0 / sqrt(key-dim).]
#
#
#  bypass-scale : 0.66          [Scale on the input in the residual connection.]
#  target-rms:   1.0            [Scaling on the output of the batchnorm]
#
#  Extra configs that are passed into the affine and linear components:
#   learning-rate-factor=1.0   [This can be used to make the affine component
#                               train faster or slower].
#   max-change=0.75    [maximum change per iteration, per component]
#   l2-regularize=0.0  [l2 regularization constant for linear and affine components.]
#
#   Documentation for the rest of the parameters (related to the
#   attention component) can be found in nnet-attention-component.h


class XconfigAttentionBlock(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        # Here we just list some likely combinations.. you can just add any
        # combinations you want to use, to this list.
        assert first_token == 'attention-block'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = { 'input':'[-1]',
                        'dim': -1,
                        'bottleneck-dim': -1,
                        'num-heads': -1,
                        'value-dim': -1,
                        'key-dim': -1,
                        'dim': -1,
                        'time-stride': 1,
                        'num-left-inputs': -1,
                        'num-right-inputs': -1,
                        'learning-rate-factor': 1.0,
                        'max-change' : 0.75,
                        'ng-affine-options' : '',
                        'l2-regularize': 0.0,
                        'num-left-inputs-required': -1,
                        'num-right-inputs-required': -1,
                        'output-context': True,
                        'target-rms': 1.0,
                        'key-scale': 0.0,
                        'bypass-scale': 0.66 }


    def check_configs(self):
        for x in [ 'bottleneck-dim', 'num-heads', 'value-dim', 'key-dim' ]:
            if self.config[x] <= 0:
                raise RuntimeError("Expected {} to be positive, got {}".format(x, self.config[x]))
        for x in ['num-left-inputs', 'num-right-inputs' ]:
            if self.config[x] < 0:
                raise RuntimeError("Expected {} to be nonnegative, got {}".format(x, self.config[x]))
        # Not checking everything here.
        if self.config['learning-rate-factor'] <= 0.0:
            raise RuntimeError("learning-rate-factor has invalid value {0}"
                               .format(self.config['learning-rate-factor']))
        if self.config['key-scale'] == 0.0:
            self.config['key-scale'] = 1.0 / math.sqrt(self.config['key-dim'])

    def output_name(self, auxiliary_output=None):
        # at a later stage we might want to expose even the pre-nonlinearity
        # vectors
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
        dim = self.config['dim']
        if dim > 0:
            return dim
        else:
            return self.descriptors['input']['dim']

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
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        output_dim = self.config['dim']
        if output_dim <= 0:
            output_dim = input_dim
        bottleneck_dim = self.config['bottleneck-dim']
        attention_input_dim = self.attention_input_dim()
        attention_output_dim = self.attention_output_dim()
        target_rms = self.config['target-rms']
        max_change = self.config['max-change']
        l2_regularize = self.config['l2-regularize']
        learning_rate_factor=self.config['learning-rate-factor']

        learning_rate_option=('learning-rate-factor={0}'.format(learning_rate_factor)
                              if learning_rate_factor != 1.0 else '')
        l2_regularize_option = ('l2-regularize={0} '.format(l2_regularize)
                                if l2_regularize != 0.0 else '')

        common_options=("{lroption} {l2option} max-change={max_change} "
                        "".format(lroption = learning_rate_option,
                                  l2option = l2_regularize_option,
                                  max_change = max_change))


        configs = []


        # The first linear component
        line = ('component name={0}.linear1 type=LinearComponent '
                'input-dim={1} output-dim={2} '
                '{3} orthonormal-constraint=-1 '
                ''.format(self.name, input_dim, bottleneck_dim,
                          common_options))

        configs.append(line)
        line = ('component-node name={0}.linear1 component={0}.linear1 input={1} '
                ''.format(self.name, input_desc))
        configs.append(line)

        # The first affine component
        line = ('component name={0}.affine1 type=NaturalGradientAffineComponent '
                'input-dim={1} output-dim={2} '
                '{3}'.format(self.name, bottleneck_dim, attention_input_dim,
                             common_options))
        configs.append(line)
        line = ('component-node name={0}.affine1 component={0}.affine1 input={0}.linear1'
                ''.format(self.name, input_desc))
        configs.append(line)


        # Batch-norm component
        if True:
            line = ('component name={0}.layernorm1 type=NormalizeComponent dim={1} '
                    ' '.format(self.name, attention_input_dim))
            configs.append(line)
            line = ('component-node name={0}.layernorm1 component={0}.layernorm1 '
                    'input={0}.affine1 '.format(self.name))
            configs.append(line)
            cur_name='layernorm1'
        else:
            cur_name='affine1'


        # The attention component
        line = ('component name={name}.attention type=RestrictedAttentionComponent '
                'value-dim={v} key-dim={k} num-left-inputs={nl} '
                'num-right-inputs={nr} num-left-inputs-required={nlr}'
                ' num-right-inputs-required={nrr} output-context={oc}'
                ' time-stride={ts} num-heads={nh} key-scale={ks}'
                ''.format(name=self.name,
                          v=self.config['value-dim'], k=self.config['key-dim'],
                          nl=self.config['num-left-inputs'],
                          nr=self.config['num-right-inputs'],
                          nlr=self.config['num-left-inputs-required'],
                          nrr=self.config['num-right-inputs-required'],
                          oc=self.config['output-context'],
                          ts=self.config['time-stride'],
                          nh=self.config['num-heads'],
                          ks=self.config['key-scale']))
        configs.append(line)
        line = ('component-node name={0}.attention component={0}.attention input={0}.{1}'
                ''.format(self.name, cur_name))
        configs.append(line)

        # The second linear component
        line = ('component name={0}.linear2 type=LinearComponent '
                'input-dim={1} output-dim={2} orthonormal-constraint=-1 '
                '{3}'.format(self.name, attention_output_dim, bottleneck_dim,
                             common_options))
        configs.append(line)
        line = ('component-node name={0}.linear2 component={0}.linear2 '
                'input={0}.attention '.format(self.name))
        configs.append(line)

        # The third linear component
        line = ('component name={0}.linear3 type=LinearComponent '
                'input-dim={1} output-dim={2} '
                '{3}'.format(self.name, bottleneck_dim, output_dim,
                             common_options))
        configs.append(line)
        line = ('component-node name={0}.linear3 component={0}.linear3 '
                'input={0}.linear2 '.format(self.name))
        configs.append(line)

        line = ('component name={0}.layernorm2 type=NormalizeComponent dim={1} '
                'target-rms={2} '.format(self.name, output_dim, target_rms))
        configs.append(line)
        line = ('component-node name={0}.layernorm2 component={0}.layernorm2 '
                'input={0}.linear3 '.format(self.name))
        configs.append(line)


        line = ('component name={0}.noop type=NoOpComponent dim={1}'.format(
            self.name, output_dim))
        configs.append(line)
        line = ('component-node name={name}.noop component={name}.noop input=Sum(Scale({b}, {i}), {name}.layernorm2)'
                ''.format(name=self.name, b=self.config['bypass-scale'], i=input_desc))
        configs.append(line)

        return configs
