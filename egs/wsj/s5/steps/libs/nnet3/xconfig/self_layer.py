# Copyright 2016    Johns Hopkins University (Author: Daniel Povey)
#           2016    Vimal Manohar
#           2018    Tom Ko, Zhu Yingke
# Apache 2.0.

""" This module contains the statistics extraction and self attention layer.
"""

from __future__ import print_function
import re
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase


class XconfigSelfLayer(XconfigLayerBase):
    """This class is for parsing lines like
    self-layer name=attention config=mean+stddev(-99:3:9:99) input=tdnn1

    This adds the self attention mechanism which consists of a series of components.  An
    example string is 'mean(-99:3:9::99)', which means, compute the mean of
    data within a window of -99 to +99, with distinct means computed every 9
    frames (we round to get the appropriate one), and with the input extracted
    on multiples of 3 frames (so this will force the input to this layer to be
    evaluated every 3 frames).  Another example string is
    'mean+stddev(-99:3:9:99)', which will also cause the standard deviation to
    be computed.

    The dimension is worked out from the input. mean and stddev add a
    dimension of input_dim each to the output dimension. If counts is
    specified, an additional dimension is added to the output to store log
    counts.

    Parameters of the class, and their defaults:
        input='[-1]'    [Descriptor giving the input of the layer.]
        dim=-1      [Output dimension of layer. If provided, must match the
                     dimension computed from input]
        config=''   [Required. Defines what stats must be computed.]
        num-heads=0
    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        assert first_token in ['self-layer']
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input': '[-1]',
                       'dim': -1,
                       'config': '',
                       'affine-dim': 300,
                       'num-heads': 0}

    def set_derived_configs(self):
        config_string = self.config['config']
        if config_string == '':
            raise RuntimeError("config has to be non-empty",
                                self.str())
        m = re.search("(mean|mean\+stddev|mean\+count|mean\+stddev\+count)"
                      "\((-?\d+):(-?\d+):(-?\d+):(-?\d+)\)",
                      config_string)
        if m is None:
            raise RuntimeError("Invalid statistic-config string: {0}".format(
                config_string), self)

        self._output_stddev = (m.group(1) in ['mean+stddev',
                                              'mean+stddev+count'])
        self._output_log_counts = (m.group(1) in ['mean+count',
                                                  'mean+stddev+count'])
        self._left_context = -int(m.group(2))
        self._input_period = int(m.group(3))
        self._stats_period = int(m.group(4))
        self._right_context = int(m.group(5))
        self._num_heads = self.config['num-heads']
        self._affine_dim = self.config['affine-dim']

        if self._output_stddev:
          output_dim = 2 * self.descriptors['input']['dim']
        else:
          output_dim = self.descriptors['input']['dim']
        if self._output_log_counts:
          output_dim = output_dim + 1

        if self._num_heads > 0:
          output_dim = output_dim * self._num_heads


        if self.config['dim'] > 0 and self.config['dim'] != output_dim:
            raise RuntimeError(
                "Invalid dim supplied {0:d} != "
                "actual output dim {1:d}".format(
                    self.config['dim'], output_dim))
        self.config['dim'] = output_dim

    def check_configs(self):
        if not (self._left_context >= 0 and self._right_context >= 0
                and self._input_period > 0 and self._stats_period > 0
                and self._left_context % self._stats_period == 0
                and self._right_context % self._stats_period == 0
                and self._stats_period % self._input_period == 0):
            raise RuntimeError(
                "Invalid configuration of statistics-extraction: {0}".format(
                    self.config['config']), self)
        super(XconfigSelfLayer, self).check_configs()

    def _generate_config(self):
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']

        # Here, the whole self layer computes a linear combination of a series of frames.
        # The linear combination weights are computed by a multi layer perceptron.
        # The multi layer perceptron consists of the following components in sequence:
        # an affine component, a ReLU component, a batchnorm component, another affine component
        # and finally a softmax component.
        # The weights are then append with the same series of frames and pass to 
        # the SelfAttentionComponent for the computation of linear combination of frames.
        # The tricky part here is that the softmax computation are done in the SelfAttentionComponent
        # as the weights of each head are arranged in columns of the CuMatrixBase
        # We need to transpose the weights to row vector, then call ApplySoftMaxPerRow(), 
        # then transpose the weights back to column vector.

        da_dim = self._affine_dim
        self_repair_scale = 1e-05
        target_rms = 1.0
        affine_options = 'max-change=0.75'

        configs = []
        # The first affine node.
        configs.append(
            'component name={0}.first_affine type=NaturalGradientAffineComponent'
            ' input-dim={1} output-dim={2} {3}'
            ''.format(self.name, input_dim, da_dim, affine_options))

        configs.append(
            'component-node name={0}.first_affine component={0}.first_affine input={1}'
            ''.format(self.name, input_desc))

        # The ReLU node.
        configs.append(
             'component name={0}.relu type=RectifiedLinearComponent dim={1}'
             ' self-repair-scale={2}'
             ''.format(self.name, da_dim, self_repair_scale))

        configs.append(
             'component-node name={0}.relu'
             ' component={0}.relu input={0}.first_affine'
             ''.format(self.name))

        # The batchnorm node.
        configs.append(
             'component name={0}.batchnorm type=BatchNormComponent dim={1}'
             ' target-rms={2}'
             ''.format(self.name, da_dim, target_rms))

        configs.append(
             'component-node name={0}.batchnorm'
             ' component={0}.batchnorm input={0}.relu'
             ''.format(self.name))

        # The second affine node.
        affine_options = 'param-stddev=0.04472135955 bias-stddev=1.0 bias-mean=0.0 max-change=0.75 l2-regularize=0.0'
        configs.append(
            'component name={0}.second_affine type=NaturalGradientAffineComponent'
            ' input-dim={1} output-dim={2} {3}'
            ''.format(self.name, da_dim, self._num_heads, affine_options))

        configs.append(
            'component-node name={0}.second_affine component={0}.second_affine input={0}.batchnorm'
            ''.format(self.name))


        # The statistic extraction node.
        configs.append(
            'component name={name}-extraction-{lc}-{rc} '
            'type=StatisticsExtractionComponent input-dim={dim} '
            'input-period={input_period} output-period={output_period} '
            'include-variance={var} '.format(
                name=self.name, lc=self._left_context, rc=self._right_context,
                dim=input_dim, input_period=self._input_period,
                output_period=self._stats_period,
                var='true' if self._output_stddev else 'false'))
        configs.append(
            'component-node name={name}-extraction-{lc}-{rc} '
            'component={name}-extraction-{lc}-{rc} input={input} '.format(
                name=self.name, lc=self._left_context, rc=self._right_context,
                input=input_desc))

        stats_dim = self._num_heads + 1 + input_dim * (2 if self._output_stddev else 1)
        configs.append(
            'component name={name}-self-attention-{lc}-{rc} '
            'type=SelfAttentionComponent input-dim={dim} '
            'input-period={input_period} left-context={lc} right-context={rc} num-heads={heads} '
            'num-log-count-features={count} output-stddevs={var} '.format(
                name=self.name, lc=self._left_context, rc=self._right_context,
                heads=self._num_heads,
                dim=stats_dim, input_period=self._stats_period,
                count=1 if self._output_log_counts else 0,
                var='true' if self._output_stddev else 'false'))
        configs.append(
            'component-node name={name}-self-attention-{lc}-{rc} '
            'component={name}-self-attention-{lc}-{rc} '
            'input=Append({name}.second_affine,{name}-extraction-{lc}-{rc}) '.format(
                name=self.name, lc=self._left_context, rc=self._right_context))
        return configs

    def output_name(self, auxiliary_output=None):
        return 'Round({name}-self-attention-{lc}-{rc}, {period})'.format(
            name=self.name, lc=self._left_context,
            rc=self._right_context, period=self._stats_period)

    def output_dim(self, auxiliary_outputs=None):
        return self.config['dim']

    def get_full_config(self):
        ans = []
        config_lines = self._generate_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                ans.append((config_name, line))

        return ans
