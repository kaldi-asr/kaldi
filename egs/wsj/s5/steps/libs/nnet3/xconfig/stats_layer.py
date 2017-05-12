# Copyright 2016    Johns Hopkins University (Author: Daniel Povey)
#           2016    Vimal Manohar
# Apache 2.0.

""" This module contains the statistics extraction and pooling layer.
"""

from __future__ import print_function
import re
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase


class XconfigStatsLayer(XconfigLayerBase):
    """This class is for parsing lines like
    stats-layer name=tdnn1-stats config=mean+stddev(-99:3:9:99) input=tdnn1

    This adds statistics-pooling and statistics-extraction components.  An
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
    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        assert first_token in ['stats-layer']
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input': '[-1]',
                       'dim': -1,
                       'config': ''}

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

        output_dim = (self.descriptors['input']['dim']
                      * (2 if self._output_stddev else 1)
                      + 1 if self._output_log_counts else 0)

        if self.config['dim'] > 0 and self.config['dim'] != output_dim:
            raise RuntimeError(
                "Invalid dim supplied {0:d} != "
                "actual output dim {1:d}".format(
                    self.config['dim'], output_dim))
        self.config['dim'] = output_dim

    def check_configs(self):
        if not (self._left_context > 0 and self._right_context > 0
                and self._input_period > 0 and self._stats_period > 0
                and self._left_context % self._stats_period == 0
                and self._right_context % self._stats_period == 0
                and self._stats_period % self._input_period == 0):
            raise RuntimeError(
                "Invalid configuration of statistics-extraction: {0}".format(
                    self.config['config']), self)
        super(XconfigStatsLayer, self).check_configs()

    def _generate_config(self):
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']

        configs = []
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

        stats_dim = 1 + input_dim * (2 if self._output_stddev else 1)
        configs.append(
            'component name={name}-pooling-{lc}-{rc} '
            'type=StatisticsPoolingComponent input-dim={dim} '
            'input-period={input_period} left-context={lc} right-context={rc} '
            'num-log-count-features={count} output-stddevs={var} '.format(
                name=self.name, lc=self._left_context, rc=self._right_context,
                dim=stats_dim, input_period=self._stats_period,
                count=1 if self._output_log_counts else 0,
                var='true' if self._output_stddev else 'false'))
        configs.append(
            'component-node name={name}-pooling-{lc}-{rc} '
            'component={name}-pooling-{lc}-{rc} '
            'input={name}-extraction-{lc}-{rc} '.format(
                name=self.name, lc=self._left_context, rc=self._right_context))
        return configs

    def output_name(self, auxiliary_output=None):
        return 'Round({name}-pooling-{lc}-{rc}, {period})'.format(
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
