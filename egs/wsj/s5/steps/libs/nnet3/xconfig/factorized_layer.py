# Copyright 2017-2018   Johns Hopkins University (Dan Povey)
#                2016    Vijayaditya Peddinti
#                2017    Google Inc. (vpeddinti@google.com)
#                2017    Vimal Manohar
# Apache 2.0.

""" This module contains layers that just map to a single component.
"""

from __future__ import print_function
import math
import re
import sys
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase


class XconfigFactorizedLayer(XconfigLayerBase):
    """This class is for parsing lines like
     'factorized-layer name=tdnn1 dim=1024 bottleneck-dim=256 bypass-scale=1.0 splicing=-3,0,3'

    This is basically the same as a relu-batchnorm-layer with the bottleneck-dim
    set, except that it supports the 'bypass-scale' option, which makes the
    whole thing a bit like a res-block.  You specify the splicing via the 'splicing'
    option instead of via 'input=xxx', as it needs to use the non-spliced inupt for
    the bypass.

    Note: the 'dim' is actually optional; it will default to the
    dimension of the input, and it must be the same as the dimension of the input.


    Parameters of the class, and their defaults:
      input='[-1]'             [Descriptor giving the input of the layer.]
      splicing='0'             [In general can be a comma-separated string describing
                                the TDNN time-offsets, like '-1,0,1' or '-3,0,3'.
                                Not specified via 'input', because we need the un-spliced
                                input so that we can do the] bypass.
      dim=-1                    [Output dimension of layer, e.g. 1024; must be set.]
      bottleneck-dim=-1          [Bottleneck dimension, must be set; e.g. 256]
      self-repair-scale=1.0e-05  [Affects the relu layer]
      learning-rate-factor=1.0   [This can be used to make the affine component
                                  train faster or slower].
      l2-regularize=0.0       [Set this to a nonzero value (e.g. 1.0e-05) to
                               add l2 regularization on the parameter norm for
                                this component.

    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        assert first_token == "factorized-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):

        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = {'input': '[-1]',
                       'dim': -1,
                       'bottleneck-dim': -1,
                       'self-repair-scale': 1.0e-05,
                       'target-rms': 1.0,
                       'extra-relu': False,
                       'splicing': '0',
                       'bypass-scale': 1.0,
                       'ng-affine-options': '',
                       'ng-linear-options': '',
                       # if second-matrix-orthonormal, the 2nd matrix
                       # has the orthonormal constraint.
                       'second-matrix-orthonormal': False,
                       # The following are passed through to components.
                       'bias-stddev': '',
                       'l2-regularize': '',
                       'learning-rate-factor': '',
                       'max-change': 0.75 }

    def check_configs(self):
        input_dim = self.descriptors['input']['dim']

        if self.config['dim'] == -1:
            self.config['dim'] = input_dim
        elif self.config['dim'] != input_dim:
            raise RuntimeError("Dimension mismatch: dim={0} vs. input-dim={1}".format(
                self.config['dim'], input_dim))
        b = self.config['bottleneck-dim']
        if b <= 0 or b >= self.config['dim']:
            raise RuntimeError("bottleneck-dim has an invalid value {0}".format(b))

        if self.config['self-repair-scale'] < 0.0 or self.config['self-repair-scale'] > 1.0:
            raise RuntimeError("self-repair-scale has invalid value {0}"
                               .format(self.config['self-repair-scale']))
        if self.config['target-rms'] < 0.0:
            raise RuntimeError("target-rms has invalid value {0}"
                               .format(self.config['target-rms']))
        if self.config['learning-rate-factor'] <= 0.0:
            raise RuntimeError("learning-rate-factor has invalid value {0}"
                               .format(self.config['learning-rate-factor']))

        splicing = self.config['splicing']
        try:
            splicing_array = [ int(x) for x in splicing.split(',') ]
            if not 0 in splicing_array:
                raise RuntimeError("0 should probably be in the splicing indexes.")
        except:
            raise RuntimeError("Invalid option splicing={0}".format(splicing))

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        # return something like: tdnn3.batchnorm
        return '{0}.batchnorm'.format(self.name)

    def output_dim(self, auxiliary_output=None):
        output_dim = self.config['dim']
        # If not set, the output-dim defaults to the input-dim.
        if output_dim <= 0:
            self.config['dim'] = self.descriptors['input']['dim']
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
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        bottleneck_dim = self.config['bottleneck-dim']
        output_dim = input_dim
        self_repair_scale = self.config['self-repair-scale']
        target_rms = self.config['target-rms']
        bypass_scale = self.config['bypass-scale']
        splicing_array = [ int(x) for x in self.config['splicing'].split(',') ]
        spliced_input_desc = 'Append({0})'.format(
            ', '.join([ 'Offset({0}, {1})'.format(input_desc, offset)
                        for offset in splicing_array ]))
        extra_relu = self.config['extra-relu']

        # e.g. spliced_input_desc =
        #   'Append(Offset(tdnn2, -1), Offset(tdnn2, 0), Offset(tdnn2, 1))'

        spliced_input_dim = input_dim * len(splicing_array)

        affine_options = self.config['ng-affine-options']
        for opt_name in [ 'max-change', 'learning-rate-factor',
                          'bias-stddev', 'l2-regularize' ]:
            value = self.config[opt_name]
            if value != '':
                affine_options += ' {0}={1}'.format(opt_name, value)

        linear_options = self.config['ng-linear-options']
        for opt_name in [ 'max-change', 'learning-rate-factor' ]:
            value = self.config[opt_name]
            if value != '':
                linear_options += ' {0}={1}'.format(opt_name, value)

        if self.config['second-matrix-orthonormal']:
            # we have to mess with the range of the parameters so they are within
            # the circle of convergence...
            affine_options += ' orthonormal-constraint=1.0 param-stddev={0}'.format(
                math.sqrt(1.0 / output_dim))
        else:
            linear_options += ' orthonormal-constraint=1.0'

        configs = []

        # First the linear component that goes to the bottleneck dim.
        # note: by default the LinearComponent uses natural gradient.
        line = ('component name={0}.linear type=LinearComponent '
                'input-dim={1} output-dim={2} {3}'
                ''.format(self.name, spliced_input_dim, bottleneck_dim,
                          linear_options))
        configs.append(line)
        line = ('component-node name={0}.linear component={0}.linear input={1}'
                ''.format(self.name, spliced_input_desc))
        configs.append(line)

        if extra_relu:
            # add a relu between the linear and the affine.
            line = ('component name={0}.relu0 type=RectifiedLinearComponent dim={1}'
                    ' self-repair-scale={2}'
                    ''.format(self.name, bottleneck_dim, self_repair_scale))
            configs.append(line)
            line = ('component-node name={0}.relu0 component={0}.relu0 '
                    'input={0}.linear'.format(self.name))
            configs.append(line)


        # Now the affine component
        line = ('component name={0}.affine type=NaturalGradientAffineComponent'
                ' input-dim={1} output-dim={2} {3}'
                ''.format(self.name, bottleneck_dim, output_dim, affine_options))
        configs.append(line)
        line = ('component-node name={0}.affine component={0}.affine input={0}.{1}'
                ''.format(self.name, ('relu0' if extra_relu else 'linear')))
        configs.append(line)

        # now the ReLU.  Its input is the output of the affine component plus
        # the non-sliced input (this is a bit like a res-block).
        line = ('component name={0}.relu type=RectifiedLinearComponent dim={1}'
                ' self-repair-scale={2}'
                ''.format(self.name, output_dim, self_repair_scale))
        configs.append(line)
        if bypass_scale != 0.0:
            line = ('component-node name={0}.relu component={0}.relu '
                    'input=Sum(Scale({1}, {2}), {0}.affine) '
                    ''.format(self.name, bypass_scale, input_desc))
        else:
            line = ('component-node name={0}.relu component={0}.relu '
                    'input={0}.affine'.format(self.name))
        configs.append(line)

        line = ('component name={0}.batchnorm type=BatchNormComponent '
                'dim={1} target-rms={2}'
                ''.format(self.name, output_dim, target_rms))
        configs.append(line)
        line = ('component-node name={0}.batchnorm component={0}.batchnorm '
                'input={0}.relu'.format(self.name))
        configs.append(line)

        return configs
