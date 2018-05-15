# Copyright 2016    Johns Hopkins University (Dan Povey)
#           2016    Vijayaditya Peddinti
#           2017    Google Inc. (vpeddinti@google.com)
#           2017    Vimal Manohar
# Apache 2.0.

""" This module contains layers that just map to a single component.
"""

from __future__ import print_function
import math
import re
import sys
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase


class XconfigRenormComponent(XconfigLayerBase):
    """This class is for parsing lines like
     'renorm-component name=renorm1 input=Append(-3,0,3)'
    which will produce just a single component, of type NormalizeComponent.

    Parameters of the class, and their defaults:
      input='[-1]'             [Descriptor giving the input of the layer.]
      target-rms=1.0           [The target RMS of the NormalizeComponent]
    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input': '[-1]',
                       'target-rms': 1.0 }

    def check_configs(self):
        assert self.config['target-rms'] > 0.0

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        return self.name

    def output_dim(self, auxiliary_output=None):
        assert auxiliary_output is None
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
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        target_rms = self.config['target-rms']

        configs = []
        line = ('component name={0} type=NormalizeComponent dim={1} target-rms={2}'.format(
            self.name, input_dim, target_rms))
        configs.append(line)
        line = ('component-node name={0} component={0} input={1}'.format(
            self.name, input_desc))
        configs.append(line)
        return configs


class XconfigBatchnormComponent(XconfigLayerBase):
    """This class is for parsing lines like
     'batchnorm-component name=batchnorm input=Append(-3,0,3)'
    which will produce just a single component, of type BatchNormComponent.

    Parameters of the class, and their defaults:
      input='[-1]'             [Descriptor giving the input of the layer.]
      target-rms=1.0           [The target RMS of the BatchNormComponent]
    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input': '[-1]',
                       'target-rms': 1.0 }

    def check_configs(self):
        assert self.config['target-rms'] > 0.0

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        return self.name

    def output_dim(self, auxiliary_output=None):
        assert auxiliary_output is None
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
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        target_rms = self.config['target-rms']

        configs = []
        line = ('component name={0} type=BatchNormComponent dim={1} target-rms={2}'.format(
            self.name, input_dim, target_rms))
        configs.append(line)
        line = ('component-node name={0} component={0} input={1}'.format(
            self.name, input_desc))
        configs.append(line)
        return configs


class XconfigNoOpComponent(XconfigLayerBase):
    """This class is for parsing lines like
     'no-op-component name=noop1 input=Append(-3,0,3)'
    which will produce just a single component, of type NoOpComponent.

    Parameters of the class, and their defaults:
      input='[-1]'             [Descriptor giving the input of the layer.]
    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input': '[-1]' }

    def check_configs(self):
        pass

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        return self.name

    def output_dim(self, auxiliary_output=None):
        assert auxiliary_output is None
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
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']

        configs = []
        line = ('component name={0} type=NoOpComponent dim={1}'.format(
            self.name, input_dim))
        configs.append(line)
        line = ('component-node name={0} component={0} input={1}'.format(
            self.name, input_desc))
        configs.append(line)
        return configs


class XconfigLinearComponent(XconfigLayerBase):
    """This class is for parsing lines like
     'linear-component name=linear1 dim=1024 input=Append(-3,0,3)'
    which will produce just a single component, of type LinearComponent, with
    output-dim 1024 in this case, and input-dim determined by the dimension
    of the input .

    Parameters of the class, and their defaults:
      input='[-1]'             [Descriptor giving the input of the layer.]
      dim=-1                   [Dimension of the output]

    The following (shown with their effective defaults) are just passed through
    to the component's config line.

      orthonormal-constraint=0.0
      max-change=0.75
      l2-regularize=0.0

    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input': '[-1]',
                       'dim': -1,
                       'orthonormal-constraint': '',
                       'max-change': 0.75,
                       'l2-regularize': '' }

    def check_configs(self):
        if self.config['dim'] <= 0:
            raise RuntimeError("'dim' must be specified and > 0.")

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        return self.name

    def output_dim(self, auxiliary_output=None):
        assert auxiliary_output is None
        assert self.config['dim'] > 0
        return self.config['dim']

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

        opts = ''
        for opt_name in ['orthonormal-constraint', 'max-change', 'l2-regularize']:
            value = self.config[opt_name]
            if value != '':
                opts += ' {0}={1}'.format(opt_name, value)

        configs = []
        line = ('component name={0} type=LinearComponent input-dim={1} output-dim={2} '
                '{3}'.format(self.name, input_dim, output_dim, opts))
        configs.append(line)
        line = ('component-node name={0} component={0} input={1}'.format(
            self.name, input_desc))
        configs.append(line)
        return configs
