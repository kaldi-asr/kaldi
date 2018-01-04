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
     'renorm-component name=renorm input=Append(-3,0,3)'
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
