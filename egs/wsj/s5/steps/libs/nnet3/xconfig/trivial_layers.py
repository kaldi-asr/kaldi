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
      include-in-init=false     [You should set this to true if this precedes a
                                `fixed-affine-layer` that is to be initialized
                                 via LDA]
    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input': '[-1]',
                       'target-rms': 1.0,
                       'include-in-init': False}

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
            if self.config['include-in-init']:
                ans.append(('init', line))
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


class XconfigDeltaLayer(XconfigLayerBase):
    """This class is for parsing lines like
     'delta-layer name=delta input=idct'
    which appends the central frame with the delta features
    (i.e. -1,0,1 since scale equals 1) and delta-delta features 
    (i.e. 1,0,-2,0,1), and then applies batchnorm to it.

    Parameters of the class, and their defaults:
      input='[-1]'             [Descriptor giving the input of the layer]
    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input': '[-1]'}

    def check_configs(self):
        pass

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        return self.name

    def output_dim(self, auxiliary_output=None):
        assert auxiliary_output is None
        input_dim = self.descriptors['input']['dim']
        return (3*input_dim)

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
        output_dim = self.output_dim()

        configs = []
        line = ('dim-range-node name={0}_copy1 input-node={0} dim={1} dim-offset=0'.format(
            input_desc, input_dim))
        configs.append(line)
        line = ('dim-range-node name={0}_copy2 input-node={0} dim={1} dim-offset=0'.format(
            input_desc, input_dim))
        configs.append(line)

        line = ('component name={0}_2 type=NoOpComponent dim={1}'.format(
            input_desc, output_dim))
        configs.append(line)
        line = ('component-node name={0}_2 component={0}_2 input=Append(Offset({0},0),'
            ' Sum(Offset(Scale(-1.0,{0}_copy1),-1), Offset({0},1)), Sum(Offset({0},-2), Offset({0},2),' 
            ' Offset(Scale(-2.0,{0}_copy2),0)))'.format(input_desc))
        configs.append(line)
        
        line = ('component name={0} type=BatchNormComponent dim={1}'.format(
            self.name, output_dim))
        configs.append(line)
        line = ('component-node name={0} component={0} input={1}_2'.format(
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
                       'l2-regularize': '',
                       'param-stddev': '',
                       'learning-rate-factor': '' }

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
        for opt_name in ['orthonormal-constraint', 'max-change', 'l2-regularize',
                         'param-stddev', 'learning-rate-factor' ]:
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


class XconfigCombineFeatureMapsLayer(XconfigLayerBase):
    """This class is for parsing lines like
      'combine-feature-maps-layer name=combine_features1 height=40 num-filters1=1 num-filters2=4'
      or
      'combine-feature-maps-layer name=combine_features1 height=40 num-filters1=1 num-filters2=4 num-filters3=2'

      It produces a PermuteComponent.  It expects its input to be two or three things
      appended together, where the first is of dimension height * num-filters1 and
      the second is of dimension height * num-filters2 (and the third, if present is
      of dimension height * num-filters2; it interpolates the filters
      so the output can be interpreted as a single feature map with the same height
      as the input and the sum of the num-filters.

      This is to be used in convolutional setups as part of how we combine the
      filterbank inputs with ivectors.
    """

    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = { 'input': '[-1]',
                        'num-filters1': -1,
                        'num-filters2': -1,
                        'num-filters3': 0,
                        'height': -1 }

    def check_configs(self):
        input_dim = self.descriptors['input']['dim']
        if (self.config['num-filters1'] <= 0 or
            self.config['num-filters2'] <= 0 or
            self.config['num-filters3'] < 0 or
            self.config['height'] <= 0):
            raise RuntimeError("invalid values of num-filters1, num-filters2 and/or height")
        f1 = self.config['num-filters1']
        f2 = self.config['num-filters2']
        f3 = self.config['num-filters3']
        h = self.config['height']
        if input_dim != (f1 + f2 + f3) * h:
            raise RuntimeError("Expected input-dim={0} based on num-filters1={1}, num-filters2={2}, "
                               "num-filters3={3} and height={4}, but got input-dim={5}".format(
                                   (f1 + f2 + f3) * h, f1, f2, f3, h, input_dim))

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
        dim = self.descriptors['input']['dim']
        num_filters1 = self.config['num-filters1']
        num_filters2 = self.config['num-filters2']
        num_filters3 = self.config['num-filters3']  # normally 0.
        height = self.config['height']
        assert dim == (num_filters1 + num_filters2 + num_filters3) * height

        column_map = []
        for h in range(height):
            for f in range(num_filters1):
                column_map.append(h * num_filters1 + f)
            for f in range(num_filters2):
                column_map.append(height * num_filters1 + h * num_filters2 + f)
            for f in range(num_filters3):
                column_map.append(height * (num_filters1 + num_filters2) + h * num_filters3 + f)

        configs = []
        line = ('component name={0} type=PermuteComponent column-map={1} '.format(
            self.name, ','.join([str(x) for x in column_map])))
        configs.append(line)

        line = ('component-node name={0} component={0} input={1}'.format(
            self.name, input_desc))
        configs.append(line)
        return configs




class XconfigAffineComponent(XconfigLayerBase):
    """This class is for parsing lines like
     'affine-component name=linear1 dim=1024 input=Append(-3,0,3)'
    which will produce just a single component, of type NaturalGradientAffineComponent,
    with output-dim 1024 in this case, and input-dim determined by the dimension
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
                       'param-stddev': '',
                       'bias-stddev': '',
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
        for opt_name in ['orthonormal-constraint', 'max-change', 'l2-regularize',
                         'param-stddev', 'bias-stddev']:
            value = self.config[opt_name]
            if value != '':
                opts += ' {0}={1}'.format(opt_name, value)

        configs = []
        line = ('component name={0} type=NaturalGradientAffineComponent input-dim={1} output-dim={2} '
                '{3}'.format(self.name, input_dim, output_dim, opts))
        configs.append(line)
        line = ('component-node name={0} component={0} input={1}'.format(
            self.name, input_desc))
        configs.append(line)
        return configs


class XconfigPerElementScaleComponent(XconfigLayerBase):
    """This class is for parsing lines like
     'scale-component name=scale1 input=Append(-3,0,3)'
    which will produce just a single component, of type NaturalGradientPerElementScaleComponent, with
    output-dim 1024 in this case, and input-dim determined by the dimension of the input .

    Parameters of the class, and their defaults:
      input='[-1]'             [Descriptor giving the input of the layer.]

    The following (shown with their effective defaults) are just passed through
    to the component's config line.  (These defaults are mostly set in the
    code).

      max-change=0.75
      l2-regularize=0.0
      param-mean=1.0   # affects initialization
      param-stddev=0.0  # affects initialization
      learning-rate-factor=1.0
    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input': '[-1]',
                       'l2-regularize': '',
                       'max-change': 0.75,
                       'param-mean': '',
                       'param-stddev': '',
                       'learning-rate-factor': '' }

    def check_configs(self):
        pass

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        return self.name

    def output_dim(self, auxiliary_output=None):
        assert auxiliary_output is None
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
        dim = self.descriptors['input']['dim']

        opts = ''
        for opt_name in ['learning-rate-factor', 'max-change', 'l2-regularize', 'param-mean',
                         'param-stddev' ]:
            value = self.config[opt_name]
            if value != '':
                opts += ' {0}={1}'.format(opt_name, value)

        configs = []
        line = ('component name={0} type=NaturalGradientPerElementScaleComponent dim={1} {2} '
                ''.format(self.name, dim, opts))
        configs.append(line)
        line = ('component-node name={0} component={0} input={1}'.format(
            self.name, input_desc))
        configs.append(line)
        return configs

class XconfigPerElementOffsetComponent(XconfigLayerBase):
    """This class is for parsing lines like
     'offset-component name=offset1 input=Append(-3,0,3)'
    which will produce just a single component, of type PerElementOffsetComponent, with
    output-dim 1024 in this case, and input-dim determined by the dimension of the input .

    Parameters of the class, and their defaults:
      input='[-1]'             [Descriptor giving the input of the layer.]

    The following (shown with their effective defaults) are just passed through
    to the component's config line.  (These defaults are mostly set in the
    code).

      max-change=0.75
      l2-regularize=0.0
      param-mean=0.0   # affects initialization
      param-stddev=0.0  # affects initialization
      learning-rate-factor=1.0
    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input': '[-1]',
                       'l2-regularize': '',
                       'max-change': 0.75,
                       'param-mean': '',
                       'param-stddev': '',
                       'learning-rate-factor': '' }

    def check_configs(self):
        pass

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        return self.name

    def output_dim(self, auxiliary_output=None):
        assert auxiliary_output is None
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
        dim = self.descriptors['input']['dim']

        opts = ''
        for opt_name in ['learning-rate-factor', 'max-change', 'l2-regularize', 'param-mean',
                         'param-stddev' ]:
            value = self.config[opt_name]
            if value != '':
                opts += ' {0}={1}'.format(opt_name, value)

        configs = []
        line = ('component name={0} type=PerElementOffsetComponent dim={1} {2} '
                ''.format(self.name, dim, opts))
        configs.append(line)
        line = ('component-node name={0} component={0} input={1}'.format(
            self.name, input_desc))
        configs.append(line)
        return configs


class XconfigDimRangeComponent(XconfigLayerBase):
    """This class is for parsing lines like
     'dim-range-component name=feature1 input=Append(-3,0,3) dim=40 dim-offset=0'
    which will produce just a single component, of part of the input.
    Parameters of the class, and their defaults:
      input='[-1]'             [Descriptor giving the input of the layer.]
      dim=-1                   [Dimension of the output.]
      dim-offset=0             [Dimension offset of the input.]
    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input': '[-1]',
                       'dim': -1,
                       'dim-offset': 0 }

    def check_configs(self):
        input_dim = self.descriptors['input']['dim']
        if self.config['dim'] <= 0:
            raise RuntimeError("'dim' must be specified and > 0.")
        elif self.config['dim'] > input_dim:
            raise RuntimeError("'dim' must be specified and lower than the input dim.")
        if self.config['dim-offset'] < 0 :
            raise RuntimeError("'dim-offset' must be specified and >= 0.")
        elif self.config['dim-offset'] + self.config['dim'] > input_dim:
            raise RuntimeError("'dim-offset' plus output dim must be lower than the input dim.")

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        return self.name

    def output_dim(self, auxiliary_output=None):
        assert auxiliary_output is None
        output_dim = self.config['dim']
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
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_node = self.descriptors['input']['final-string']
        output_dim = self.config['dim']
        dim_offset = self.config['dim-offset']

        configs = []
        line = ('dim-range-node name={0} input-node={1} dim={2} dim-offset={3}'.format(
            self.name, input_node, output_dim, dim_offset))
        configs.append(line)
        return configs
