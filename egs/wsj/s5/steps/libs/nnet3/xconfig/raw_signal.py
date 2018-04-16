# Copyright 2017 Pegah Ghahremani
# Apache 2.0.

""" This module contains layer types for processig raw waveform frames.
"""

from __future__ import print_function
import math
import re
import sys
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase

# This class is used for frequency-domain filter learning.
# This class is for parsing lines like
# 'preprocess-fft-abs-lognorm-affine-log-layer fft-dim=512 num-left-inputs=1'
# 'num-right-inputs=2 l2-reg=0.001'
# preprocess : applies windowing and pre-emphasis on input frames.
# fft : compute real and imaginary part of discrete cosine transform
#       using sine and cosine transform.
# abs : computes absolute value of real and complex part of fft.
# lognorm : normalize input in log-space using batchnorm followed by per-element
#           scale and offset.
# affine : filterbank learned using AffineComponent

class XconfigFftFilterLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        # Here we just list some likely combinations.. you can just add any
        # combinations you want to use, to this list.
        assert first_token in ['preprocess-fft-abs-lognorm-affine-log-layer',
                               'preprocess-fft-abs-norm-lognorm-affine-log-layer',
                               'preprocess-fft-abs-norm-affine-log-layer',
                               'preprocess-fft-abs-log-layer']
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = { 'input':'[-1]',
                        'dim': -1,
                        'max-change' : 0.75,
                        'target-rms' : 1.0,
                        'learning-rate-factor' : 1.0,
                        'max-change' : 0.75,
                        'max-param-value' : 1.0,
                        'min-param-value' : 0.0,
                        'l2-regularize' : 0.005,
                        'learning-rate-factor' : 1,
                        'dim' : -1,
                        'write-init-config' : True,
                        'num-filters' : 100,
                        'sin-transform-file' : '',
                        'cos-transform-file' : '',
                        'scale': 1.0,
                        'half-fft-range' : False} # l2-regularize and min-param-value
                                                   # and max-param-value affects
                                                   # layers affine layer.
    def check_configs(self):
        if self.config['target-rms'] < 0.0:
            raise RuntimeError("target-rms has invalid value {0}"
                               .format(self.config['target-rms']))
        if self.config['learning-rate-factor'] <= 0.0:
            raise RuntimeError("learning-rate-factor has invalid value {0}"
                               .format(self.config['learning-rate-factor']))
        if self.config['max-param-value'] < self.config['min-param-value']:
            raise RuntimeError("max-param-value {0} should be larger than "
                               "min-param-value {1}."
                               "".format(self.config['max-param-value'],
                                         self.config['min-param-value']))

        if self.config['sin-transform-file'] is None:
            raise RuntimeError("sin-transform-file must be set.")

        if self.config['cos-transform-file'] is None:
            raise RuntimeError("cos-transform-file must be set.")

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output == None

        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        last_nonlinearity = split_layer_name[-2]
        return '{0}.{1}'.format(self.name, last_nonlinearity)


    def output_dim(self):
        split_layer_name = self.layer_type.split('-')
        if 'affine' in split_layer_name:
            output_dim = self.config['num-filters']
            if 'norm' in split_layer_name:
                output_dim = output_dim + 1
        else:
            input_dim = self.descriptors['input']['dim']
            fft_dim = (2**(input_dim-1).bit_length())
            half_fft_range = self.config['half-fft-range']
            output_dim = (fft_dim/2 if half_fft_range is True else fft_dim)
        return output_dim

    def get_full_config(self):
        ans = []
        config_lines = self._generate_config()

        for line in config_lines:
            if len(line) == 2:
                # 'ref' or 'final' tuple already exist in the line
                # These lines correspond to fft component.
                # which contains FixedAffineComponent.
                assert(line[0] == 'init' or line[0] == 'ref' or line[0] == 'final')
                ans.append(line)
            else:
                for config_name in ['ref', 'final']:
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
        dim = self.config['dim']
        min_param_value = self.config['min-param-value']
        max_param_value = self.config['max-param-value']
        target_rms = self.config['target-rms']
        max_change = self.config['max-change']
        #ng_affine_options = self.config['ng-affine-options']
        learning_rate_factor= self.config['learning-rate-factor']
        learning_rate_option=('learning-rate-factor={0}'.format(learning_rate_factor)
                              if learning_rate_factor != 1.0 else '')
        cos_file = self.config['cos-transform-file']
        sin_file = self.config['sin-transform-file']
        num_filters = self.config['num-filters']
        l2_regularize = self.config['l2-regularize']
        half_fft_range = self.config['half-fft-range']
        fft_dim = (2**(input_dim-1).bit_length())
        cur_dim = input_dim
        cur_node = input_desc
        scale = self.config['scale']
        configs = []
        for nonlinearity in nonlinearities:
            if nonlinearity == 'preprocess':
                configs.append('component name={0}.preprocess type=ShiftInputComponent '
                               'input-dim={1} output-dim={1} dither=0.0 max-shift=0.0 '
                               'preprocess=true'.format(self.name, cur_dim))

                configs.append('component-node name={0}.preprocess '
                               'component={0}.preprocess input={1}'
                               ''.format(self.name, cur_node))
                cur_node = '{0}.preprocess'.format(self.name)

            elif nonlinearity == 'fft':
                #if self.config['write-init-config']:
                #    line = ('output-node name=output input={0}'
                #            ''.format(input_desc))
                #    configs.append(('init', line))
                output_dim = (fft_dim/2 if half_fft_range is True else fft_dim)
                line = ('component name={0}.cosine type=FixedAffineComponent '
                       'matrix={1}'
                       ''.format(self.name, cos_file))
                configs.append(('final', line))

                line = ('component name={0}.cosine type=FixedAffineComponent '
                        'input-dim={1} output-dim={2}'
                        ''.format(self.name, cur_dim, output_dim))
                configs.append(('ref', line))

                line = ('component-node name={0}.cosine component={0}.cosine '
                        'input={1}'.format(self.name, cur_node))
                configs.append(('final', line))
                configs.append(('ref', line))

                line = ('component name={0}.sine type=FixedAffineComponent '
                        'matrix={1}'.format(self.name, sin_file))
                configs.append(('final', line))

                line = ('component name={0}.sine type=FixedAffineComponent '
                        'input-dim={1} output-dim={2}'
                        ''.format(self.name, cur_dim, output_dim))
                configs.append(('ref', line))

                line = ('component-node name={0}.sine component={0}.sine '
                        'input={1}'.format(self.name, cur_node))
                configs.append(('final', line))
                configs.append(('ref', line))

                cur_node = []
                if half_fft_range:
                    cur_node.append('{0}.cosine'.format(self.name))
                    cur_node.append('{0}.sine'.format(self.name))
                else:
                    configs.append('dim-range-node name={0}.sine.half input-node={0}.sine '
                                   'dim-offset=0 dim={1}'.format(self.name, fft_dim/2))
                    configs.append('dim-range-node name={0}.cosine.half input-node={0}.cosine '
                                   'dim-offset=0 dim={1}'.format(self.name, fft_dim/2))
                    cur_node.append('{0}.cosine.half'.format(self.name))
                    cur_node.append('{0}.sine.half'.format(self.name))
                cur_dim = fft_dim / 2
            elif nonlinearity == 'abs2':
                assert(len(cur_node) == 2 and
                       cur_node[0] == '{0}.cosine'.format(self.name) and
                       cur_node[1] == '{0}.sine'.format(self.name))
                configs.append('component name={0}.cos.sqr type=ElementwiseProductComponent '
                               'input-dim={1} output-dim={2}'
                               ''.format(self.name, cur_dim * 2, cur_dim))
                configs.append('component-node name={0}.cos.sqr component={0}.cos.sqr '
                               'input=Append({1},{1})'
                               ''.format(self.name, cur_node[0]))

                configs.append('component name={0}.sin.sqr type=ElementwiseProductComponent '
                               'input-dim={1} output-dim={2}'
                               ''.format(self.name, cur_dim * 2, cur_dim))
                configs.append('component-node name={0}.sin.sqr component={0}.cos.sqr '
                               'input=Append({1},{1})'
                               ''.format(self.name, cur_node[1]))
                configs.append('component name={0}.abs type=NoOpComponent dim={1}'
                               ''.format(self.name, cur_dim))
                configs.append('component-node name={0}.abs component={0}.abs '
                               'input=Sum({0}.sin.sqr, {0}.cos.sqr)'
                               ''.format(self.name))
                cur_node = '{0}.abs'.format(self.name)

            elif nonlinearity == 'abs':
                assert(len(cur_node) == 2 and
                       cur_node[0] == '{0}.cosine'.format(self.name) and
                       cur_node[1] == '{0}.sine'.format(self.name))
                permute_vec = []
                for i in range(fft_dim/2):
                    permute_vec.append(i)
                    permute_vec.append(i+fft_dim/2)
                permute_vec_str = ','.join([str(x) for x in permute_vec])
                configs.append('component name={0}.permute type=PermuteComponent '
                               'column-map={1}'.format(self.name, permute_vec_str))
                configs.append('component-node name={0}.permute component={0}.permute '
                               'input=Append({1},{2})'
                               ''.format(self.name, cur_node[0], cur_node[1]))

                configs.append('component name={0}.abs type=PnormComponent '
                               'input-dim={1} output-dim={2}'
                               ''.format(self.name, fft_dim, fft_dim/2))
                configs.append('component-node name={0}.abs component={0}.abs '
                               'input={0}.permute'.format(self.name))
                cur_node = '{0}.abs'.format(self.name)
                cur_dim = fft_dim / 2
            elif nonlinearity == 'norm':
                assert(isinstance(cur_node, str))
                configs.append('component name={0}.norm type=NormalizeComponent '
                               'dim={1} target-rms=1.0 add-log-stddev=true '.format(self.name, cur_dim))
                configs.append('component-node name={0}.norm component={0}.norm '
                                'input={1}'.format(self.name, cur_node))
                configs.append('dim-range-node name={0}.norm.no.energy input-node={0}.norm '
                                'dim-offset=0 dim={1}'.format(self.name, cur_dim))
                configs.append('dim-range-node name={0}.norm.energy input-node={0}.norm '
                               'dim-offset={1} dim=1'.format(self.name, cur_dim))
                cur_node = '{0}.norm.no.energy'.format(self.name)
                cur_dim = fft_dim / 2
            elif nonlinearity == 'lognorm':
                assert(isinstance(cur_node, str))
                configs.append('component name={0}.norm.log type=LogComponent '
                               'dim={1} log-floor=1e-4 additive-offset=false '
                               ''.format(self.name, cur_dim))
                configs.append('component-node name={0}.norm.log component={0}.norm.log '
                               'input={1}'.format(self.name, cur_node))
                configs.append('component name={0}.norm.batch type=BatchNormComponent '
                               'dim={1} target-rms={2} '
                               ''.format(self.name, cur_dim, target_rms))
                configs.append('component-node name={0}.norm.batch '
                               'component={0}.norm.batch '
                               'input={0}.norm.log'.format(self.name))
                configs.append('component name={0}.norm.so type=ScaleAndOffsetComponent '
                               'dim={1} max-change=0.5 scale={2}'
                               ''.format(self.name, cur_dim, scale))
                configs.append('component-node name={0}.norm.so component={0}.norm.so '
                               'input={0}.norm.batch '.format(self.name))
                configs.append('component name={0}.norm.exp type=ExpComponent dim={1} '
                               ''.format(self.name, cur_dim))
                configs.append('component-node name={0}.norm.exp component={0}.norm.exp '
                               'input={0}.norm.so'.format(self.name))
                #configs.append('component name={0}.norm.exp type=ExpComponent dim={1} '
                #               ''.format(self.name, cur_dim))
                #configs.append('component-node name={0}.norm.exp component={0}.norm.exp '
                #               'input={0}.norm.batch'.format(self.name))
                cur_node = '{0}.norm.exp'.format(self.name)
                cur_dim = fft_dim / 2


            elif nonlinearity == 'lognorm2':
                configs.append("component name={0}.lognorm type=CompositeComponent "
                               "num-components=4 "
                               "component1='type=LogComponent dim={1} log-floor=1e-4 additive-offset=false' "
                               "component2='type=BatchNormComponent dim={1} target-rms={2}' "
                               "component3='type=ScaleAndOffsetComponent dim={1} max-change=0.5' "
                               "component4='type=ExpComponent dim={1}' "
                               "".format(self.name, cur_dim, target_rms))
                configs.append('component-node name={0}.lognorm '
                               'component={0}.lognorm input={1}'
                               ''.format(self.name, cur_node))

                cur_node = '{0}.lognorm'.format(self.name)
                cur_dim = fft_dim / 2

            elif nonlinearity == 'affine':
                configs.append('component name={0}.filterbank type=AffineComponent '
                               'input-dim={1} output-dim={2} max-change={3} '
                               'min-param-value={4} max-param-value={5} '
                               'bias-stddev=0.0 l2-regularize={6}'
                               ''.format(self.name, cur_dim, num_filters, max_change,
                                         min_param_value, max_param_value,
                                         l2_regularize))
                configs.append('component-node name={0}.filterbank '
                               'component={0}.filterbank input={1}'
                               ''.format(self.name, cur_node))
                cur_node = '{0}.filterbank'.format(self.name)
                cur_dim = num_filters
            elif nonlinearity == 'log':
                configs.append('component name={0}.log type=LogComponent '
                               'log-floor=1e-4 additive-offset=false dim={1}'
                               ''.format(self.name, cur_dim))

                if 'norm' in nonlinearities:
                    configs.append('component-node name={0}.log0 '
                                   'component={0}.log input={1}'
                                   ''.format(self.name, cur_node))
                    configs.append('component name={0}.log.sum type=NoOpComponent '
                                   'dim={1}'.format(self.name, cur_dim+1))
                    configs.append('component-node name={0}.log component={0}.log.sum '
                                   'input=Append({0}.log0, {0}.norm.energy)'
                                   ''.format(self.name))
                    cur_dim = fft_dim / 2 + 1
                else:
                    configs.append('component-node name={0}.log '
                                   'component={0}.log input={1}'
                                   ''.format(self.name, cur_node))
                    cur_dim = fft_dim / 2
                cur_node = '{0}.log'.format(self.name)



            else:
                raise RuntimeError("Unknown nonlinearity type: {0}"
                                   "".format(nonlinearity))
        return configs

class XconfigTimeDomainLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token in ['preprocess-tconv-abs-log-nin-affine-layer']
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input':'[-1]',
                       'dim': -1,
                       'frame-dim': 80,
                       'max-change' : 0.75,
                       'num-filters' : 100,
                       'log-floor' : 0.0001,
                       'nin-mid-dim' : 75,
                       'nin-forward-dim' : 500,
                       'sub-frames-per-frame': 8,
                       'frames-left-context':1,
                       'frames-right-context':0,
                       'max-shift': 0.2}


    def check_configs(self):
        if self.config['frames-left-context'] < 0:
            raise RuntimeError("frames-left-context should be > 0."
                               "".format(self.config['frames-left-context']))
        if self.config['frames-right-context'] < 0:
            raise RuntimeError("frames-right-context should be > 0."
                               "".format(self.config['sub-frames-right-context']))


    def output_name(self, auxiliary_output=None):
        assert auxiliary_output == None

        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        last_nonlinearity = split_layer_name[-2]
        if last_nonlinearity == 'affine':
            return '{0}.post.forward'.format(self.name)

    def output_dim(self):
        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-2] == 'affine'
        return self.config['nin-forward-dim']

    def get_full_config(self):
        ans = []
        config_lines = self._generate_config()

        for line in config_lines:
            if len(line) == 2:
                # 'ref' or 'final' tuple already exist in the line
                # These lines correspond to fft component.
                # which contains FixedAffineComponent.
                assert(line[0] == 'init' or line[0] == 'ref' or line[0] == 'final')
                ans.append(line)
            else:
                for config_name in ['ref', 'final']:
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
        dim = self.config['dim']
        frame_dim = self.config['frame-dim']
        max_change = self.config['max-change']
        nin_mid_dim = self.config['nin-mid-dim']
        pool_left_context = self.config['frames-left-context']
        pool_right_context = self.config['frames-right-context']
        nin_forward_dim = self.config['nin-forward-dim']
        log_floor = self.config['log-floor']
        num_filters  = self.config['num-filters']
        samples_per_sub_frame = frame_dim / self.config['sub-frames-per-frame']
        filter_step = samples_per_sub_frame
        filter_dim = input_dim - (frame_dim if 'preprocess' in nonlinearities else 0) - frame_dim + filter_step
        cur_node = input_desc
        cur_dim = input_dim
        configs = []
        for nonlinearity in nonlinearities:
            if nonlinearity == 'preprocess':
                configs.append('component name={0}.preprocess type=ShiftInputComponent '
                               'input-dim={1} output-dim={2} dither=0.0 max-shift={3} '
                               'preprocess=true '.format(self.name, cur_dim,
                                cur_dim - frame_dim,
                                self.config['max-shift']))

                configs.append('component-node name={0}.preprocess '
                               'component={0}.preprocess input={1}'
                               ''.format(self.name, cur_node))
                cur_node = '{0}.preprocess'.format(self.name)
                cur_dim = cur_dim - frame_dim

            elif nonlinearity == 'tconv':
                # add Convolution component and PermuteComponent
                configs.append('component name={0}.tconv type=ConvolutionComponent '
                               'input-x-dim={1}  input-y-dim=1 input-z-dim=1 '
                               'filt-x-dim={2} filt-y-dim=1 filt-x-step={3} '
                               'filt-y-step=1 num-filters={4} '
                               'input-vectorization-order=zyx param-stddev={5} '
                               'bias-stddev=0.01 max-change={6}'
                               ''.format(self.name, cur_dim, filter_dim,
                                        filter_step, num_filters,
                                        0.9 / (filter_dim**0.5),
                                        max_change))

                configs.append('component-node name={0}.tconv '
                               'component={0}.tconv input={1}'
                               ''.format(self.name, cur_node))

                # adding PermuteComponent and appending filter outputs.
                conv_output_dim = self.config['sub-frames-per-frame'] * (pool_left_context + pool_right_context + 1)
                permute_vec = []
                for i in range(num_filters):
                    for j in range(conv_output_dim):
                        permute_vec.append(i+j*num_filters)
                permute_vec_str = ','.join([str(x) for x in permute_vec])
                configs.append('component name={0}.permute type=PermuteComponent '
                               'column-map={1}'
                               ''.format(self.name, permute_vec_str))
                append_str = ','.join(['Offset({0}.tconv,{1})'.format(self.name, x) for x in range(-1*pool_left_context, pool_right_context+1)])
                configs.append('component-node name={0}.permute '
                               'component={0}.permute input=Append({1})'
                               ''.format(self.name, append_str))

                cur_node = '{0}.permute'.format(self.name)
                cur_dim = num_filters * conv_output_dim

            elif nonlinearity == 'abs':
                configs.append('component name={0}.abs type=PnormComponent '
                               'input-dim={1} output-dim={1}'
                               ''.format(self.name, cur_dim))
                configs.append('component-node name={0}.abs component={0}.abs '
                               'input={1}'.format(self.name, cur_node))

                cur_node = '{0}.abs'.format(self.name)
                cur_dim = cur_dim

            elif nonlinearity == 'log':
                configs.append('component name={0}.log type=LogComponent '
                               'dim={1} log-floor={2} additive-offset=false '
                               ''.format(self.name, cur_dim, log_floor))
                configs.append('component-node name={0}.log component={0}.log '
                               'input={1}'.format(self.name, cur_node))

                cur_node = '{0}.log'.format(self.name)
                cur_dim = cur_dim

            elif nonlinearity == 'nin':
                configs.append("component name={0}.nin type=CompositeComponent "
                               "num-components=4 "
                               "component1='type=RectifiedLinearComponent dim={1} self-repair-scale=1e-05' "
                               "component2='type=NaturalGradientRepeatedAffineComponent input-dim={1} output-dim={2} num-repeats={3} param-stddev={4} bias-stddev=0' "
                               "component3='type=RectifiedLinearComponent dim={2} self-repair-scale=1e-05' "
                               "component4='type=NaturalGradientRepeatedAffineComponent input-dim={2}  output-dim={1} num-repeats={3} param-stddev={5} bias-mean=0.1 bias-stddev=0 ' "
                               "".format(self.name, cur_dim, nin_mid_dim * num_filters,
                                         num_filters, 2.0 / (cur_dim**0.5),
                                         2.0 / (nin_mid_dim * num_filters)**0.5))

                configs.append('component-node name={0}.nin component={0}.nin '
                               'input={1}'
                               ''.format(self.name, cur_node))
                configs.append("component name={0}.post.nin type=CompositeComponent "
                               "num-components=2 component1='type=RectifiedLinearComponent dim={1} self-repair-scale=1e-05' "
                               "component2='type=NormalizeComponent dim={1} add-log-stddev=true '"
                               "".format(self.name, cur_dim))
                configs.append('component-node name={0}.post.nin component={0}.post.nin input={0}.nin'
                               ''.format(self.name))

                cur_node= '{0}.post.nin'.format(self.name)
                cur_dim = cur_dim + 1

            elif nonlinearity == 'affine':
                configs.append('component name={0}.forward.nin type=NaturalGradientAffineComponent '
                               'input-dim={1} output-dim={2} bias-stddev=0'
                               ''.format(self.name, cur_dim, nin_forward_dim))
                configs.append('component-node name={0}.forward.nin component={0}.forward.nin '
                               'input={1}'.format(self.name, cur_node))
                configs.append("component name={0}.post.forward type=CompositeComponent num-components=2 "
                               "component1='type=RectifiedLinearComponent dim={1} self-repair-scale=1e-05' "
                               "component2='type=NormalizeComponent dim={1}'"
                               "".format(self.name, nin_forward_dim))
                configs.append('component-node name={0}.post.forward component={0}.post.forward '
                               'input={0}.forward.nin'.format(self.name, cur_node))

                cur_node = '{0}.post.forward'.format(self.name)
                cur_dim = nin_forward_dim

            else:
                raise RuntimeError("Unknown nonlinearity type: {0}"
                                   "".format(nonlinearity))
        return configs
