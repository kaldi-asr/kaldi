# Copyright 2016    Johns Hopkins University (Dan Povey)
#           2016    Vijayaditya Peddinti
#           2016    Yiming Wang
# Apache 2.0.


""" This module has the implementations of different LSTM layers.
"""
from __future__ import print_function
import math
import re
import sys
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase


# This class is for lines like
#   'lstm-layer name=lstm1 input=[-1] delay=-3'
# It generates an LSTM sub-graph without output projections.
# The output dimension of the layer may be specified via 'cell-dim=xxx', but if not specified,
# the dimension defaults to the same as the input.
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   cell-dim=-1              [Dimension of the cell]
#   delay=-1                 [Delay in the recurrent connections of the LSTM ]
#   clipping-threshold=30    [nnet3 LSTMs use a gradient clipping component at the recurrent connections.
#                             This is the threshold used to decide if clipping has to be activated ]
#   zeroing-interval=20      [interval at which we (possibly) zero out the recurrent derivatives.]
#   zeroing-threshold=15     [We only zero out the derivs every zeroing-interval, if derivs exceed this value.]
#   self_repair_scale_nonlinearity=1e-5      [It is a constant scaling the self-repair vector computed in derived classes of NonlinearComponent]
#                                       i.e.,  SigmoidComponent, TanhComponent and RectifiedLinearComponent ]
#   ng-per-element-scale-options=''     [Additional options used for the diagonal matrices in the LSTM ]
#   ng-affine-options=''                [Additional options used for the full matrices in the LSTM, can be used to do things like set biases to initialize to 1]
#   decay-time=-1            [If >0, an approximate maximum on how many frames
#                            can be remembered via summation into the cell
#                            contents c_t; enforced by putting a scaling factor
#                            of recurrence_scale = 1 - abs(delay)/decay_time on
#                            the recurrence, i.e. the term c_{t-1} in the LSTM
#                            equations.  E.g. setting this to 20 means no more
#                            than about 20 frames' worth of history,
#                            i.e. history since about t = t-20, can be
#                            accumulated in c_t.]
#  l2-regularize=0.0         Constant controlling l2 regularization for this layer
class XconfigLstmLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "lstm-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input':'[-1]',
                        'cell-dim' : -1, # this is a compulsory argument
                        'clipping-threshold' : 30.0,
                        'delay' : -1,
                        'ng-per-element-scale-options' : ' max-change=0.75',
                        'ng-affine-options' : ' max-change=0.75 ',
                        'self-repair-scale-nonlinearity' : 0.00001,
                        'zeroing-interval' : 20,
                        'zeroing-threshold' : 15.0,
                       'l2-regularize': 0.0,
                        'decay-time':  -1.0
                        }

    def set_derived_configs(self):
        if self.config['cell-dim'] <= 0:
            self.config['cell-dim'] = self.descriptors['input']['dim']

    def check_configs(self):
        key = 'cell-dim'
        if self.config['cell-dim'] <= 0:
            raise RuntimeError("cell-dim has invalid value {0}.".format(self.config[key]))

        if self.config['delay'] == 0:
            raise RuntimeError("delay cannot be zero")

        for key in ['self-repair-scale-nonlinearity']:
            if self.config[key] < 0.0 or self.config[key] > 1.0:
                raise RuntimeError("{0} has invalid value {1}.".format(key, self.config[key]))

    def auxiliary_outputs(self):
        return ['c_t']

    def output_name(self, auxiliary_output = None):
        node_name = 'm_t'
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                node_name = auxiliary_output
            else:
                raise RuntimeError("Unknown auxiliary output name {0}".format(auxiliary_output))

        return '{0}.{1}'.format(self.name, node_name)

    def output_dim(self, auxiliary_output = None):
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                if node_name == 'c_t':
                    return self.config['cell-dim']
                # add code for other auxiliary_outputs here when we decide to expose them
            else:
                raise RuntimeError("Unknown auxiliary output name {0}".format(auxiliary_output))

        return self.config['cell-dim']

    def get_full_config(self):
        ans = []
        config_lines = self._generate_lstm_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in LSTM initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # convenience function to generate the LSTM config
    def _generate_lstm_config(self):

        # assign some variables to reduce verbosity
        name = self.name
        # in the below code we will just call descriptor_strings as descriptors for conciseness
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        cell_dim = self.config['cell-dim']
        delay = self.config['delay']
        decay_time = self.config['decay-time']
        # we expect decay_time to be either -1, or large, like 10 or 50.
        recurrence_scale = (1.0 if decay_time < 0 else
                            1.0 - (abs(delay) / decay_time))
        assert recurrence_scale > 0   # or user may have set decay-time much
                                      # too small.
        bptrunc_str = ("clipping-threshold={0}"
                      " zeroing-threshold={1}"
                      " zeroing-interval={2}"
                      " recurrence-interval={3}"
                      " scale={4}"
                      "".format(self.config['clipping-threshold'],
                                self.config['zeroing-threshold'],
                                self.config['zeroing-interval'],
                                abs(delay), recurrence_scale))
        repair_nonlin = self.config['self-repair-scale-nonlinearity']
        repair_nonlin_str = "self-repair-scale={0:.10f}".format(repair_nonlin) if repair_nonlin is not None else ''
        affine_str = self.config['ng-affine-options']
        # Natural gradient per element scale parameters
        ng_per_element_scale_options = self.config['ng-per-element-scale-options']
        if re.search('param-mean', ng_per_element_scale_options) is None and \
           re.search('param-stddev', ng_per_element_scale_options) is None:
           ng_per_element_scale_options += " param-mean=0.0 param-stddev=1.0 "
        pes_str = ng_per_element_scale_options
        l2_regularize = self.config['l2-regularize']
        l2_regularize_option = ('l2-regularize={0} '.format(l2_regularize)
                                if l2_regularize != 0.0 else '')


        configs = []

        # To see the equations implemented here, see
        # eqs (1)-(6) in https://arxiv.org/abs/1402.1128
        # naming convention:
        # <layer-name>.W_<outputname>.<input_name> e.g. Lstm1.W_i.xr for matrix
        # providing output to gate i and operating on an appended vector [x,r]

        configs.append("### Begin LTSM layer '{0}'".format(name))
        configs.append("# Input gate control : W_i* matrices")
        configs.append("component name={0}.W_i.xr type=NaturalGradientAffineComponent input-dim={1} "
                       "output-dim={2} {3} {4}".format(name, input_dim + cell_dim, cell_dim,
                                                       affine_str, l2_regularize_option))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_i.c type=NaturalGradientPerElementScaleComponent "
                       "dim={1} {2} {3} ".format(name, cell_dim, pes_str,
                                                 l2_regularize_option))
        configs.append("# Forget gate control : W_f* matrices")
        configs.append("component name={0}.W_f.xr type=NaturalGradientAffineComponent input-dim={1} "
                       "output-dim={2} {3} {4}".format(name, input_dim + cell_dim, cell_dim,
                                                       affine_str, l2_regularize_option))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_f.c type=NaturalGradientPerElementScaleComponent "
                       "dim={1} {2} {3}".format(name, cell_dim, pes_str, l2_regularize_option))

        configs.append("#  Output gate control : W_o* matrices")
        configs.append("component name={0}.W_o.xr type=NaturalGradientAffineComponent input-dim={1} "
                       "output-dim={2} {3} {4}".format(name, input_dim + cell_dim, cell_dim,
                                                       affine_str, l2_regularize_option))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_o.c type=NaturalGradientPerElementScaleComponent "
                       " dim={1} {2} {3}".format(name, cell_dim, pes_str,
                                                 l2_regularize_option))

        configs.append("# Cell input matrices : W_c* matrices")
        configs.append("component name={0}.W_c.xr type=NaturalGradientAffineComponent input-dim={1} "
                       "output-dim={2} {3} {4}".format(name, input_dim + cell_dim, cell_dim,
                                                       affine_str, l2_regularize_option))


        configs.append("# Defining the non-linearities")
        configs.append("component name={0}.i type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.f type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.o type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.g type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.h type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))

        configs.append("# Defining the components for other cell computations")
        configs.append("component name={0}.c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}"
                       "".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}"
                       "".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.m type=ElementwiseProductComponent input-dim={1} output-dim={2}"
                       "".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.c type=BackpropTruncationComponent dim={1} {2}"
                       "".format(name, cell_dim, bptrunc_str))

        # c1_t and c2_t defined below
        configs.append("component-node name={0}.c_t component={0}.c input=Sum({0}.c1_t, {0}.c2_t)".format(name))
        delayed_c_t_descriptor = "IfDefined(Offset({0}.c_t, {1}))".format(name, delay)

        configs.append("# i_t")
        configs.append("component-node name={0}.i1_t component={0}.W_i.xr input=Append({1}, IfDefined(Offset({0}.r_t, {2})))"
                       "".format(name, input_descriptor, delay))
        configs.append("component-node name={0}.i2_t component={0}.w_i.c  input={1}".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.i_t component={0}.i input=Sum({0}.i1_t, {0}.i2_t)".format(name))

        configs.append("# f_t")
        configs.append("component-node name={0}.f1_t component={0}.W_f.xr input=Append({1}, IfDefined(Offset({0}.r_t, {2})))"
                       "".format(name, input_descriptor, delay))
        configs.append("component-node name={0}.f2_t component={0}.w_f.c  input={1}".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.f_t component={0}.f input=Sum({0}.f1_t, {0}.f2_t)".format(name))

        configs.append("# o_t")
        configs.append("component-node name={0}.o1_t component={0}.W_o.xr input=Append({1}, IfDefined(Offset({0}.r_t, {2})))"
                       "".format(name, input_descriptor, delay))
        configs.append("component-node name={0}.o2_t component={0}.w_o.c input={0}.c_t".format(name))
        configs.append("component-node name={0}.o_t component={0}.o input=Sum({0}.o1_t, {0}.o2_t)".format(name))

        configs.append("# h_t")
        configs.append("component-node name={0}.h_t component={0}.h input={0}.c_t".format(name))

        configs.append("# g_t")
        configs.append("component-node name={0}.g1_t component={0}.W_c.xr input=Append({1}, IfDefined(Offset({0}.r_t, {2})))"
                       "".format(name, input_descriptor, delay))
        configs.append("component-node name={0}.g_t component={0}.g input={0}.g1_t".format(name))

        configs.append("# parts of c_t")
        configs.append("component-node name={0}.c1_t component={0}.c1  input=Append({0}.f_t, {1})"
                       "".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.c2_t component={0}.c2 input=Append({0}.i_t, {0}.g_t)"
                       "".format(name))

        configs.append("# m_t")
        configs.append("component-node name={0}.m_t component={0}.m input=Append({0}.o_t, {0}.h_t)"
                       "".format(name))

        # add the recurrent connections
        configs.append("component name={0}.r type=BackpropTruncationComponent dim={1} {2}"
                       "".format(name, cell_dim, bptrunc_str))
        configs.append("component-node name={0}.r_t component={0}.r input={0}.m_t".format(name))
        configs.append("### End LTSM layer '{0}'".format(name))
        return configs


# This class is for lines like
#   'lstmp-layer name=lstm1 input=[-1] delay=-3'
# (you can also use the name 'lstmp-batchnorm-layer' if you want it to be followed
# by batchnorm).
# It generates an LSTM sub-graph with output projections. It can also generate
# outputs without projection, but you could use the XconfigLstmLayer for this
# simple LSTM.
# The output dimension of the layer may be specified via 'cell-dim=xxx', but if not specified,
# the dimension defaults to the same as the input.
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   cell-dim=-1            [Dimension of the cell]
#   recurrent-projection_dim [Dimension of the projection used in recurrent connections, e.g. cell-dim/4]
#   non-recurrent-projection-dim   [Dimension of the projection in non-recurrent connections,
#                                   in addition to recurrent-projection-dim, e.g. cell-dim/4]
#   delay=-1                 [Delay in the recurrent connections of the LSTM ]
#   clipping-threshold=30    [nnet3 LSTMs use a gradient clipping component at the recurrent connections.
#                             This is the threshold used to decide if clipping has to be activated ]
#   zeroing-interval=20      [interval at which we (possibly) zero out the recurrent derivatives.]
#   zeroing-threshold=15     [We only zero out the derivs every zeroing-interval, if derivs exceed this value.]
#   self_repair_scale_nonlinearity=1e-5      [It is a constant scaling the self-repair vector computed in derived classes of NonlinearComponent]
#                                       i.e.,  SigmoidComponent, TanhComponent and RectifiedLinearComponent ]
#   ng-per-element-scale-options=''   [Additional options used for the diagonal matrices in the LSTM ]
#   ng-affine-options=''              [Additional options used for the full matrices in the LSTM, can be used to do things like set biases to initialize to 1]
#   decay-time=-1            [If >0, an approximate maximum on how many frames
#                            can be remembered via summation into the cell
#                            contents c_t; enforced by putting a scaling factor
#                            of recurrence_scale = 1 - abs(delay)/decay_time on
#                            the recurrence, i.e. the term c_{t-1} in the LSTM
#                            equations.  E.g. setting this to 20 means no more
#                            than about 20 frames' worth of history,
#                            i.e. history since about t = t-20, can be
#                            accumulated in c_t.]
#  l2-regularize=0.0         Constant controlling l2 regularization for this layer
class XconfigLstmpLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        # lstmp-batchnorm-layer is like lstmp-layer but followed by a batchnorm
        # component.
        assert first_token in ["lstmp-layer", "lstmp-batchnorm-layer"]
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input' : '[-1]',
                        'cell-dim' : -1, # this is a compulsory argument
                        'recurrent-projection-dim' : -1,  # defaults to cell-dim / 4
                        'non-recurrent-projection-dim' : -1, # defaults to
                                                             # recurrent-projection-dim
                        'clipping-threshold' : 30.0,
                        'delay' : -1,
                        'ng-per-element-scale-options' : ' max-change=0.75 ',
                        'ng-affine-options' : ' max-change=0.75 ',
                        'self-repair-scale-nonlinearity' : 0.00001,
                        'zeroing-interval' : 20,
                        'zeroing-threshold' : 15.0,
                        'dropout-proportion' : -1.0, # If -1.0, no dropout components will be added
                        'dropout-per-frame' : False,  # If false, regular dropout, not per frame.
                        'decay-time':  -1.0,
                       'l2-regularize': 0.0,
                       }

    def set_derived_configs(self):
        if self.config['recurrent-projection-dim'] <= 0:
            self.config['recurrent-projection-dim'] = self.config['cell-dim'] / 4

        if self.config['non-recurrent-projection-dim'] <= 0:
            self.config['non-recurrent-projection-dim'] = \
               self.config['recurrent-projection-dim']

    def check_configs(self):
        for key in ['cell-dim', 'recurrent-projection-dim',
                    'non-recurrent-projection-dim']:
            if self.config[key] <= 0:
                raise RuntimeError("{0} has invalid value {1}.".format(
                    key, self.config[key]))

        if self.config['delay'] == 0:
            raise RuntimeError("delay cannot be zero")

        if (self.config['recurrent-projection-dim'] +
            self.config['non-recurrent-projection-dim'] >
            self.config['cell-dim']):
            raise RuntimeError("recurrent+non-recurrent projection dim exceeds "
                                "cell dim.")
        for key in ['self-repair-scale-nonlinearity']:
            if self.config[key] < 0.0 or self.config[key] > 1.0:
                raise RuntimeError("{0} has invalid value {2}."
                                   .format(self.layer_type, key,
                                           self.config[key]))

        if ((self.config['dropout-proportion'] > 1.0 or
             self.config['dropout-proportion'] < 0.0) and
             self.config['dropout-proportion'] != -1.0 ):
             raise RuntimeError("dropout-proportion has invalid value {0}."
                                .format(self.config['dropout-proportion']))

    def auxiliary_outputs(self):
        return ['c_t']

    def output_name(self, auxiliary_output = None):
        node_name = ( 'rp_t_batchnorm' if self.layer_type == 'lstmp-batchnorm-layer'
                      else 'rp_t' )
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                node_name = auxiliary_output
            else:
                raise Exception("In {0} of type {1}, unknown auxiliary output name {1}".format(self.layer_type, auxiliary_output))

        return '{0}.{1}'.format(self.name, node_name)

    def output_dim(self, auxiliary_output = None):
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                if node_name == 'c_t':
                    return self.config['cell-dim']
                # add code for other auxiliary_outputs here when we decide to expose them
            else:
                raise Exception("In {0} of type {1}, unknown auxiliary output name {1}".format(self.layer_type, auxiliary_output))

        return self.config['recurrent-projection-dim'] + self.config['non-recurrent-projection-dim']

    def get_full_config(self):
        ans = []
        config_lines = self._generate_lstm_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in LSTM initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # convenience function to generate the LSTM config
    def _generate_lstm_config(self):

        # assign some variables to reduce verbosity
        name = self.name
        # in the below code we will just call descriptor_strings as descriptors for conciseness
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        cell_dim = self.config['cell-dim']
        rec_proj_dim = self.config['recurrent-projection-dim']
        nonrec_proj_dim = self.config['non-recurrent-projection-dim']
        delay = self.config['delay']
        repair_nonlin = self.config['self-repair-scale-nonlinearity']
        repair_nonlin_str = "self-repair-scale={0:.10f}".format(repair_nonlin) if repair_nonlin is not None else ''
        decay_time = self.config['decay-time']
        # we expect decay_time to be either -1, or large, like 10 or 50.
        recurrence_scale = (1.0 if decay_time < 0 else
                            1.0 - (abs(delay) / decay_time))
        assert recurrence_scale > 0   # or user may have set decay-time much
                                      # too small.
        bptrunc_str = ("clipping-threshold={0}"
                      " zeroing-threshold={1}"
                      " zeroing-interval={2}"
                      " recurrence-interval={3}"
                      " scale={4}"
                      "".format(self.config['clipping-threshold'],
                                self.config['zeroing-threshold'],
                                self.config['zeroing-interval'],
                                abs(delay), recurrence_scale))
        affine_str = self.config['ng-affine-options']
        pes_str = self.config['ng-per-element-scale-options']
        dropout_proportion = self.config['dropout-proportion']
        dropout_per_frame = 'true' if self.config['dropout-per-frame'] else 'false'

        # Natural gradient per element scale parameters
        if re.search('param-mean', pes_str) is None and \
           re.search('param-stddev', pes_str) is None:
           pes_str += " param-mean=0.0 param-stddev=1.0 "
        l2_regularize = self.config['l2-regularize']
        l2_regularize_option = ('l2-regularize={0} '.format(l2_regularize)
                                if l2_regularize != 0.0 else '')

        configs = []

        # the equations implemented here are from Sak et. al. "Long Short-Term
        # Memory Recurrent Neural Network Architectures for Large Scale Acoustic
        # Modeling"
        # https://arxiv.org/pdf/1402.1128.pdf
        # See equations (7) to (14).
        # naming convention <layer-name>.W_<outputname>.<input_name>
        # e.g. Lstm1.W_i.xr for matrix providing output to gate i and operating
        # on an appended vector [x,r]
        configs.append("# Input gate control : W_i* matrices")
        configs.append("component name={0}.W_i.xr type=NaturalGradientAffineComponent input-dim={1} "
                       "output-dim={2} {3} {4}".format(name, input_dim + rec_proj_dim,
                                                       cell_dim, affine_str, l2_regularize_option))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_i.c type=NaturalGradientPerElementScaleComponent "
                       "dim={1} {2} {3}".format(name, cell_dim, pes_str,
                                                l2_regularize_option))
        configs.append("# Forget gate control : W_f* matrices")
        configs.append("component name={0}.W_f.xr type=NaturalGradientAffineComponent input-dim={1} "
                       "output-dim={2} {3} {4}".format(name, input_dim + rec_proj_dim, cell_dim,
                                                       affine_str, l2_regularize_option))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_f.c type=NaturalGradientPerElementScaleComponent  "
                       "dim={1} {2} {3}".format(name, cell_dim, pes_str, l2_regularize_option))

        configs.append("#  Output gate control : W_o* matrices")
        configs.append("component name={0}.W_o.xr type=NaturalGradientAffineComponent input-dim={1} "
                       "output-dim={2} {3} {4}".format(name, input_dim + rec_proj_dim, cell_dim,
                                                       affine_str, l2_regularize_option))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_o.c type=NaturalGradientPerElementScaleComponent "
                       "dim={1} {2} {3}".format(name, cell_dim, pes_str, l2_regularize_option))

        configs.append("# Cell input matrices : W_c* matrices")
        configs.append("component name={0}.W_c.xr type=NaturalGradientAffineComponent input-dim={1} "
                       "output-dim={2} {3} {4}".format(name, input_dim + rec_proj_dim, cell_dim,
                                                       affine_str, l2_regularize_option))

        configs.append("# Defining the non-linearities")
        configs.append("component name={0}.i type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.f type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.o type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.g type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.h type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        if dropout_proportion != -1.0:
            configs.append("component name={0}.dropout type=DropoutComponent dim={1} "
                           "dropout-proportion={2} dropout-per-frame={3}"
                           .format(name, cell_dim, dropout_proportion, dropout_per_frame))
        configs.append("# Defining the components for other cell computations")
        configs.append("component name={0}.c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}"
                       "".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}"
                       "".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.m type=ElementwiseProductComponent input-dim={1} output-dim={2}"
                       "".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.c type=BackpropTruncationComponent dim={1} {2}"
                       "".format(name, cell_dim, bptrunc_str))

        # c1_t and c2_t defined below
        configs.append("component-node name={0}.c_t component={0}.c input=Sum({0}.c1_t, {0}.c2_t)".format(name))
        delayed_c_t_descriptor = "IfDefined(Offset({0}.c_t, {1}))".format(name, delay)

        recurrent_connection = '{0}.r_t'.format(name)
        configs.append("# i_t")
        configs.append("component-node name={0}.i1_t component={0}.W_i.xr input=Append({1}, IfDefined(Offset({2}, {3})))"
                       "".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.i2_t component={0}.w_i.c  input={1}".format(name, delayed_c_t_descriptor))
        if dropout_proportion != -1.0:
            configs.append("component-node name={0}.i_t_predrop component={0}.i input=Sum({0}.i1_t, {0}.i2_t)".format(name))
            configs.append("component-node name={0}.i_t component={0}.dropout input={0}.i_t_predrop".format(name))
        else:
            configs.append("component-node name={0}.i_t component={0}.i input=Sum({0}.i1_t, {0}.i2_t)".format(name))

        configs.append("# f_t")
        configs.append("component-node name={0}.f1_t component={0}.W_f.xr input=Append({1}, IfDefined(Offset({2}, {3})))"
                       "".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.f2_t component={0}.w_f.c  input={1}".format(name, delayed_c_t_descriptor))
        if dropout_proportion != -1.0:
            configs.append("component-node name={0}.f_t_predrop component={0}.f input=Sum({0}.f1_t, {0}.f2_t)".format(name))
            configs.append("component-node name={0}.f_t component={0}.dropout input={0}.f_t_predrop".format(name))
        else:
            configs.append("component-node name={0}.f_t component={0}.f input=Sum({0}.f1_t, {0}.f2_t)".format(name))

        configs.append("# o_t")
        configs.append("component-node name={0}.o1_t component={0}.W_o.xr input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.o2_t component={0}.w_o.c input={0}.c_t".format(name))
        if dropout_proportion != -1.0:
            configs.append("component-node name={0}.o_t_predrop component={0}.o input=Sum({0}.o1_t, {0}.o2_t)".format(name))
            configs.append("component-node name={0}.o_t component={0}.dropout input={0}.o_t_predrop".format(name))
        else:
            configs.append("component-node name={0}.o_t component={0}.o input=Sum({0}.o1_t, {0}.o2_t)".format(name))

        configs.append("# h_t")
        configs.append("component-node name={0}.h_t component={0}.h input={0}.c_t".format(name))

        configs.append("# g_t")
        configs.append("component-node name={0}.g1_t component={0}.W_c.xr input=Append({1}, IfDefined(Offset({2}, {3})))"
                       "".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.g_t component={0}.g input={0}.g1_t".format(name))

        configs.append("# parts of c_t")
        configs.append("component-node name={0}.c1_t component={0}.c1  input=Append({0}.f_t, {1})".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.c2_t component={0}.c2 input=Append({0}.i_t, {0}.g_t)".format(name))

        configs.append("# m_t")
        configs.append("component-node name={0}.m_t component={0}.m input=Append({0}.o_t, {0}.h_t)".format(name))

        # add the recurrent connections
        configs.append("# projection matrices : Wrm and Wpm")
        configs.append("component name={0}.W_rp.m type=NaturalGradientAffineComponent input-dim={1} "
                       "output-dim={2} {3} {4}".format(name, cell_dim, rec_proj_dim + nonrec_proj_dim,
                                                       affine_str, l2_regularize_option))
        configs.append("component name={0}.r type=BackpropTruncationComponent dim={1} {2}"
                       "".format(name, rec_proj_dim, bptrunc_str))

        configs.append("# r_t and p_t : rp_t will be the output (if we're not doing batchnorm)")
        configs.append("component-node name={0}.rp_t component={0}.W_rp.m input={0}.m_t"
                       "".format(name))
        configs.append("dim-range-node name={0}.r_t_preclip input-node={0}.rp_t dim-offset=0 "
                       "dim={1}".format(name, rec_proj_dim))
        configs.append("component-node name={0}.r_t component={0}.r input={0}.r_t_preclip".format(name))

        if self.layer_type == "lstmp-batchnorm-layer":
            # Add the batchnorm component, if requested to include batchnorm.
            configs.append("component name={0}.rp_t_batchnorm type=BatchNormComponent dim={1} ".format(
                name, rec_proj_dim + nonrec_proj_dim))
            configs.append("component-node name={0}.rp_t_batchnorm component={0}.rp_t_batchnorm "
                           "input={0}.rp_t".format(name))

        return configs


# This class is for lines like
#   'fast-lstm-layer name=lstm1 input=[-1] delay=-3'
# (you can also use the name 'fast-lstm-batchnorm-layer' if you want it to be followed
# by batchnorm).
# It generates an LSTM sub-graph without output projections.
# Unlike 'lstm-layer', the core nonlinearities of the LSTM are done in a special-purpose
# component (LstmNonlinearityComponent), and most of the affine parts of the LSTM are combined
# into one.
#
# The output dimension of the layer may be specified via 'cell-dim=xxx', but if not specified,
# the dimension defaults to the same as the input.
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   cell-dim=-1              [Dimension of the cell]
#   delay=-1                 [Delay in the recurrent connections of the LSTM ]
#   clipping-threshold=30    [nnet3 LSTMs use a gradient clipping component at the recurrent connections.
#                             This is the threshold used to decide if clipping has to be activated ]
#   zeroing-interval=20      [interval at which we (possibly) zero out the recurrent derivatives.]
#   zeroing-threshold=15     [We only zero out the derivs every zeroing-interval, if derivs exceed this value.]
#   lstm-nonlinearity-options=' max-change=0.75 '  [Options string to pass into the LSTM nonlinearity component.]
#   ng-affine-options=' max-change=1.5 '           [Additional options used for the full matrices in the LSTM, can be used to
#                                      do things like set biases to initialize to 1]
#   decay-time=-1            [If >0, an approximate maximum on how many frames
#                            can be remembered via summation into the cell
#                            contents c_t; enforced by putting a scaling factor
#                            of recurrence_scale = 1 - abs(delay)/decay_time on
#                            the recurrence, i.e. the term c_{t-1} in the LSTM
#                            equations.  E.g. setting this to 20 means no more
#                            than about 20 frames' worth of history,
#                            i.e. history since about t = t-20, can be
#                            accumulated in c_t.]
#  l2-regularize=0.0         Constant controlling l2 regularization for this layer
class XconfigFastLstmLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token in ["fast-lstm-layer", "fast-lstm-batchnorm-layer"]
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input':'[-1]',
                        'cell-dim' : -1, # this is a compulsory argument
                        'clipping-threshold' : 30.0,
                        'zeroing-interval' : 20,
                        'zeroing-threshold' : 15.0,
                        'delay' : -1,
                        # if you want to set 'self-repair-scale' (c.f. the
                        # self-repair-scale-nonlinearity config value in older LSTM layers), you can
                        # add 'self-repair-scale=xxx' to
                        # lstm-nonlinearity-options.
                        'lstm-nonlinearity-options' : ' max-change=0.75',
                        # the affine layer contains 4 of our old layers -> use a
                        # larger max-change than the normal value of 0.75.
                        'ng-affine-options' : ' max-change=1.5',
                        'l2-regularize': 0.0,
                        'decay-time':  -1.0
                        }
        self.c_needed = False  # keep track of whether the 'c' output is needed.

    def set_derived_configs(self):
        if self.config['cell-dim'] <= 0:
            self.config['cell-dim'] = self.descriptors['input']['dim']

    def check_configs(self):
        key = 'cell-dim'
        if self.config['cell-dim'] <= 0:
            raise RuntimeError("cell-dim has invalid value {0}.".format(self.config[key]))
        if self.config['delay'] == 0:
            raise RuntimeError("delay cannot be zero")



    def auxiliary_outputs(self):
        return ['c']

    def output_name(self, auxiliary_output = None):
        node_name = ('m_batchnorm' if self.layer_type == 'fast-lstm-batchnorm-layer'
                      else 'm')
        if auxiliary_output is not None:
            if auxiliary_output == 'c':
                node_name = 'c'
                self.c_needed = True
            else:
                raise RuntimeError("Unknown auxiliary output name {0}".format(auxiliary_output))
        return '{0}.{1}'.format(self.name, node_name)

    def output_dim(self, auxiliary_output = None):
        if auxiliary_output is not None:
            if auxiliary_output == 'c':
                self.c_needed = True
                return self.config['cell-dim']
                # add code for other auxiliary_outputs here when we decide to expose them
            else:
                raise RuntimeError("Unknown auxiliary output name {0}".format(auxiliary_output))
        return self.config['cell-dim']

    def get_full_config(self):
        ans = []
        config_lines = self._generate_lstm_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in LSTM initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # convenience function to generate the LSTM config
    def _generate_lstm_config(self):

        # assign some variables to reduce verbosity
        name = self.name
        # in the below code we will just call descriptor_strings as descriptors for conciseness
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        cell_dim = self.config['cell-dim']
        delay = self.config['delay']
        affine_str = self.config['ng-affine-options']
        l2_regularize = self.config['l2-regularize']
        l2_regularize_option = ('l2-regularize={0} '.format(l2_regularize)
                                if l2_regularize != 0.0 else '')
        decay_time = self.config['decay-time']
        # we expect decay_time to be either -1, or large, like 10 or 50.
        recurrence_scale = (1.0 if decay_time < 0 else
                            1.0 - (abs(delay) / decay_time))
        assert recurrence_scale > 0   # or user may have set decay-time much
                                      # too small.
        bptrunc_str = ("clipping-threshold={0}"
                      " zeroing-threshold={1}"
                      " zeroing-interval={2}"
                      " recurrence-interval={3}"
                      " scale={4}"
                      "".format(self.config['clipping-threshold'],
                                self.config['zeroing-threshold'],
                                self.config['zeroing-interval'],
                                abs(delay), recurrence_scale))
        lstm_str = self.config['lstm-nonlinearity-options']


        configs = []

        # the equations implemented here are equations (1) through (6) of
        # https://arxiv.org/pdf/1402.1128.pdf.
        # naming convention
        # <layer-name>.W_<outputname>.<input_name> e.g. Lstm1.W_i.xr for matrix
        # providing output to gate i and operating on an appended vector [x,r]
        configs.append("### Begin LTSM layer '{0}'".format(name))
        configs.append("# Gate control: contains W_i, W_f, W_c and W_o matrices as blocks.")

        configs.append("component name={0}.W_all type=NaturalGradientAffineComponent input-dim={1} "
                       "output-dim={2} {3} {4}".format(name, input_dim + cell_dim, cell_dim * 4,
                                                       affine_str, l2_regularize_option))

        configs.append("# The core LSTM nonlinearity, implemented as a single component.")
        configs.append("# Input = (i_part, f_part, c_part, o_part, c_{t-1}), output = (c_t, m_t)")
        configs.append("# See cu-math.h:ComputeLstmNonlinearity() for details.")
        configs.append("component name={0}.lstm_nonlin type=LstmNonlinearityComponent "
                       "cell-dim={1} {2} {3}".format(name, cell_dim, lstm_str,
                                                     l2_regularize_option))

        configs.append("# Component for backprop truncation, to avoid gradient blowup in long training examples.")
        configs.append("component name={0}.cm_trunc type=BackpropTruncationComponent dim={1} "
                       "{2}".format(name, 2 * cell_dim, bptrunc_str))

        configs.append("###  Nodes for the components above.")
        configs.append("component-node name={0}.W_all component={0}.W_all input=Append({1}, "
                       "IfDefined(Offset({0}.m_trunc, {2})))".format(
                           name, input_descriptor, delay))

        configs.append("component-node name={0}.lstm_nonlin component={0}.lstm_nonlin "
                       "input=Append({0}.W_all, IfDefined(Offset({0}.c_trunc, {1})))".format(
                           name, delay))
        # we can print .c later if needed, but it generates a warning since it's not used.  could use c_trunc instead
        #configs.append("dim-range-node name={0}.c input-node={0}.lstm_nonlin dim-offset=0 dim={1}".format(name, cell_dim))
        configs.append("dim-range-node name={0}.m input-node={0}.lstm_nonlin dim-offset={1} dim={1}".format(name, cell_dim))
        configs.append("component-node name={0}.cm_trunc component={0}.cm_trunc input={0}.lstm_nonlin".format(name))
        configs.append("dim-range-node name={0}.c_trunc input-node={0}.cm_trunc dim-offset=0 dim={1}".format(name, cell_dim))
        configs.append("dim-range-node name={0}.m_trunc input-node={0}.cm_trunc dim-offset={1} dim={1}".format(name, cell_dim))

        if self.layer_type == "fast-lstm-batchnorm-layer":
            # Add the batchnorm component, if requested to include batchnorm.
            configs.append("component name={0}.m_batchnorm type=BatchNormComponent dim={1} ".format(
                name, cell_dim))
            configs.append("component-node name={0}.m_batchnorm component={0}.m_batchnorm "
                           "input={0}.m".format(name))
        configs.append("### End LTSM layer '{0}'".format(name))
        return configs



# This class is for lines like
#   'lstmb-layer name=lstm1 input=[-1] delay=-3'
#
# LSTMB is not something we've published; it's LSTM with a bottleneck in the
# middle of the W_all matrix (where W_all is a matrix that combines the 8 full
# matrices of standard LSTM).  W_all is factored into W_all_a and W_all_b, where
# W_all_a is constrained to have orthonormal rows (this keeps it training stably).
#
# It also contains a couple of other improvements: W_all_b is followed by
# trainable ScaleAndOffsetComponent (this is a bit like the idea from the
# publication "Self-stabilized deep neural network" by Ghahramani et al).
# And the LSTM is followed by a batchnorm component (this is by default; it's not
# part of the layer name, like lstmb-batchnorm-layer).

#
# The output dimension of the layer may be specified via 'cell-dim=xxx', but if not specified,
# the dimension defaults to the same as the input.
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   cell-dim=-1              [Dimension of the cell]
#   bottleneck-dim=-1        [Bottleneck dim, should be less than cell-dim plus the input dim.]
#   delay=-1                 [Delay in the recurrent connections of the LSTM ]
#   clipping-threshold=30    [nnet3 LSTMs use a gradient clipping component at the recurrent connections.
#                             This is the threshold used to decide if clipping has to be activated ]
#   zeroing-interval=20      [interval at which we (possibly) zero out the recurrent derivatives.]
#   zeroing-threshold=15     [We only zero out the derivs every zeroing-interval, if derivs exceed this value.]
#   lstm-nonlinearity-options=' max-change=0.75 '  [Options string to pass into the LSTM nonlinearity component.]
#   ng-affine-options=' max-change=1.5 '           [Additional options used for the full matrices in the LSTM, can be used to
#                                      do things like set biases to initialize to 1]
#   decay-time=-1            [If >0, an approximate maximum on how many frames
#                            can be remembered via summation into the cell
#                            contents c_t; enforced by putting a scaling factor
#                            of recurrence_scale = 1 - abs(delay)/decay_time on
#                            the recurrence, i.e. the term c_{t-1} in the LSTM
#                            equations.  E.g. setting this to 20 means no more
#                            than about 20 frames' worth of history,
#                            i.e. history since about t = t-20, can be
#                            accumulated in c_t.]
#  l2-regularize=0.0         Constant controlling l2 regularization for this layer
class XconfigLstmbLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == 'lstmb-layer'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = { 'input':'[-1]',
                        'cell-dim' : -1, # this is a required argument
                        'bottleneck-dim': -1, # this is a required argument
                        'clipping-threshold': 30.0,
                        'zeroing-interval': 20,
                        'zeroing-threshold': 15.0,
                        'orthonormal-constraint': 1.0,
                        'delay' : -1,
                        'lstm-nonlinearity-options' : ' max-change=0.75',
                        # the recurrence scale is the scale on m_trunc, used in the
                        # recurrence (to balance its size with the input).
                        'self-scale' : 1.0,
                        # the affine layer contains 4 of our old layers -> use a
                        # larger max-change than the normal value of 0.75.
                        'ng-affine-options' : ' max-change=1.5',
                        'l2-regularize': 0.0,
                        'decay-time':  -1.0
                        }

    def set_derived_configs(self):
        if self.config['cell-dim'] <= 0:
            self.config['cell-dim'] = self.descriptors['input']['dim']

    def check_configs(self):
        if self.config['cell-dim'] <= 0:
            raise RuntimeError("cell-dim has invalid value {0}.".format(
                self.config['cell-dim']))
        if self.config['bottleneck-dim'] <= 0:
            raise RuntimeError("bottleneck-dim has invalid value {0}.".format(
                self.config['bottleneck-dim']))
        if self.config['delay'] == 0:
            raise RuntimeError("delay cannot be zero")

    def auxiliary_outputs(self):
        return []

    def output_name(self, auxiliary_output = None):
        assert auxiliary_output is None
        return '{0}.m_batchnorm'.format(self.name)

    def output_dim(self, auxiliary_output = None):
        assert auxiliary_output is None
        return self.config['cell-dim']

    def get_full_config(self):
        ans = []
        config_lines = self._generate_lstm_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in LSTM initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # convenience function to generate the LSTM config
    def _generate_lstm_config(self):

        # assign some variables to reduce verbosity
        name = self.name
        # in the below code we will just call descriptor_strings as descriptors for conciseness
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        cell_dim = self.config['cell-dim']
        bottleneck_dim = self.config['bottleneck-dim']
        self_scale = self.config['self-scale']
        delay = self.config['delay']
        affine_str = self.config['ng-affine-options']
        l2_regularize = self.config['l2-regularize']
        l2_regularize_option = ('l2-regularize={0} '.format(l2_regularize)
                                if l2_regularize != 0.0 else '')
        decay_time = self.config['decay-time']
        # we expect decay_time to be either -1, or large, like 10 or 50.
        recurrence_scale = (1.0 if decay_time < 0 else
                            1.0 - (abs(delay) / decay_time))
        assert recurrence_scale > 0   # or user may have set decay-time much
                                      # too small.
        bptrunc_str = ("clipping-threshold={0}"
                      " zeroing-threshold={1}"
                      " zeroing-interval={2}"
                      " recurrence-interval={3}"
                      " scale={4}"
                      "".format(self.config['clipping-threshold'],
                                self.config['zeroing-threshold'],
                                self.config['zeroing-interval'],
                                abs(delay), recurrence_scale))
        lstm_str = self.config['lstm-nonlinearity-options']


        configs = []

        # See XconfigFastLstmLayer to understand what's going on here.  This
        # differs from that code by a factorization of the W_all matrix into two
        # pieces with a smaller dimension in between (with the first of the two
        # pieces constrained to have orthonormal rows).  Note: we don't apply l2
        # regularization to this layer, since, with the orthonormality
        # constraint, it's meaningless.
        configs.append("### Begin LTSM layer '{0}'".format(name))
        configs.append("component name={0}.W_all_a type=LinearComponent input-dim={1} "
                       "orthonormal-constraint={2} output-dim={3} {4}".format(
                           name, input_dim + cell_dim,
                           self.config['orthonormal-constraint'],
                           bottleneck_dim, affine_str))

        configs.append("component name={0}.W_all_b type=LinearComponent input-dim={1} "
                       "output-dim={2} {3} {4}".format(name, bottleneck_dim, cell_dim * 4,
                                                       affine_str, l2_regularize_option))
        configs.append("component name={0}.W_all_b_so type=ScaleAndOffsetComponent dim={1} "
                       "max-change=0.75".format(name, cell_dim * 4))


        configs.append("# The core LSTM nonlinearity, implemented as a single component.")
        configs.append("# Input = (i_part, f_part, c_part, o_part, c_{t-1}), output = (c_t, m_t)")
        configs.append("# See cu-math.h:ComputeLstmNonlinearity() for details.")
        configs.append("component name={0}.lstm_nonlin type=LstmNonlinearityComponent "
                       "cell-dim={1} {2} {3}".format(name, cell_dim, lstm_str,
                                                     l2_regularize_option))
        configs.append("# Component for backprop truncation, to avoid gradient blowup in long training examples.")

        configs.append("component name={0}.cm_trunc type=BackpropTruncationComponent dim={1} {2}".format(
            name, 2 * cell_dim, bptrunc_str))
        configs.append("component name={0}.m_batchnorm type=BatchNormComponent dim={1} ".format(
            name, cell_dim))

        configs.append("###  Nodes for the components above.")
        configs.append("component-node name={0}.W_all_a component={0}.W_all_a input=Append({1}, "
                       "IfDefined(Offset(Scale({2}, {0}.m_trunc), {3})))".format(
                           name, input_descriptor, self_scale, delay))
        configs.append("component-node name={0}.W_all_b component={0}.W_all_b "
                       "input={0}.W_all_a".format(name))
        configs.append("component-node name={0}.W_all_b_so component={0}.W_all_b_so "
                       "input={0}.W_all_b".format(name))

        configs.append("component-node name={0}.lstm_nonlin component={0}.lstm_nonlin "
                       "input=Append({0}.W_all_b_so, IfDefined(Offset({0}.c_trunc, {1})))".format(
                           name, delay))
        configs.append("dim-range-node name={0}.m input-node={0}.lstm_nonlin dim-offset={1} "
                       "dim={1}".format(name, cell_dim))
        configs.append("component-node name={0}.cm_trunc component={0}.cm_trunc input={0}.lstm_nonlin".format(name))
        configs.append("dim-range-node name={0}.c_trunc input-node={0}.cm_trunc dim-offset=0 "
                       "dim={1}".format(name, cell_dim))
        configs.append("dim-range-node name={0}.m_trunc input-node={0}.cm_trunc dim-offset={1} "
                       "dim={1}".format(name, cell_dim))
        configs.append("component-node name={0}.m_batchnorm component={0}.m_batchnorm "
                       "input={0}.m".format(name))
        configs.append("### End LTSM layer '{0}'".format(name))
        return configs




# This class is for lines like
#   'fast-lstmp-layer name=lstm1 input=[-1] delay=-3'
# or:
#   'fast-lstmp-layer name=lstm1 input=[-1] delay=-3 cell-dim=1024 recurrent-projection-dim=512 non-recurrent-projection-dim=512'
# (you can also use the name 'fast-lstmp-batchnorm-layer' if you want it to be followed
# by batchnorm).
# It generates an LSTM sub-graph with output projections (i.e. a projected LSTM, AKA LSTMP).
# Unlike 'lstmp-layer', the core nonlinearities of the LSTM are done in a special-purpose
# component (LstmNonlinearityComponent), and most of the affine parts of the LSTM are combined
# into one.
#
# The output dimension of the layer may be specified via 'cell-dim=xxx', but if not specified,
# the dimension defaults to the same as the input.
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   cell-dim=-1              [Dimension of the cell]
#   recurrent-projection_dim [Dimension of the projection used in recurrent connections, e.g. cell-dim/4]
#   non-recurrent-projection-dim   [Dimension of the projection in non-recurrent connections,
#                                   in addition to recurrent-projection-dim, e.g. cell-dim/4]
#   delay=-1                 [Delay in the recurrent connections of the LSTM ]
#   clipping-threshold=30    [nnet3 LSTMs use a gradient clipping component at the recurrent connections.
#                             This is the threshold used to decide if clipping has to be activated ]
#   zeroing-interval=20      [interval at which we (possibly) zero out the recurrent derivatives.]
#   zeroing-threshold=15     [We only zero out the derivs every zeroing-interval, if derivs exceed this value.]
#   lstm-nonlinearity-options=' max-change=0.75 '  [Options string to pass into the LSTM nonlinearity component.]
#   ng-affine-options=' max-change=1.5 '           [Additional options used for the full matrices in the LSTM, can be used to
#                                      do things like set biases to initialize to 1]
#   decay-time=-1            [If >0, an approximate maximum on how many frames
#                            can be remembered via summation into the cell
#                            contents c_t; enforced by putting a scaling factor
#                            of recurrence_scale = 1 - abs(delay)/decay_time on
#                            the recurrence, i.e. the term c_{t-1} in the LSTM
#                            equations.  E.g. setting this to 20 means no more
#                            than about 20 frames' worth of history,
#                            i.e. history since about t = t-20, can be
#                            accumulated in c_t.]
#  l2-regularize=0.0         Constant controlling l2 regularization for this layer
class XconfigFastLstmpLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token in ['fast-lstmp-layer', 'fast-lstmp-batchnorm-layer']
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input':'[-1]',
                        'cell-dim' : -1, # this is a compulsory argument
                        'recurrent-projection-dim' : -1,
                        'non-recurrent-projection-dim' : -1,
                        'clipping-threshold' : 30.0,
                        'delay' : -1,
                        # if you want to set 'self-repair-scale' (c.f. the
                        # self-repair-scale-nonlinearity config value in older LSTM layers), you can
                        # add 'self-repair-scale=xxx' to
                        # lstm-nonlinearity-options.
                        'lstm-nonlinearity-options' : ' max-change=0.75',
                        # the affine layer contains 4 of our old layers -> use a
                        # larger max-change than the normal value of 0.75.
                        'ng-affine-options' : ' max-change=1.5',
                        'l2-regularize': 0.0,
                        'decay-time':  -1.0,
                        'zeroing-interval' : 20,
                        'zeroing-threshold' : 15.0,
                        'dropout-proportion' : -1.0, # If -1.0, no dropout will
                                                     # be used)
                         }

    def set_derived_configs(self):
        if self.config['recurrent-projection-dim'] <= 0:
            self.config['recurrent-projection-dim'] = self.config['cell-dim'] / 4

        if self.config['non-recurrent-projection-dim'] <= 0:
            self.config['non-recurrent-projection-dim'] = \
               self.config['recurrent-projection-dim']


    def check_configs(self):
        for key in ['cell-dim', 'recurrent-projection-dim',
                    'non-recurrent-projection-dim']:
            if self.config[key] <= 0:
                raise RuntimeError("{0} has invalid value {1}.".format(
                    key, self.config[key]))
        if self.config['delay'] == 0:
            raise RuntimeError("delay cannot be zero")
        if (self.config['recurrent-projection-dim'] +
            self.config['non-recurrent-projection-dim'] >
            self.config['cell-dim']):
            raise RuntimeError("recurrent+non-recurrent projection dim exceeds "
                                "cell dim")
        if ((self.config['dropout-proportion'] > 1.0 or
             self.config['dropout-proportion'] < 0.0) and
             self.config['dropout-proportion'] != -1.0 ):
            raise RuntimeError("dropout-proportion has invalid value {0}.".format(self.config['dropout-proportion']))


    def auxiliary_outputs(self):
        return ['c_t']

    def output_name(self, auxiliary_output = None):
        node_name = ('rp_batchnorm' if self.layer_type == 'fast-lstmp-batchnorm-layer'
                     else 'rp')
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                node_name = auxiliary_output
            else:
                raise RuntimeError("Unknown auxiliary output name {0}".format(auxiliary_output))

        return '{0}.{1}'.format(self.name, node_name)

    def output_dim(self, auxiliary_output = None):
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                if node_name == 'c':
                    return self.config['cell-dim']
                # add code for other auxiliary_outputs here when we decide to expose them
            else:
                raise RuntimeError("Unknown auxiliary output name {0}".format(auxiliary_output))
        return self.config['recurrent-projection-dim'] + \
               self.config['non-recurrent-projection-dim']

    def get_full_config(self):
        ans = []
        config_lines = self._generate_lstm_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in LSTM initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # convenience function to generate the LSTM config
    def _generate_lstm_config(self):
        # assign some variables to reduce verbosity
        name = self.name
        # in the below code we will just call descriptor_strings as descriptors for conciseness
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        cell_dim = self.config['cell-dim']
        delay = self.config['delay']
        rec_proj_dim = self.config['recurrent-projection-dim']
        nonrec_proj_dim = self.config['non-recurrent-projection-dim']
        affine_str = self.config['ng-affine-options']
        decay_time = self.config['decay-time']
        # we expect decay_time to be either -1, or large, like 10 or 50.
        recurrence_scale = (1.0 if decay_time < 0 else
                            1.0 - (abs(delay) / decay_time))
        assert recurrence_scale > 0   # or user may have set decay-time much
                                      # too small.
        bptrunc_str = ("clipping-threshold={0}"
                      " zeroing-threshold={1}"
                      " zeroing-interval={2}"
                      " recurrence-interval={3}"
                      " scale={4}"
                      "".format(self.config['clipping-threshold'],
                                self.config['zeroing-threshold'],
                                self.config['zeroing-interval'],
                                abs(delay), recurrence_scale))

        lstm_str = self.config['lstm-nonlinearity-options']
        dropout_proportion = self.config['dropout-proportion']
        l2_regularize = self.config['l2-regularize']
        l2_regularize_option = ('l2-regularize={0} '.format(l2_regularize)
                                if l2_regularize != 0.0 else '')

        configs = []

        # the equations implemented here are from Sak et. al. "Long Short-Term
        # Memory Recurrent Neural Network Architectures for Large Scale Acoustic
        # Modeling"
        # https://arxiv.org/pdf/1402.1128.pdf
        # See equations (7) to (14).
        # naming convention
        # <layer-name>.W_<outputname>.<input_name> e.g. Lstm1.W_i.xr for matrix providing output to gate i and operating on an appended vector [x,r]
        configs.append("##  Begin LTSM layer '{0}'".format(name))
        configs.append("# Gate control: contains W_i, W_f, W_c and W_o matrices as blocks.")
        configs.append("component name={0}.W_all type=NaturalGradientAffineComponent input-dim={1} "
                       "output-dim={2} {3} {4}".format(
                           name, input_dim + rec_proj_dim, cell_dim * 4,
                           affine_str, l2_regularize_option))
        configs.append("# The core LSTM nonlinearity, implemented as a single component.")
        configs.append("# Input = (i_part, f_part, c_part, o_part, c_{t-1}), output = (c_t, m_t)")
        configs.append("# See cu-math.h:ComputeLstmNonlinearity() for details.")
        configs.append("component name={0}.lstm_nonlin type=LstmNonlinearityComponent cell-dim={1} "
                       "use-dropout={2} {3} {4}"
                       .format(name, cell_dim,
                               "true" if dropout_proportion != -1.0 else "false",
                               lstm_str, l2_regularize_option))
        configs.append("# Component for backprop truncation, to avoid gradient blowup in long training examples.")
        configs.append("component name={0}.cr_trunc type=BackpropTruncationComponent "
                       "dim={1} {2}".format(name, cell_dim + rec_proj_dim, bptrunc_str))
        if dropout_proportion != -1.0:
            configs.append("component name={0}.dropout_mask type=DropoutMaskComponent output-dim=3 "
                           "dropout-proportion={1} "
                           .format(name, dropout_proportion))
        configs.append("# Component specific to 'projected' LSTM (LSTMP), contains both recurrent");
        configs.append("# and non-recurrent projections")
        configs.append("component name={0}.W_rp type=NaturalGradientAffineComponent "
                       "input-dim={1} output-dim={2} {3} {4}".format(
                           name, cell_dim, rec_proj_dim + nonrec_proj_dim,
                           affine_str, l2_regularize_option))
        configs.append("###  Nodes for the components above.")
        configs.append("component-node name={0}.W_all component={0}.W_all input=Append({1}, "
                       "IfDefined(Offset({0}.r_trunc, {2})))".format(name, input_descriptor, delay))

        if dropout_proportion != -1.0:
            # note: the 'input' is a don't-care as the component never uses it; it's required
            # in component-node lines.
            configs.append("component-node name={0}.dropout_mask component={0}.dropout_mask "
                           "input={0}.dropout_mask".format(name))
            configs.append("component-node name={0}.lstm_nonlin component={0}.lstm_nonlin "
                           "input=Append({0}.W_all, IfDefined(Offset({0}.c_trunc, {1})), "
                           "{0}.dropout_mask)".format(name, delay))
        else:
            configs.append("component-node name={0}.lstm_nonlin component={0}.lstm_nonlin "
                           "input=Append({0}.W_all, IfDefined(Offset({0}.c_trunc, {1})))".format(
                               name, delay))
        configs.append("dim-range-node name={0}.c input-node={0}.lstm_nonlin "
                       "dim-offset=0 dim={1}".format(name, cell_dim))
        configs.append("dim-range-node name={0}.m input-node={0}.lstm_nonlin "
                       "dim-offset={1} dim={1}".format(name, cell_dim))
        configs.append("# {0}.rp is the output node of this layer (if we're not "
                       "including batchnorm)".format(name))
        configs.append("component-node name={0}.rp component={0}.W_rp input={0}.m".format(name))
        configs.append("dim-range-node name={0}.r input-node={0}.rp dim-offset=0 "
                       "dim={1}".format(name, rec_proj_dim))
        configs.append("# Note: it's not 100% efficient that we have to stitch the c")
        configs.append("# and r back together to truncate them but it probably");
        configs.append("# makes the deriv truncation more accurate .")
        configs.append("component-node name={0}.cr_trunc component={0}.cr_trunc "
                       "input=Append({0}.c, {0}.r)".format(name))
        configs.append("dim-range-node name={0}.c_trunc input-node={0}.cr_trunc "
                       "dim-offset=0 dim={1}".format(name, cell_dim))
        configs.append("dim-range-node name={0}.r_trunc input-node={0}.cr_trunc "
                       "dim-offset={1} dim={2}".format(name, cell_dim, rec_proj_dim))
        if self.layer_type == "fast-lstmp-batchnorm-layer":
            # Add the batchnorm component, if requested to include batchnorm.
            configs.append("component name={0}.rp_batchnorm type=BatchNormComponent dim={1} ".format(
                name, rec_proj_dim + nonrec_proj_dim))
            configs.append("component-node name={0}.rp_batchnorm component={0}.rp_batchnorm "
                           "input={0}.rp".format(name))
        configs.append("### End LSTM Layer '{0}'".format(name))

        return configs
