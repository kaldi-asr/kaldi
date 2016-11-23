# Copyright 2016    Johns Hopkins University (Dan Povey)
#           2016    Vijayaditya Peddinti
#           2016    Yiming Wang
# Apache 2.0.


""" This module has the implementations of different LSTM layers.
"""
import re

from libs.nnet3.xconfig.basic_layers import XconfigLayerBase
from libs.nnet3.xconfig.utils import XconfigParserError as xparser_error


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
#   clipping-threshold=30    [nnet3 LSTMs use a gradient clipping component at the recurrent connections. This is the threshold used to decide if clipping has to be activated ]
#   norm-based-clipping=True [specifies if the gradient clipping has to activated based on total norm or based on per-element magnitude]
#   self_repair_scale_nonlinearity=1e-5      [It is a constant scaling the self-repair vector computed in derived classes of NonlinearComponent]
#                                       i.e.,  SigmoidComponent, TanhComponent and RectifiedLinearComponent ]
#   ng-per-element-scale-options=''     [Additional options used for the diagonal matrices in the LSTM ]
#   ng-affine-options=''                [Additional options used for the full matrices in the LSTM, can be used to do things like set biases to initialize to 1]
class XconfigLstmLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "lstm-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input':'[-1]',
                        'cell-dim' : -1, # this is a compulsory argument
                        'clipping-threshold' : 30.0,
                        'norm-based-clipping' : True,
                        'delay' : -1,
                        'ng-per-element-scale-options' : ' max-change=0.75',
                        'ng-affine-options' : ' max-change=0.75 ',
                        'self-repair-scale-nonlinearity' : 0.00001,
                        'zeroing-interval' : 20,
                        'zeroing-threshold' : 3.0
                        }

    def set_derived_configs(self):
        if self.config['cell-dim'] <= 0:
            self.config['cell-dim'] = self.InputDim()

    def check_configs(self):
        key = 'cell-dim'
        if self.config['cell-dim'] <= 0:
            raise xparser_error("cell-dim has invalid value {0}.".format(self.config[key]), self.str())

        for key in ['self-repair-scale-nonlinearity']:
            if self.config[key] < 0.0 or self.config[key] > 1.0:
                raise xparser_error("{0} has invalid value {1}.".format(key, self.config[key]))

    def auxiliary_outputs(self):
        return ['c_t']

    def output_name(self, auxiliary_output = None):
        node_name = 'm_t'
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                node_name = auxiliary_output
            else:
                raise xparser_error("Unknown auxiliary output name {0}".format(auxiliary_output), self.str())

        return '{0}.{1}'.format(self.name, node_name)

    def output_dim(self, auxiliary_output = None):
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                if node_name == 'c_t':
                    return self.config['cell-dim']
                # add code for other auxiliary_outputs here when we decide to expose them
            else:
                raise xparser_error("Unknown auxiliary output name {0}".format(auxiliary_output), self.str())

        return self.config['cell-dim']

    def get_full_config(self):
        ans = []
        config_lines = self.generate_lstm_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in LSTM initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # convenience function to generate the LSTM config
    def generate_lstm_config(self):

        # assign some variables to reduce verbosity
        name = self.name
        # in the below code we will just call descriptor_strings as descriptors for conciseness
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        cell_dim = self.config['cell-dim']
        delay = self.config['delay']

        repair_nonlin = self.config['self-repair-scale-nonlinearity']
        repair_nonlin_str = "self-repair-scale={0:.10f}".format(repair_nonlin) if repair_nonlin is not None else ''
        bptrunc_str = ("clipping-threshold={0}"
                      " zeroing-threshold={1}"
                      " zeroing-interval={2}"
                      " recurrence-interval={3}"
                      "".format(self.config['clipping-threshold'],
                                self.config['zeroing-threshold'],
                                self.config['zeroing-interval'],
                                abs(delay)))
        affine_str = self.config['ng-affine-options']
        # Natural gradient per element scale parameters
        # TODO: decide if we want to keep exposing these options
        if re.search('param-mean', ng_per_element_scale_options) is None and \
           re.search('param-stddev', ng_per_element_scale_options) is None:
           ng_per_element_scale_options += " param-mean=0.0 param-stddev=1.0 "
        pes_str = ng_per_element_scale_options



        configs = []

        # the equations implemented here are
        # TODO: write these
        # naming convention
        # <layer-name>.W_<outputname>.<input_name> e.g. Lstm1.W_i.xr for matrix providing output to gate i and operating on an appended vector [x,r]
        configs.append("# Input gate control : W_i* matrices")
        configs.append("component name={0}.W_i.xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + cell_dim, cell_dim, affine_str))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_i.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))

        configs.append("# Forget gate control : W_f* matrices")
        configs.append("component name={0}.W_f.xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + cell_dim, cell_dim, affine_str))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_f.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))

        configs.append("#  Output gate control : W_o* matrices")
        configs.append("component name={0}.W_o.xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + cell_dim, cell_dim, affine_str))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_o.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))

        configs.append("# Cell input matrices : W_c* matrices")
        configs.append("component name={0}.W_c.xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + cell_dim, cell_dim, affine_str))


        configs.append("# Defining the non-linearities")
        configs.append("component name={0}.i type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.f type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.o type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.g type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.h type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))

        configs.append("# Defining the components for other cell computations")
        configs.append("component name={0}.c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.m type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.c type=BackpropTruncationComponent dim={1} {2}".format(name, cell_dim, bptrunc_str))

        # c1_t and c2_t defined below
        configs.append("component-node name={0}.c_t component={0}.c input=Sum({0}.c1_t, {0}.c2_t)".format(name))
        delayed_c_t_descriptor = "IfDefined(Offset({0}.c_t, {1}))".format(name, delay)

        configs.append("# i_t")
        configs.append("component-node name={0}.i1_t component={0}.W_i.xr input=Append({1}, IfDefined(Offset({0}.r_t, {2})))".format(name, input_descriptor, delay))
        configs.append("component-node name={0}.i2_t component={0}.w_i.c  input={1}".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.i_t component={0}.i input=Sum({0}.i1_t, {0}.i2_t)".format(name))

        configs.append("# f_t")
        configs.append("component-node name={0}.f1_t component={0}.W_f.xr input=Append({1}, IfDefined(Offset({0}.r_t, {2})))".format(name, input_descriptor, delay))
        configs.append("component-node name={0}.f2_t component={0}.w_f.c  input={1}".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.f_t component={0}.f input=Sum({0}.f1_t, {0}.f2_t)".format(name))

        configs.append("# o_t")
        configs.append("component-node name={0}.o1_t component={0}.W_o.xr input=Append({1}, IfDefined(Offset({0}.r_t, {2})))".format(name, input_descriptor, delay))
        configs.append("component-node name={0}.o2_t component={0}.w_o.c input={0}.c_t".format(name))
        configs.append("component-node name={0}.o_t component={0}.o input=Sum({0}.o1_t, {0}.o2_t)".format(name))

        configs.append("# h_t")
        configs.append("component-node name={0}.h_t component={0}.h input={0}.c_t".format(name))

        configs.append("# g_t")
        configs.append("component-node name={0}.g1_t component={0}.W_c.xr input=Append({1}, IfDefined(Offset({0}.r_t, {2})))".format(name, input_descriptor, delay))
        configs.append("component-node name={0}.g_t component={0}.g input={0}.g1_t".format(name))

        configs.append("# parts of c_t")
        configs.append("component-node name={0}.c1_t component={0}.c1  input=Append({0}.f_t, {1})".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.c2_t component={0}.c2 input=Append({0}.i_t, {0}.g_t)".format(name))

        configs.append("# m_t")
        configs.append("component-node name={0}.m_t component={0}.m input=Append({0}.o_t, {0}.h_t)".format(name))

        # add the recurrent connections
        configs.append("component name={0}.r type=BackpropTruncationComponent dim={1} {2}".format(name, cell_dim, bptrunc_str))
        configs.append("component-node name={0}.r_t component={0}.r input={0}.m_t".format(name))

        return configs


# This class is for lines like
#   'lstmp-layer name=lstm1 input=[-1] delay=-3'
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
#   recurrent_projection_dim [Dimension of the projection used in recurrent connections]
#   non_recurrent_projection_dim        [Dimension of the projection in non-recurrent connections]
#   delay=-1                 [Delay in the recurrent connections of the LSTM ]
#   clipping-threshold=30    [nnet3 LSTMs use a gradient clipping component at the recurrent connections. This is the threshold used to decide if clipping has to be activated ]
#   norm-based-clipping=True [specifies if the gradient clipping has to activated based on total norm or based on per-element magnitude]
#   self_repair_scale_nonlinearity=1e-5      [It is a constant scaling the self-repair vector computed in derived classes of NonlinearComponent]
#                                       i.e.,  SigmoidComponent, TanhComponent and RectifiedLinearComponent ]
#   ng-per-element-scale-options=''   [Additional options used for the diagonal matrices in the LSTM ]
#   ng-affine-options=''              [Additional options used for the full matrices in the LSTM, can be used to do things like set biases to initialize to 1]
class XconfigLstmpLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        print first_token
        assert first_token == "lstmp-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input' : '[-1]',
                        'cell-dim' : -1, # this is a compulsory argument
                        'recurrent-projection-dim' : -1,
                        'non-recurrent-projection-dim' : -1,
                        'clipping-threshold' : 30.0,
                        'norm-based-clipping' : True,
                        'delay' : -1,
                        'ng-per-element-scale-options' : ' max-change=0.75 ',
                        'ng-affine-options' : ' max-change=0.75 ',
                        'self-repair-scale-nonlinearity' : 0.00001,
                        'zeroing-interval' : 20,
                        'zeroing-threshold' : 3.0
                       }

    def set_derived_configs(self):
        if self.config['cell-dim'] <= 0:
            self.config['cell-dim'] = self.InputDim()

        for key in ['recurrent-projection-dim', 'non-recurrent-projection-dim']:
            if self.config[key] <= 0:
                self.config[key] = self.config['cell-dim'] / 2

    def check_configs(self):
        for key in ['cell-dim', 'recurrent-projection-dim', 'non-recurrent-projection-dim']:
            if self.config[key] <= 0:
                raise xparser_error("{0} has invalid value {1}.".format(key, self.config[key]), self.str())

        for key in ['self-repair-scale-nonlinearity']:
            if self.config[key] < 0.0 or self.config[key] > 1.0:
                raise xparser_error("{0} has invalid value {2}.".format(self.layer_type,
                                                                               key,
                                                                               self.config[key]))
    def auxiliary_outputs(self):
        return ['c_t']

    def output_name(self, auxiliary_output = None):
        node_name = 'rp_t'
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
        config_lines = self.generate_lstm_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in LSTM initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # convenience function to generate the LSTM config
    def generate_lstm_config(self):

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
        bptrunc_str = ("clipping-threshold={0}"
                      " zeroing-threshold={1}"
                      " zeroing-interval={2}"
                      " recurrence-interval={3}"
                      "".format(self.config['clipping-threshold'],
                                self.config['zeroing-threshold'],
                                self.config['zeroing-interval'],
                                abs(delay)))
        affine_str = self.config['ng-affine-options']
        pes_str = self.config['ng-per-element-scale-options']

        # Natural gradient per element scale parameters
        # TODO: decide if we want to keep exposing these options
        if re.search('param-mean', pes_str) is None and \
           re.search('param-stddev', pes_str) is None:
           pes_str += " param-mean=0.0 param-stddev=1.0 "

        configs = []
        # the equations implemented here are from Sak et. al. "Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling"
        # http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43905.pdf
        # naming convention
        # <layer-name>.W_<outputname>.<input_name> e.g. Lstm1.W_i.xr for matrix providing output to gate i and operating on an appended vector [x,r]
        configs.append("# Input gate control : W_i* matrices")
        configs.append("component name={0}.W_i.xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, cell_dim, affine_str))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_i.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))

        configs.append("# Forget gate control : W_f* matrices")
        configs.append("component name={0}.W_f.xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, cell_dim, affine_str))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_f.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))

        configs.append("#  Output gate control : W_o* matrices")
        configs.append("component name={0}.W_o.xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, cell_dim, affine_str))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_o.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))

        configs.append("# Cell input matrices : W_c* matrices")
        configs.append("component name={0}.W_c.xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, cell_dim, affine_str))

        configs.append("# Defining the non-linearities")
        configs.append("component name={0}.i type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.f type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.o type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.g type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.h type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))

        configs.append("# Defining the components for other cell computations")
        configs.append("component name={0}.c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.m type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.c type=BackpropTruncationComponent dim={1} {2}".format(name, cell_dim, bptrunc_str))

        # c1_t and c2_t defined below
        configs.append("component-node name={0}.c_t component={0}.c input=Sum({0}.c1_t, {0}.c2_t)".format(name))
        delayed_c_t_descriptor = "IfDefined(Offset({0}.c_t, {1}))".format(name, delay)

        recurrent_connection = '{0}.r_t'.format(name)
        configs.append("# i_t")
        configs.append("component-node name={0}.i1_t component={0}.W_i.xr input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.i2_t component={0}.w_i.c  input={1}".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.i_t component={0}.i input=Sum({0}.i1_t, {0}.i2_t)".format(name))

        configs.append("# f_t")
        configs.append("component-node name={0}.f1_t component={0}.W_f.xr input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.f2_t component={0}.w_f.c  input={1}".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.f_t component={0}.f input=Sum({0}.f1_t, {0}.f2_t)".format(name))

        configs.append("# o_t")
        configs.append("component-node name={0}.o1_t component={0}.W_o.xr input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.o2_t component={0}.w_o.c input={0}.c_t".format(name))
        configs.append("component-node name={0}.o_t component={0}.o input=Sum({0}.o1_t, {0}.o2_t)".format(name))

        configs.append("# h_t")
        configs.append("component-node name={0}.h_t component={0}.h input={0}.c_t".format(name))

        configs.append("# g_t")
        configs.append("component-node name={0}.g1_t component={0}.W_c.xr input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.g_t component={0}.g input={0}.g1_t".format(name))

        configs.append("# parts of c_t")
        configs.append("component-node name={0}.c1_t component={0}.c1  input=Append({0}.f_t, {1})".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.c2_t component={0}.c2 input=Append({0}.i_t, {0}.g_t)".format(name))

        configs.append("# m_t")
        configs.append("component-node name={0}.m_t component={0}.m input=Append({0}.o_t, {0}.h_t)".format(name))

        # add the recurrent connections
        configs.append("# projection matrices : Wrm and Wpm")
        configs.append("component name={0}.W_rp.m type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, rec_proj_dim + nonrec_proj_dim, affine_str))
        configs.append("component name={0}.r type=BackpropTruncationComponent dim={1} {2}".format(name, rec_proj_dim, bptrunc_str))

        configs.append("# r_t and p_t : rp_t will be the output")
        configs.append("component-node name={0}.rp_t component={0}.W_rp.m input={0}.m_t".format(name))
        configs.append("dim-range-node name={0}.r_t_preclip input-node={0}.rp_t dim-offset=0 dim={1}".format(name, rec_proj_dim))
        configs.append("component-node name={0}.r_t component={0}.r input={0}.r_t_preclip".format(name))

        return configs

# Same as the LSTMP layer except that the matrix multiplications are combined
# we probably keep only version after experimentation. One year old experiments
# show that this version is slightly worse and might require some tuning
class XconfigLstmpcLayer(XconfigLstmpLayer):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "lstmpc-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    # convenience function to generate the LSTM config
    def generate_lstm_config(self):
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
        bptrunc_str = ("clipping-threshold={0}"
                      " zeroing-threshold={1}"
                      " zeroing-interval={2}"
                      " recurrence-interval={3}"
                      "".format(self.config['clipping-threshold'],
                                self.config['zeroing-threshold'],
                                self.config['zeroing-interval'],
                                abs(delay)))
        affine_str = self.config['ng-affine-options']
        # Natural gradient per element scale parameters
        # TODO: decide if we want to keep exposing these options
        if re.search('param-mean', ng_per_element_scale_options) is None and \
           re.search('param-stddev', ng_per_element_scale_options) is None:
           ng_per_element_scale_options += " param-mean=0.0 param-stddev=1.0 "
        pes_str = ng_per_element_scale_options

        configs = []
        # naming convention
        # <layer-name>.W_<outputname>.<input_name> e.g. Lstm1.W_i.xr for matrix providing output to gate i and operating on an appended vector [x,r]
        configs.append("# Full W_ifoc* matrix")
        configs.append("component name={0}.W_ifoc.xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, 4*cell_dim, affine_str))
        configs.append("# note : the cell outputs pass through a diagonal matrix")

        # we will not combine the diagonal matrix operations as one of these has a different delay
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_i.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))
        configs.append("component name={0}.w_f.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))
        configs.append("component name={0}.w_o.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))

        configs.append("# Defining the non-linearities")
        configs.append("component name={0}.i type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.f type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.o type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.g type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.h type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))

        configs.append("# Defining the components for other cell computations")
        configs.append("component name={0}.c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.m type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.c type=BackpropTruncationComponent dim={1} {2}".format(name, cell_dim, bptrunc_str))

        # c1_t and c2_t defined below
        configs.append("component-node name={0}.c_t component={0}.c input=Sum({0}.c1_t, {0}.c2_t)".format(name))
        delayed_c_t_descriptor = "IfDefined(Offset({0}.c_t, {1}))".format(name, delay)
        rec_connection = '{0}.rp_t'.format(name)

        component_nodes.append("component-node name={0}.ifoc_t component={0}.W_ifoc.xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))


        offset = 0
        component_nodes.append("# i_t")
        component_nodes.append("dim-range-node name={0}.i1_t input-node={0}.ifoc_t dim-offset={1} dim={2}".format(name, offset, cell_dim))
        offset += cell_dim
        component_nodes.append("component-node name={0}.i2_t component={0}.w_i.cinput={1}".format(name, delayed_c_t_descriptor))
        component_nodes.append("component-node name={0}.i_t component={0}.i input=Sum({0}.i1_t, {0}.i2_t)".format(name))

        component_nodes.append("# f_t")
        component_nodes.append("dim-range-node name={0}.f1_t input-node={0}.ifoc_t dim-offset={1} dim={2}".format(name, offset, cell_dim))
        offset += cell_dim
        component_nodes.append("component-node name={0}.f2_t component={0}.w_f.c  input={1}".format(name, delayed_c_t_descriptor))
        component_nodes.append("component-node name={0}.f_t component={0}.f input=Sum({0}.f1_t, {0}.f2_t)".format(name))

        component_nodes.append("# o_t")
        component_nodes.append("dim-range-node name={0}.o1_t input-node={0}.ifoc_t dim-offset={1} dim={2}".format(name, offset, cell_dim))
        offset += cell_dim
        component_nodes.append("component-node name={0}.o2_t component={0}.w_o.c input={0}.c_t".format(name))
        component_nodes.append("component-node name={0}.o_t component={0}.o input=Sum({0}.o1_t, {0}.o2_t)".format(name))

        component_nodes.append("# h_t")
        component_nodes.append("component-node name={0}.h_t component={0}.h input={0}.c_t".format(name))

        component_nodes.append("# g_t")
        component_nodes.append("dim-range-node name={0}.g1_t input-node={0}.ifoc_t dim-offset={1} dim={2}".format(name, offset, cell_dim))
        offset += cell_dim
        component_nodes.append("component-node name={0}.g_t component={0}.g input={0}.g1_t".format(name))


        configs.append("# parts of c_t")
        configs.append("component-node name={0}.c1_t component={0}.c1  input=Append({0}.f_t, {1})".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.c2_t component={0}.c2 input=Append({0}.i_t, {0}.g_t)".format(name))

        configs.append("# m_t")
        configs.append("component-node name={0}.m_t component={0}.m input=Append({0}.o_t, {0}.h_t)".format(name))

        # add the recurrent connections
        configs.append("# projection matrices : Wrm and Wpm")
        configs.append("component name={0}.W_rp.m type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim + non_recurrent_projection_dim, affine_str))
        configs.append("component name={0}.r type=BackpropTruncationComponent dim={1} {2}".format(name, recurrent_projection_dim, bptrunc_str))

        configs.append("# r_t and p_t : rp_t will be the output")
        configs.append("component-node name={0}.rp_t component={0}.W_rp.m input={0}.m_t".format(name))
        configs.append("dim-range-node name={0}.r_t_preclip input-node={0}.rp_t dim-offset=0 dim={1}".format(name, recurrent_projection_dim))
        configs.append("component-node name={0}.r_t component={0}.r input={0}.r_t_preclip".format(name))

        return configs
