#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
import sys
import warnings
import copy
from operator import itemgetter

def GetSumDescriptor(inputs):
    sum_descriptors = inputs
    while len(sum_descriptors) != 1:
        cur_sum_descriptors = []
        pair = []
        while len(sum_descriptors) > 0:
            value = sum_descriptors.pop()
            if value.strip() != '':
                pair.append(value)
            if len(pair) == 2:
                cur_sum_descriptors.append("Sum({0}, {1})".format(pair[0], pair[1]))
                pair = []
        if pair:
            cur_sum_descriptors.append(pair[0])
        sum_descriptors = cur_sum_descriptors
    return sum_descriptors

# adds the input nodes and returns the descriptor
def AddInputLayer(config_lines, feat_dim, splice_indexes=[0], ivector_dim=0):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    output_dim = 0
    components.append('input-node name=input dim=' + str(feat_dim))
    list = [('Offset(input, {0})'.format(n) if n != 0 else 'input') for n in splice_indexes]
    output_dim += len(splice_indexes) * feat_dim
    if ivector_dim > 0:
        components.append('input-node name=ivector dim=' + str(ivector_dim))
        list.append('ReplaceIndex(ivector, t, 0)')
        output_dim += ivector_dim
    if len(list) > 1:
        splice_descriptor = "Append({0})".format(", ".join(list))
    else:
        splice_descriptor = list[0]
    print(splice_descriptor)
    return {'descriptor': splice_descriptor,
            'dimension': output_dim}

def AddNoOpLayer(config_lines, name, input):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append('component name={0}_noop type=NoOpComponent dim={1}'.format(name, input['dimension']))
    component_nodes.append('component-node name={0}_noop component={0}_noop input={1}'.format(name, input['descriptor']))

    return {'descriptor':  '{0}_noop'.format(name),
            'dimension': input['dimension']}

def AddLdaLayer(config_lines, name, input, lda_file):
    return AddFixedAffineLayer(config_lines, name, input, lda_file)

def AddFixedAffineLayer(config_lines, name, input, matrix_file):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append('component name={0}_fixaffine type=FixedAffineComponent matrix={1}'.format(name, matrix_file))
    component_nodes.append('component-node name={0}_fixaffine component={0}_fixaffine input={1}'.format(name, input['descriptor']))

    return {'descriptor':  '{0}_fixaffine'.format(name),
            'dimension': input['dimension']}


def AddBlockAffineLayer(config_lines, name, input, output_dim, num_blocks):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    assert((input['dimension'] % num_blocks == 0) and
            (output_dim % num_blocks == 0))
    components.append('component name={0}_block_affine type=BlockAffineComponent input-dim={1} output-dim={2} num-blocks={3}'.format(name, input['dimension'], output_dim, num_blocks))
    component_nodes.append('component-node name={0}_block_affine component={0}_block_affine input={1}'.format(name, input['descriptor']))

    return {'descriptor' : '{0}_block_affine'.format(name),
                           'dimension' : output_dim}

def AddPermuteLayer(config_lines, name, input, column_map):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    permute_indexes = ",".join(map(lambda x: str(x), column_map))
    components.append('component name={0}_permute type=PermuteComponent column-map={1}'.format(name, permute_indexes))
    component_nodes.append('component-node name={0}_permute component={0}_permute input={1}'.format(name, input['descriptor']))

    return {'descriptor': '{0}_permute'.format(name),
            'dimension': input['dimension']}

def AddAffineLayer(config_lines, name, input, output_dim, ng_affine_options = "", max_change_per_component = 0.75):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    # Per-component max-change option
    max_change_options = "max-change={0:.2f}".format(max_change_per_component) if max_change_per_component is not None else ''

    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3} {4}".format(name, input['dimension'], output_dim, ng_affine_options, max_change_options))
    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))

    return {'descriptor':  '{0}_affine'.format(name),
            'dimension': output_dim}

def AddAffRelNormLayer(config_lines, name, input, output_dim, ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0, self_repair_scale = None, max_change_per_component = 0.75):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    # self_repair_scale is a constant scaling the self-repair vector computed in RectifiedLinearComponent
    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    # Per-component max-change option
    max_change_options = "max-change={0:.2f}".format(max_change_per_component) if max_change_per_component is not None else ''

    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3} {4}".format(name, input['dimension'], output_dim, ng_affine_options, max_change_options))
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1} {2}".format(name, output_dim, self_repair_string))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, output_dim, norm_target_rms))

    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_relu component={0}_relu input={0}_affine".format(name))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_relu".format(name))

    return {'descriptor':  '{0}_renorm'.format(name),
            'dimension': output_dim}

def AddAffPnormLayer(config_lines, name, input, pnorm_input_dim, pnorm_output_dim, ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], pnorm_input_dim, ng_affine_options))
    components.append("component name={0}_pnorm type=PnormComponent input-dim={1} output-dim={2}".format(name, pnorm_input_dim, pnorm_output_dim))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, pnorm_output_dim, norm_target_rms))

    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_pnorm component={0}_pnorm input={0}_affine".format(name))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_pnorm".format(name))

    return {'descriptor':  '{0}_renorm'.format(name),
            'dimension': pnorm_output_dim}

def AddConvolutionLayer(config_lines, name, input,
                       input_x_dim, input_y_dim, input_z_dim,
                       filt_x_dim, filt_y_dim,
                       filt_x_step, filt_y_step,
                       num_filters, input_vectorization,
                       param_stddev = None, bias_stddev = None,
                       filter_bias_file = None,
                       is_updatable = True):
    assert(input['dimension'] == input_x_dim * input_y_dim * input_z_dim)
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    conv_init_string = ("component name={name}_conv type=ConvolutionComponent "
                       "input-x-dim={input_x_dim} input-y-dim={input_y_dim} input-z-dim={input_z_dim} "
                       "filt-x-dim={filt_x_dim} filt-y-dim={filt_y_dim} "
                       "filt-x-step={filt_x_step} filt-y-step={filt_y_step} "
                       "input-vectorization-order={vector_order}".format(name = name,
                       input_x_dim = input_x_dim, input_y_dim = input_y_dim, input_z_dim = input_z_dim,
                       filt_x_dim = filt_x_dim, filt_y_dim = filt_y_dim,
                       filt_x_step = filt_x_step, filt_y_step = filt_y_step,
                       vector_order = input_vectorization))
    if filter_bias_file is not None:
        conv_init_string += " matrix={0}".format(filter_bias_file)
    else:
        conv_init_string += " num-filters={0}".format(num_filters)

    components.append(conv_init_string)
    component_nodes.append("component-node name={0}_conv_t component={0}_conv input={1}".format(name, input['descriptor']))

    num_x_steps = (1 + (input_x_dim - filt_x_dim) / filt_x_step)
    num_y_steps = (1 + (input_y_dim - filt_y_dim) / filt_y_step)
    output_dim = num_x_steps * num_y_steps * num_filters;
    return {'descriptor':  '{0}_conv_t'.format(name),
            'dimension': output_dim,
            '3d-dim': [num_x_steps, num_y_steps, num_filters],
            'vectorization': 'zyx'}

# The Maxpooling component assumes input vectorizations of type zyx
def AddMaxpoolingLayer(config_lines, name, input,
                      input_x_dim, input_y_dim, input_z_dim,
                      pool_x_size, pool_y_size, pool_z_size,
                      pool_x_step, pool_y_step, pool_z_step):
    if input_x_dim < 1 or input_y_dim < 1 or input_z_dim < 1:
        raise Exception("non-positive maxpooling input size ({0}, {1}, {2})".
                 format(input_x_dim, input_y_dim, input_z_dim))
    if pool_x_size > input_x_dim or pool_y_size > input_y_dim or pool_z_size > input_z_dim:
        raise Exception("invalid maxpooling pool size vs. input size")
    if pool_x_step > pool_x_size or pool_y_step > pool_y_size or pool_z_step > pool_z_size:
        raise Exception("invalid maxpooling pool step vs. pool size")

    assert(input['dimension'] == input_x_dim * input_y_dim * input_z_dim)
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append('component name={name}_maxp type=MaxpoolingComponent '
                      'input-x-dim={input_x_dim} input-y-dim={input_y_dim} input-z-dim={input_z_dim} '
                      'pool-x-size={pool_x_size} pool-y-size={pool_y_size} pool-z-size={pool_z_size} '
                      'pool-x-step={pool_x_step} pool-y-step={pool_y_step} pool-z-step={pool_z_step} '.
                      format(name = name,
                      input_x_dim = input_x_dim, input_y_dim = input_y_dim, input_z_dim = input_z_dim,
                      pool_x_size = pool_x_size, pool_y_size = pool_y_size, pool_z_size = pool_z_size,
                      pool_x_step = pool_x_step, pool_y_step = pool_y_step, pool_z_step = pool_z_step))

    component_nodes.append('component-node name={0}_maxp_t component={0}_maxp input={1}'.format(name, input['descriptor']))

    num_pools_x = 1 + (input_x_dim - pool_x_size) / pool_x_step;
    num_pools_y = 1 + (input_y_dim - pool_y_size) / pool_y_step;
    num_pools_z = 1 + (input_z_dim - pool_z_size) / pool_z_step;
    output_dim = num_pools_x * num_pools_y * num_pools_z;

    return {'descriptor':  '{0}_maxp_t'.format(name),
            'dimension': output_dim,
            '3d-dim': [num_pools_x, num_pools_y, num_pools_z],
            'vectorization': 'zyx'}


def AddSoftmaxLayer(config_lines, name, input):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_log_softmax type=LogSoftmaxComponent dim={1}".format(name, input['dimension']))
    component_nodes.append("component-node name={0}_log_softmax component={0}_log_softmax input={1}".format(name, input['descriptor']))

    return {'descriptor':  '{0}_log_softmax'.format(name),
            'dimension': input['dimension']}


def AddSigmoidLayer(config_lines, name, input, self_repair_scale = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    # self_repair_scale is a constant scaling the self-repair vector computed in SigmoidComponent
    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    components.append("component name={0}_sigmoid type=SigmoidComponent dim={1}".format(name, input['dimension'], self_repair_string))
    component_nodes.append("component-node name={0}_sigmoid component={0}_sigmoid input={1}".format(name, input['descriptor']))
    return {'descriptor':  '{0}_sigmoid'.format(name),
            'dimension': input['dimension']}

def AddOutputLayer(config_lines, input, label_delay = None, suffix=None, objective_type = "linear"):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    name = 'output'
    if suffix is not None:
        name = '{0}-{1}'.format(name, suffix)

    if label_delay is None:
        component_nodes.append('output-node name={0} input={1} objective={2}'.format(name, input['descriptor'], objective_type))
    else:
        component_nodes.append('output-node name={0} input=Offset({1},{2}) objective={3}'.format(name, input['descriptor'], label_delay, objective_type))

def AddFinalLayer(config_lines, input, output_dim,
        ng_affine_options = " param-stddev=0 bias-stddev=0 ",
        max_change_per_component = 1.5,
        label_delay=None,
        use_presoftmax_prior_scale = False,
        prior_scale_file = None,
        include_log_softmax = True,
        add_final_sigmoid = False,
        name_affix = None,
        objective_type = "linear"):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    if name_affix is not None:
        final_node_prefix = 'Final-' + str(name_affix)
    else:
        final_node_prefix = 'Final'

    prev_layer_output = AddAffineLayer(config_lines,
            final_node_prefix , input, output_dim,
            ng_affine_options, max_change_per_component)
    if include_log_softmax:
        if use_presoftmax_prior_scale :
            components.append('component name={0}-fixed-scale type=FixedScaleComponent scales={1}'.format(final_node_prefix, prior_scale_file))
            component_nodes.append('component-node name={0}-fixed-scale component={0}-fixed-scale input={1}'.format(final_node_prefix,
                prev_layer_output['descriptor']))
            prev_layer_output['descriptor'] = "{0}-fixed-scale".format(final_node_prefix)
        prev_layer_output = AddSoftmaxLayer(config_lines, final_node_prefix, prev_layer_output)
    elif add_final_sigmoid:
        # Useful when you need the final outputs to be probabilities
        # between 0 and 1.
        # Usually used with an objective-type such as "quadratic"
        prev_layer_output = AddSigmoidLayer(config_lines, final_node_prefix, prev_layer_output)
    # we use the same name_affix as a prefix in for affine/scale nodes but as a
    # suffix for output node
    AddOutputLayer(config_lines, prev_layer_output, label_delay, suffix = name_affix, objective_type = objective_type)

def AddLstmLayer(config_lines,
                 name, input, cell_dim,
                 recurrent_projection_dim = 0,
                 non_recurrent_projection_dim = 0,
                 clipping_threshold = 30.0,
                 zeroing_threshold = 15.0,
                 zeroing_interval = 20,
                 ng_per_element_scale_options = "",
                 ng_affine_options = "",
                 lstm_delay = -1,
                 self_repair_scale_nonlinearity = None,
                 max_change_per_component = 0.75):
    assert(recurrent_projection_dim >= 0 and non_recurrent_projection_dim >= 0)
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    input_descriptor = input['descriptor']
    input_dim = input['dimension']
    name = name.strip()

    if (recurrent_projection_dim == 0):
        add_recurrent_projection = False
        recurrent_projection_dim = cell_dim
        recurrent_connection = "m_t"
    else:
        add_recurrent_projection = True
        recurrent_connection = "r_t"
    if (non_recurrent_projection_dim == 0):
        add_non_recurrent_projection = False
    else:
        add_non_recurrent_projection = True

    # self_repair_scale_nonlinearity is a constant scaling the self-repair vector computed in derived classes of NonlinearComponent,
    # i.e.,  SigmoidComponent, TanhComponent and RectifiedLinearComponent
    self_repair_nonlinearity_string = "self-repair-scale={0:.10f}".format(self_repair_scale_nonlinearity) if self_repair_scale_nonlinearity is not None else ''
    # Natural gradient per element scale parameters
    ng_per_element_scale_options += " param-mean=0.0 param-stddev=1.0 "
    # Per-component max-change option
    max_change_options = "max-change={0:.2f}".format(max_change_per_component) if max_change_per_component is not None else ''
    # Parameter Definitions W*(* replaced by - to have valid names)
    components.append("# Input gate control : W_i* matrices")
    components.append("component name={0}_W_i-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3} {4}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options, max_change_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_ic type=NaturalGradientPerElementScaleComponent  dim={1} {2} {3}".format(name, cell_dim, ng_per_element_scale_options, max_change_options))

    components.append("# Forget gate control : W_f* matrices")
    components.append("component name={0}_W_f-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3} {4}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options, max_change_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_fc type=NaturalGradientPerElementScaleComponent  dim={1} {2} {3}".format(name, cell_dim, ng_per_element_scale_options, max_change_options))

    components.append("#  Output gate control : W_o* matrices")
    components.append("component name={0}_W_o-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3} {4}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options, max_change_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_oc type=NaturalGradientPerElementScaleComponent  dim={1} {2} {3}".format(name, cell_dim, ng_per_element_scale_options, max_change_options))

    components.append("# Cell input matrices : W_c* matrices")
    components.append("component name={0}_W_c-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3} {4}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options, max_change_options))


    components.append("# Defining the non-linearities")
    components.append("component name={0}_i type=SigmoidComponent dim={1} {2}".format(name, cell_dim, self_repair_nonlinearity_string))
    components.append("component name={0}_f type=SigmoidComponent dim={1} {2}".format(name, cell_dim, self_repair_nonlinearity_string))
    components.append("component name={0}_o type=SigmoidComponent dim={1} {2}".format(name, cell_dim, self_repair_nonlinearity_string))
    components.append("component name={0}_g type=TanhComponent dim={1} {2}".format(name, cell_dim, self_repair_nonlinearity_string))
    components.append("component name={0}_h type=TanhComponent dim={1} {2}".format(name, cell_dim, self_repair_nonlinearity_string))

    components.append("# Defining the cell computations")
    components.append("component name={0}_c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_m type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c type=BackpropTruncationComponent dim={1} "
        "clipping-threshold={2} zeroing-threshold={3} zeroing-interval={4} "
        "recurrence-interval={5}".format(name, cell_dim, clipping_threshold, zeroing_threshold,
        zeroing_interval, abs(lstm_delay)))

    # c1_t and c2_t defined below
    component_nodes.append("component-node name={0}_c_t component={0}_c input=Sum({0}_c1_t, {0}_c2_t)".format(name))
    c_tminus1_descriptor = "IfDefined(Offset({0}_c_t, {1}))".format(name, lstm_delay)

    component_nodes.append("# i_t")
    component_nodes.append("component-node name={0}_i1 component={0}_W_i-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_i2 component={0}_w_ic  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_i_t component={0}_i input=Sum({0}_i1, {0}_i2)".format(name))

    component_nodes.append("# f_t")
    component_nodes.append("component-node name={0}_f1 component={0}_W_f-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_f2 component={0}_w_fc  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_f_t component={0}_f input=Sum({0}_f1,{0}_f2)".format(name))

    component_nodes.append("# o_t")
    component_nodes.append("component-node name={0}_o1 component={0}_W_o-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_o2 component={0}_w_oc input={0}_c_t".format(name))
    component_nodes.append("component-node name={0}_o_t component={0}_o input=Sum({0}_o1, {0}_o2)".format(name))

    component_nodes.append("# h_t")
    component_nodes.append("component-node name={0}_h_t component={0}_h input={0}_c_t".format(name))

    component_nodes.append("# g_t")
    component_nodes.append("component-node name={0}_g1 component={0}_W_c-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_g_t component={0}_g input={0}_g1".format(name))

    component_nodes.append("# parts of c_t")
    component_nodes.append("component-node name={0}_c1_t component={0}_c1  input=Append({0}_f_t, {1})".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_c2_t component={0}_c2 input=Append({0}_i_t, {0}_g_t)".format(name))

    component_nodes.append("# m_t")
    component_nodes.append("component-node name={0}_m_t component={0}_m input=Append({0}_o_t, {0}_h_t)".format(name))

    # add the recurrent connections
    if (add_recurrent_projection and add_non_recurrent_projection):
        components.append("# projection matrices : Wrm and Wpm")
        components.append("component name={0}_W-m type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3} {4}".format(name, cell_dim, recurrent_projection_dim + non_recurrent_projection_dim, ng_affine_options, max_change_options))
        components.append("component name={0}_r type=BackpropTruncationComponent dim={1} "
            "clipping-threshold={2} zeroing-threshold={3} zeroing-interval={4} "
            "recurrence-interval={5}".format(name, recurrent_projection_dim, clipping_threshold,
            zeroing_threshold, zeroing_interval, abs(lstm_delay)))
        component_nodes.append("# r_t and p_t")
        component_nodes.append("component-node name={0}_rp_t component={0}_W-m input={0}_m_t".format(name))
        component_nodes.append("dim-range-node name={0}_r_t_preclip input-node={0}_rp_t dim-offset=0 dim={1}".format(name, recurrent_projection_dim))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_rp_t'.format(name)
        output_dim = recurrent_projection_dim + non_recurrent_projection_dim

    elif add_recurrent_projection:
        components.append("# projection matrices : Wrm")
        components.append("component name={0}_Wrm type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3} {4}".format(
            name, cell_dim, recurrent_projection_dim, ng_affine_options, max_change_options))
        components.append("component name={0}_r type=BackpropTruncationComponent dim={1} "
            "clipping-threshold={2} zeroing-threshold={3} zeroing-interval={4} "
            "recurrence-interval={5}".format(name, recurrent_projection_dim, clipping_threshold,
            zeroing_threshold, zeroing_interval, abs(lstm_delay)))
        component_nodes.append("# r_t")
        component_nodes.append("component-node name={0}_r_t_preclip component={0}_Wrm input={0}_m_t".format(name))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = recurrent_projection_dim

    else:
        components.append("component name={0}_r type=BackpropTruncationComponent dim={1} "
            "clipping-threshold={2} zeroing-threshold={3} zeroing-interval={4} "
            "recurrence-interval={5}".format(name, cell_dim, clipping_threshold,
            zeroing_threshold, zeroing_interval, abs(lstm_delay)))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_m_t".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = cell_dim

    return {
            'descriptor': output_descriptor,
            'dimension':output_dim
            }

def AddBLstmLayer(config_lines,
                  name, input, cell_dim,
                  recurrent_projection_dim = 0,
                  non_recurrent_projection_dim = 0,
                  clipping_threshold = 1.0,
                  zeroing_threshold = 3.0,
                  zeroing_interval = 20,
                  ng_per_element_scale_options = "",
                  ng_affine_options = "",
                  lstm_delay = [-1,1],
                  self_repair_scale_nonlinearity = None,
                  max_change_per_component = 0.75):
    assert(len(lstm_delay) == 2 and lstm_delay[0] < 0 and lstm_delay[1] > 0)
    output_forward = AddLstmLayer(config_lines = config_lines,
                                  name = "{0}_forward".format(name),
                                  input = input,
                                  cell_dim = cell_dim,
                                  recurrent_projection_dim = recurrent_projection_dim,
                                  non_recurrent_projection_dim = non_recurrent_projection_dim,
                                  clipping_threshold = clipping_threshold,
                                  zeroing_threshold = zeroing_threshold,
                                  zeroing_interval = zeroing_interval,
                                  ng_per_element_scale_options = ng_per_element_scale_options,
                                  ng_affine_options = ng_affine_options,
                                  lstm_delay = lstm_delay[0],
                                  self_repair_scale_nonlinearity = self_repair_scale_nonlinearity,
                                  max_change_per_component = max_change_per_component)
    output_backward = AddLstmLayer(config_lines = config_lines,
                                   name = "{0}_backward".format(name),
                                   input = input,
                                   cell_dim = cell_dim,
                                   recurrent_projection_dim = recurrent_projection_dim,
                                   non_recurrent_projection_dim = non_recurrent_projection_dim,
                                   clipping_threshold = clipping_threshold,
                                   zeroing_threshold = zeroing_threshold,
                                   zeroing_interval = zeroing_interval,
                                   ng_per_element_scale_options = ng_per_element_scale_options,
                                   ng_affine_options = ng_affine_options,
                                   lstm_delay = lstm_delay[1],
                                   self_repair_scale_nonlinearity = self_repair_scale_nonlinearity,
                                   max_change_per_component = max_change_per_component)
    output_descriptor = 'Append({0}, {1})'.format(output_forward['descriptor'], output_backward['descriptor'])
    output_dim = output_forward['dimension'] + output_backward['dimension']

    return {
            'descriptor': output_descriptor,
            'dimension':output_dim
            }

