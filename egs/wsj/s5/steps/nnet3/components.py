#!/usr/bin/env python

from __future__ import print_function
import os 
import math
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

def AddGradientScaleLayer(config_lines, name, input, scale = 1.0, scales_vec = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    if scales_vec is None:
        components.append('component name={0}_gradient_scale type=ScaleGradientComponent dim={1} scale={2}'.format(name, input['dimension'], scale))
    else:
        components.append('component name={0}_gradient_scale type=ScaleGradientComponent scales={2}'.format(name, scales_vec))

    component_nodes.append('component-node name={0}_gradient_scale component={0}_gradient_scale input={1}'.format(name, input['descriptor']))

    return {'descriptor':  '{0}_gradient_scale'.format(name),
            'dimension': input['dimension']}

def AddFixedScaleLayer(config_lines, name, input,
                       scale = 1.0, scales_vec = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    if scales_vec is None:
        components.append('component name={0}-fixed-scale type=FixedScaleComponent dim={1} scale={2}'.format(name, input['dimension'], scale))
    else:
        components.append('component name={0}-fixed-scale type=FixedScaleComponent scales={2}'.format(name, scales_vec))

    component_nodes.append('component-node name={0}-fixed-scale component={0}-fixed-scale input={1}'.format(name, input['descriptor']))

    return {'descriptor':  '{0}-fixed-scale'.format(name),
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

def AddAffineLayer(config_lines, name, input, output_dim, ng_affine_options = ""):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))

    return {'descriptor':  '{0}_affine'.format(name),
            'dimension': output_dim}

def AddAffRelNormLayer(config_lines, name, input, output_dim, ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0, self_repair_scale = None, add_norm_layer = True):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    # self_repair_scale is a constant scaling the self-repair vector computed in RectifiedLinearComponent
    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1} {2}".format(name, output_dim, self_repair_string))
    if add_norm_layer:
      components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, output_dim, norm_target_rms))

    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_relu component={0}_relu input={0}_affine".format(name))
    if add_norm_layer:
      component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_relu".format(name))

    return {'descriptor':  ('{0}_renorm'.format(name) if add_norm_layer is True else '{0}_relu'.format(name)),
            'dimension': output_dim}

def AddAffRelNormWithEphemeralLayer(config_lines, name, input, ephemeral_name, ephemeral_input, output_dim, ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0, self_repair_scale = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    # self_repair_scale is a constant scaling the self-repair vector computed in RectifiedLinearComponent
    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1} {2}".format(name, output_dim, self_repair_string))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, output_dim, norm_target_rms))

    components.append("component name={0}_ephemeral type=NaturalGradientAffineComponent input-dim={1} output-dim={2}".format(ephemeral_name, ephemeral_input['dimension'], output_dim))
    components.append("component name={0}_dropout_affine type=DropoutComponent dim={1} dropout-proportion=0.0".format(ephemeral_name, output_dim))

    component_nodes.append("component-node name={0}_ephemeral component={0}_ephemeral input={1}".format(ephemeral_name, ephemeral_input['descriptor']))
    component_nodes.append("component-node name={0}_dropout_affine component={0}_dropout_affine input={0}_ephemeral".format(ephemeral_name))

    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_relu component={0}_relu input=Sum({0}_affine, {1}_dropout_affine)".format(name, ephemeral_name))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_relu ".format(name))

    return {'descriptor':  '{0}_renorm'.format(name),
            'dimension': output_dim}

# Layer with SVD layer type and ephemeral connection
def AddAffRelNormSvdWithEphemeralLayer(config_lines, name, input, ephemeral_name, svd_dim, output_dim, ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0, self_repair_scale = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    param_stddev = 0.3/math.sqrt(svd_dim)
    ng_affine_for_svd_options = " bias-stddev=0  param-stddev={0} ".format(param_stddev)
    # self_repair_scale is a constant scaling the self-repair vector computed in RectifiedLinearComponent
    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''

    # svd components
    components.append("component name={0}_affine_svd type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], svd_dim, ng_affine_options))
    components.append("component name={0}_relu_svd type=RectifiedLinearComponent dim={1} {2}".format(name, svd_dim, self_repair_string))
    components.append("component name={0}_renorm_svd type=NormalizeComponent dim={1} target-rms={2}".format(name, svd_dim, norm_target_rms))

    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, svd_dim, output_dim, ng_affine_for_svd_options))
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1} {2}".format(name, output_dim, self_repair_string))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, output_dim, norm_target_rms))

    # ephemeral components and component nodes
    components.append("component name={0}_ephemeral type=NaturalGradientAffineComponent input-dim={1} output-dim={2}".format(ephemeral_name, input['dimension'], output_dim))
    components.append("component name={0}_dropout_affine type=DropoutComponent dim={1} dropout-proportion=0.0".format(ephemeral_name, output_dim))
    
    component_nodes.append("component-node name={0}_ephemeral component={0}_ephemeral input={1}".format(ephemeral_name, input['descriptor']))
    component_nodes.append("component-node name={0}_dropout_affine component={0}_dropout_affine input={0}_ephemeral".format(ephemeral_name))

    # svd component node
    component_nodes.append("component-node name={0}_affine_svd component={0}_affine_svd input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_relu_svd component={0}_relu_svd input={0}_affine_svd".format(name, ephemeral_name))
    component_nodes.append("component-node name={0}_renorm_svd component={0}_renorm_svd input={0}_relu_svd".format(name))

    component_nodes.append("component-node name={0}_affine component={0}_affine input={0}_renorm_svd".format(name))
    component_nodes.append("component-node name={0}_relu component={0}_relu input=Sum({0}_affine, {1}_dropout_affine)".format(name, ephemeral_name))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_relu".format(name, ephemeral_name))

    return {'descriptor':  '{0}_renorm'.format(name),
            'dimension': output_dim}

# no affine before ephemeral connection
def AddAffRelNormWithDirectEphemeralLayer(config_lines, name, input, ephemeral_name, ephemeral_input, output_dim, ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0, self_repair_scale = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    # self_repair_scale is a constant scaling the self-repair vector computed in RectifiedLinearComponent
    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1} {2}".format(name, output_dim, self_repair_string))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, output_dim, norm_target_rms))

    components.append("component name={0}_dropout type=DropoutComponent dim={1} dropout-proportion=0.0".format(ephemeral_name, ephemeral_input['dimension']))
    components.append("component name={0}_append type=NoOpComponent dim={1}".format(name, input['dimension']))
    
    component_nodes.append("component-node name={0}_dropout component={0}_dropout input={1}".format(ephemeral_name, ephemeral_input['descriptor']))
    component_nodes.append("component-node name={0}_append component={0}_append input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_affine component={0}_affine input={0}_append".format(name, ephemeral_name))
    component_nodes.append("component-node name={0}_relu component={0}_relu input={0}_affine".format(name, ephemeral_name))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input=Sum({0}_relu, {0}_dropout)".format(name))

    return {'descriptor':  '{0}_renorm'.format(name),
            'dimension': output_dim}
# no affine before ephemeral connection
def AddAffRelNormWithDirectEphemeralLayerV3(config_lines, name, input, ephemeral_name, ephemeral_input, output_dim, ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0, self_repair_scale = None, block_affine_scale = 0.2):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    dropout_affine_options = " param-stddev={0} bias-stddev=0".format(block_affine_scale)
    # self_repair_scale is a constant scaling the self-repair vector computed in RectifiedLinearComponent
    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1} {2}".format(name, output_dim, self_repair_string))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, output_dim, norm_target_rms))

    components.append("component name={0}_affine_ephemeral type=BlockAffineComponent input-dim={1} output-dim={1} num-blocks={1} {2}".format(ephemeral_name, ephemeral_input['dimension'], dropout_affine_options))
    components.append("component name={0}_dropout type=DropoutComponent dim={1} dropout-proportion=0.0".format(ephemeral_name, ephemeral_input['dimension']))
   
    component_nodes.append("component-node name={0}_affine_ephemeral component={0}_affine_ephemeral input={1}".format(ephemeral_name, ephemeral_input['descriptor']))
    component_nodes.append("component-node name={0}_dropout component={0}_dropout input={0}_affine_ephemeral".format(ephemeral_name))
    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_relu component={0}_relu input={0}_affine".format(name))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input=Sum({0}_relu, {1}_dropout)".format(name, ephemeral_name))

    return {'descriptor':  '{0}_renorm'.format(name),
            'dimension': output_dim}
# no affine before ephemeral connection
def AddAffRelNormWithDirectEphemeralLayerV2(config_lines, name, input, ephemeral_name, ephemeral_input, output_dim, ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0, self_repair_scale = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    # self_repair_scale is a constant scaling the self-repair vector computed in RectifiedLinearComponent
    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1} {2}".format(name, output_dim, self_repair_string))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, output_dim, norm_target_rms))
    
    components.append("component name={0}_affine_ephemeral type=BlockAffineComponent input-dim={1} output-dim={1} num-blocks={1} {2}".format(ephemeral_name, ephemeral_input['dimension'], ng_affine_options))
    components.append("component name={0}_dropout type=DropoutComponent dim={1} dropout-proportion=0.0".format(ephemeral_name, ephemeral_input['dimension']))
    components.append("component name={0}_append type=NoOpComponent dim={1}".format(name, input['dimension']))
   
    component_nodes.append("component-node name={0}_affine_ephemeral component={0}_affine_ephemeral input={1}".format(ephemeral_name, ephemeral_input['descriptor']))
    component_nodes.append("component-node name={0}_dropout component={0}_dropout input={0}_affine_ephemeral".format(ephemeral_name))
    component_nodes.append("component-node name={0}_append component={0}_append input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_affine component={0}_affine input=Sum({0}_append, {1}_dropout)".format(name, ephemeral_name))
    component_nodes.append("component-node name={0}_relu component={0}_relu input={0}_affine".format(name, ephemeral_name))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_relu".format(name))

    return {'descriptor':  '{0}_renorm'.format(name),
            'dimension': output_dim}
# no affine before ephemeral connection
# This layer is the same as V2 but it has separate ReLU gate for ephemeral connection
def AddAffRelNormWithDirectEphemeralLayerV4(config_lines, name, input, ephemeral_name, ephemeral_input, output_dim, ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0, self_repair_scale = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    # self_repair_scale is a constant scaling the self-repair vector computed in RectifiedLinearComponent
    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1} {2}".format(name, output_dim, self_repair_string))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, output_dim, norm_target_rms))
    
    components.append("component name={0}_affine_ephemeral type=BlockAffineComponent input-dim={1} output-dim={1} num-blocks={1} {2}".format(ephemeral_name, ephemeral_input['dimension'], ng_affine_options))
    components.append("component name={0}_relu_ephemeral type=RectifiedLinearComponent dim={1} {2}".format(ephemeral_name, ephemeral_input['dimension'], self_repair_string))
    components.append("component name={0}_dropout type=DropoutComponent dim={1} dropout-proportion=0.0".format(ephemeral_name, ephemeral_input['dimension']))
    components.append("component name={0}_append type=NoOpComponent dim={1}".format(name, input['dimension']))
   
    component_nodes.append("component-node name={0}_affine_ephemeral component={0}_affine_ephemeral input={1}".format(ephemeral_name, ephemeral_input['descriptor']))
    component_nodes.append("component-node name={0}_relu_ephemeral component={0}_relu_ephemeral input={0}_affine_ephemeral".format(ephemeral_name))
    component_nodes.append("component-node name={0}_dropout component={0}_dropout input={0}_relu_ephemeral".format(ephemeral_name))
    component_nodes.append("component-node name={0}_append component={0}_append input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_affine component={0}_affine input=Sum({0}_append, {1}_dropout)".format(name, ephemeral_name))
    component_nodes.append("component-node name={0}_relu component={0}_relu input={0}_affine".format(name, ephemeral_name))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_relu".format(name))

    return {'descriptor':  '{0}_renorm'.format(name),
            'dimension': output_dim}
# no affine before ephemeral connection
# This layer is the same as V4 but it has separate sigmoid gate for ephemeral connection
def AddAffRelNormWithDirectEphemeralLayerV5(config_lines, name, input, ephemeral_name, ephemeral_input, output_dim, ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0, self_repair_scale = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    # self_repair_scale is a constant scaling the self-repair vector computed in RectifiedLinearComponent
    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1} {2}".format(name, output_dim, self_repair_string))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, output_dim, norm_target_rms))
    
    components.append("component name={0}_affine_ephemeral type=NaturalGradientAffineComponent input-dim={1} output-dim={1} {2}".format(ephemeral_name, ephemeral_input['dimension'], ng_affine_options))
    components.append("component name={0}_sigmoid_ephemeral type=SigmoidComponent dim={1} {2}".format(ephemeral_name, ephemeral_input['dimension'], self_repair_string))
    components.append("component name={0}_gate_ephemeral type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(ephemeral_name, 2*ephemeral_input['dimension'], ephemeral_input['dimension']))
    components.append("component name={0}_dropout type=DropoutComponent dim={1} dropout-proportion=0.0".format(ephemeral_name, ephemeral_input['dimension']))
    components.append("component name={0}_append type=NoOpComponent dim={1}".format(name, input['dimension']))
    components.append("component name={0}_append_ephemeral type=NoOpComponent dim={1}".format(ephemeral_name, ephemeral_input['dimension']))
   
    component_nodes.append("component-node name={0}_affine_ephemeral component={0}_affine_ephemeral input={1}".format(ephemeral_name, ephemeral_input['descriptor']))
    component_nodes.append("component-node name={0}_sigmoid_ephemeral component={0}_sigmoid_ephemeral input={0}_affine_ephemeral".format(ephemeral_name))
    component_nodes.append("component-node name={0}_append_ephemeral component={0}_append_ephemeral input={1}".format(ephemeral_name, ephemeral_input['descriptor'])) 
    component_nodes.append("component-node name={0}_gate_ephemeral component={0}_gate_ephemeral input=Append({0}_sigmoid_ephemeral, {0}_append_ephemeral)".format(ephemeral_name))
    component_nodes.append("component-node name={0}_dropout component={0}_dropout input={0}_gate_ephemeral".format(ephemeral_name))
    component_nodes.append("component-node name={0}_append component={0}_append input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_affine component={0}_affine input=Sum({0}_append, {1}_dropout)".format(name, ephemeral_name))
    component_nodes.append("component-node name={0}_relu component={0}_relu input={0}_affine".format(name, ephemeral_name))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_relu".format(name))

    return {'descriptor':  '{0}_renorm'.format(name),
            'dimension': output_dim}

# no affine before ephemeral connection
# This layer is the same as V5 but it has separate sigmoid gate with full transformation and diagonal matrix for scaling for ephemeral connection
def AddAffRelNormWithDirectEphemeralLayerV6(config_lines, name, input, ephemeral_name, ephemeral_input, output_dim, ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0, self_repair_scale = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    # self_repair_scale is a constant scaling the self-repair vector computed in RectifiedLinearComponent
    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1} {2}".format(name, output_dim, self_repair_string))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, output_dim, norm_target_rms))
    
    components.append("component name={0}_affine_ephemeral type=NaturalGradientAffineComponent input-dim={1} output-dim={1} {2}".format(ephemeral_name, ephemeral_input['dimension'], ng_affine_options))
    components.append("component name={0}_diag_ephemeral type=BlockAffineComponent input-dim={1} output-dim={1} num-blocks={1} {2}".format(ephemeral_name, ephemeral_input['dimension'], ng_affine_options))
    components.append("component name={0}_sigmoid_ephemeral type=SigmoidComponent dim={1} {2}".format(ephemeral_name, ephemeral_input['dimension'], self_repair_string))
    components.append("component name={0}_gate_ephemeral type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(ephemeral_name, 2*ephemeral_input['dimension'], ephemeral_input['dimension']))
    components.append("component name={0}_dropout type=DropoutComponent dim={1} dropout-proportion=0.0".format(ephemeral_name, ephemeral_input['dimension']))
    components.append("component name={0}_append_ephemeral type=NoOpComponent dim={1}".format(name, input['dimension']))
    components.append("component name={0}_append type=NoOpComponent dim={1}".format(name, input['dimension']))

    component_nodes.append("component-node name={0}_affine_ephemeral component={0}_affine_ephemeral input={1}".format(ephemeral_name, ephemeral_input['descriptor']))
    component_nodes.append("component-node name={0}_sigmoid_ephemeral component={0}_sigmoid_ephemeral input={0}_affine_ephemeral".format(ephemeral_name))
    component_nodes.append("component-node name={0}_diag_ephemeral component={0}_diag_ephemeral input={1}".format(ephemeral_name, ephemeral_input['descriptor'])) 
    component_nodes.append("component-node name={0}_gate_ephemeral component={0}_gate_ephemeral input=Append({0}_sigmoid_ephemeral, {0}_diag_ephemeral)".format(ephemeral_name))
    component_nodes.append("component-node name={0}_dropout component={0}_dropout input={0}_gate_ephemeral".format(ephemeral_name))

    component_nodes.append("component-node name={0}_append component={0}_append input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_affine component={0}_affine input=Sum({0}_append, {1}_dropout)".format(name, ephemeral_name))
    component_nodes.append("component-node name={0}_relu component={0}_relu input={0}_affine".format(name, ephemeral_name))
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
        label_delay=None,
        use_presoftmax_prior_scale = False,
        prior_scale_file = None,
        include_log_softmax = True,
        add_final_sigmoid = False,
        name_affix = None,
        objective_type = "linear",
        objective_scale = 1.0,
        objective_scales_vec = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    if name_affix is not None:
        final_node_prefix = 'Final-' + str(name_affix)
    else:
        final_node_prefix = 'Final'

    prev_layer_output = AddAffineLayer(config_lines,
            final_node_prefix , input, output_dim,
            ng_affine_options)
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
    if (objective_scale != 1.0 or objective_scales_vec is not None):
        prev_layer_output = AddGradientScaleLayer(config_lines, final_node_prefix, prev_layer_output, objective_scale, objective_scales_vec)

    AddOutputLayer(config_lines, prev_layer_output, label_delay, suffix = name_affix, objective_type = objective_type)

def AddLstmLayer(config_lines,
                 name, input, cell_dim,
                 recurrent_projection_dim = 0,
                 non_recurrent_projection_dim = 0,
                 clipping_threshold = 1.0,
                 norm_based_clipping = "false",
                 ng_per_element_scale_options = "",
                 ng_affine_options = "",
                 lstm_delay = -1,
                 self_repair_scale_nonlinearity = None,
                 self_repair_scale_clipgradient = None):
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
    # self_repair_scale_clipgradient is a constant scaling the self-repair vector computed in ClipGradientComponent
    self_repair_clipgradient_string = "self-repair-scale={0:.2f}".format(self_repair_scale_clipgradient) if self_repair_scale_clipgradient is not None else ''
    # Natural gradient per element scale parameters
    ng_per_element_scale_options += " param-mean=0.0 param-stddev=1.0 "
    # Parameter Definitions W*(* replaced by - to have valid names)
    components.append("# Input gate control : W_i* matrices")
    components.append("component name={0}_W_i-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_ic type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("# Forget gate control : W_f* matrices")
    components.append("component name={0}_W_f-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_fc type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("#  Output gate control : W_o* matrices")
    components.append("component name={0}_W_o-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_oc type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("# Cell input matrices : W_c* matrices")
    components.append("component name={0}_W_c-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))


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
    components.append("component name={0}_c type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} {4}".format(name, cell_dim, clipping_threshold, norm_based_clipping, self_repair_clipgradient_string))

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
        components.append("component name={0}_W-m type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim + non_recurrent_projection_dim, ng_affine_options))
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} {4}".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping, self_repair_clipgradient_string))
        component_nodes.append("# r_t and p_t")
        component_nodes.append("component-node name={0}_rp_t component={0}_W-m input={0}_m_t".format(name))
        component_nodes.append("dim-range-node name={0}_r_t_preclip input-node={0}_rp_t dim-offset=0 dim={1}".format(name, recurrent_projection_dim))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_rp_t'.format(name)
        output_dim = recurrent_projection_dim + non_recurrent_projection_dim

    elif add_recurrent_projection:
        components.append("# projection matrices : Wrm")
        components.append("component name={0}_Wrm type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim, ng_affine_options))
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} {4}".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping, self_repair_clipgradient_string))
        component_nodes.append("# r_t")
        component_nodes.append("component-node name={0}_r_t_preclip component={0}_Wrm input={0}_m_t".format(name))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = recurrent_projection_dim

    else:
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} {4}".format(name, cell_dim, clipping_threshold, norm_based_clipping, self_repair_clipgradient_string))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_m_t".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = cell_dim

    return {
            'descriptor': output_descriptor,
            'dimension':output_dim
            }
def GenerateDescriptor(splice_indexes, subset_dim, prev_layer_output):
  try:
      zero_index = splice_indexes.index(0)
  except ValueError:
      zero_index = None
  # I just assume the prev_layer_output_descriptor is a simple forwarding descriptor
  prev_layer_output_descriptor = prev_layer_output['descriptor']
  subset_output = prev_layer_output
  if subset_dim > 0:
      # if subset_dim is specified the script expects a zero in the splice indexes
      assert(zero_index is not None)
      subset_node_config = "dim-range-node name=Tdnn_input_{0} input-node={1} dim-offset={2} dim={3}".format(i, prev_layer_output_descriptor, 0, subset_dim)
      subset_output = {'descriptor' : 'Tdnn_input_{0}'.format(i),
                       'dimension' : subset_dim}
      config_lines['component-nodes'].append(subset_node_config)
  appended_descriptors = []
  appended_dimension = 0
  for j in range(len(splice_indexes)):
      if j == zero_index:
          appended_descriptors.append(prev_layer_output['descriptor'])
          appended_dimension += prev_layer_output['dimension']
          continue
      appended_descriptors.append('Offset({0}, {1})'.format(subset_output['descriptor'], splice_indexes[j]))
      appended_dimension += subset_output['dimension']
  return {'descriptor' : "Append({0})".format(" , ".join(appended_descriptors)),
                       'dimension'  : appended_dimension}

def AddBLstmLayer(config_lines,
                  name, input, cell_dim,
                  recurrent_projection_dim = 0,
                  non_recurrent_projection_dim = 0,
                  clipping_threshold = 1.0,
                  norm_based_clipping = "false",
                  ng_per_element_scale_options = "",
                  ng_affine_options = "",
                  lstm_delay = [-1,1],
                  self_repair_scale_nonlinearity = None,
                  self_repair_scale_clipgradient = None):
    assert(len(lstm_delay) == 2 and lstm_delay[0] < 0 and lstm_delay[1] > 0)
    output_forward = AddLstmLayer(config_lines, "{0}_forward".format(name), input, cell_dim,
                                  recurrent_projection_dim, non_recurrent_projection_dim,
                                  clipping_threshold, norm_based_clipping,
                                  ng_per_element_scale_options, ng_affine_options,
                                  lstm_delay = lstm_delay[0],
                                  self_repair_scale_nonlinearity = self_repair_scale_nonlinearity,
                                  self_repair_scale_clipgradient = self_repair_scale_clipgradient)
    output_backward = AddLstmLayer(config_lines, "{0}_backward".format(name), input, cell_dim,
                                   recurrent_projection_dim, non_recurrent_projection_dim,
                                   clipping_threshold, norm_based_clipping,
                                   ng_per_element_scale_options, ng_affine_options,
                                   lstm_delay = lstm_delay[1],
                                   self_repair_scale_nonlinearity = self_repair_scale_nonlinearity,
                                   self_repair_scale_clipgradient = self_repair_scale_clipgradient)
    output_descriptor = 'Append({0}, {1})'.format(output_forward['descriptor'], output_backward['descriptor'])
    output_dim = output_forward['dimension'] + output_backward['dimension']

    return {
            'descriptor': output_descriptor,
            'dimension':output_dim
            }

def AddSigmoidGate(config_lines, name, input, dp_prop = 0.0, self_repair_scale_nonlinearity=None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    self_repair_nonlinearity_string = "self-repair-scale={0:.10f}".format(self_repair_scale_nonlinearity) if self_repair_scale_nonlinearity is not None else ''

    components.append("component name={0}_gate_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={1}".format(name, input['dimension']))
    components.append("component name={0}_gate_sigmoid type=SigmoidComponent dim={1} {2}".format(name, input['dimension'], self_repair_nonlinearity_string))
    components.append("component name={0}_gate type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * input['dimension'], input['dimension']))
    components.append("component name={0}_gate_dropout type=DropoutComponent dim={1} dropout-proportion={2}".format(name, input['dimension'], dp_prop))

    component_nodes.append("component-node name={0}_gate_affine component={0}_gate_affine input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_gate_sigmoid component={0}_gate_sigmoid input={0}_gate_affine".format(name))
    component_nodes.append("component-node name={0}_gate component={0}_gate input=Append({0}_gate_affine, {1})".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_gate_dropout component={0}_gate_dropout input={0}_gate".format(name))
    return { 
            'descriptor' : '{0}_gate_dropout'.format(name),
            'dimension' : input['dimension']
           }   
def AddTwinBLstmLayerRegularizer(config_lines,
                  name, forward_input, backward_input, cell_dim,
                  recurrent_projection_dim = 0,
                  non_recurrent_projection_dim = 0,
                  clipping_threshold = 1.0,
                  norm_based_clipping = "false",
                  ng_per_element_scale_options = "",
                  ng_affine_options = "",
                  lstm_delay = [-1,1],
                  self_repair_scale_nonlinearity = None,
                  self_repair_scale_clipgradient = None,
                  add_reg = False,
                  append_twins = False):
    assert(len(lstm_delay) == 2 and lstm_delay[0] < 0 and lstm_delay[1] > 0)
    output_forward = AddLstmLayer(config_lines, "{0}_forward".format(name), forward_input, cell_dim,
                                  recurrent_projection_dim, non_recurrent_projection_dim,
                                  clipping_threshold, norm_based_clipping,
                                  ng_per_element_scale_options, ng_affine_options,
                                  lstm_delay = lstm_delay[0],
                                  self_repair_scale_nonlinearity = self_repair_scale_nonlinearity,
                                  self_repair_scale_clipgradient = self_repair_scale_clipgradient)
    output_backward = AddLstmLayer(config_lines, "{0}_backward".format(name), backward_input, cell_dim,
                                   recurrent_projection_dim, non_recurrent_projection_dim,
                                   clipping_threshold, norm_based_clipping,
                                   ng_per_element_scale_options, ng_affine_options,
                                   lstm_delay = lstm_delay[1],
                                   self_repair_scale_nonlinearity = self_repair_scale_nonlinearity,
                                   self_repair_scale_clipgradient = self_repair_scale_clipgradient)


    if add_reg:
      # add affine layer for regularizer term NormRel(HX)
      regularize_name = "{0}_regularize".format(name);
      regularize_output = AddAffineLayer(config_lines, regularize_name, output_forward, output_backward['dimension'], ng_affine_options = ng_affine_options)

    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    
    
    if add_reg:
      # component to compute regularizer term l2_norm(HX - X_twin)
      components.append("component name={0}_negate_twin type=FixedScaleComponent dim={1} scale={2}".format(name, output_backward['dimension'], -1.0))
      component_nodes.append("component-node name={0}_negate_twin component={0}_negate_twin input={1}".format(name, output_backward['descriptor']))
     
      components.append("component name={0}_regularizer type=NoOpComponent dim={1}".format(name, output_backward['dimension']))
      component_nodes.append("component-node name={0}_regularizer component={0}_regularizer input=Sum({1}, {0}_negate_twin)".format(name, regularize_output['descriptor']))

      components.append("component name={0}_scaled_regularizer type=FixedScaleComponent dim={1} scale={2}".format(name, output_backward['dimension'], 1.0/output_backward['dimension']))
      component_nodes.append("component-node name={0}_scaled_regularizer component={0}_scaled_regularizer input={0}_regularizer".format(name))


      # Dropout component for DP(Append(X_twin, HX), dp), which applies dp on X_twin and 1-dp and its complement on HX.
      components.append("component name={0}_dropout_twin_regularize type=DropoutComponent dim={1} dropout-proportion=0.0 complement=true".format(name, 2*output_backward['dimension']))
      component_nodes.append("component-node name={0}_dropout_twin_regularize component={0}_dropout_twin_regularize input=Append({1}, {2})".format(name, output_backward['descriptor'], regularize_output['descriptor']))
       
      # component node to sum x + DP(Hx, 1-dp) + DP(x_twin, dp)
      # dim-range nodes for two subset as regularization term Hx and twin part Y
      component_nodes.append("dim-range-node name={0}_dropout_twin input-node={0}_dropout_twin_regularize dim-offset=0 dim={1}".format(name, output_backward['dimension']))
      component_nodes.append("dim-range-node name={0}_dropout_regularize input-node={0}_dropout_twin_regularize dim-offset={1} dim={1}".format(name, output_backward['dimension']))

      # component to connect twins using ephemeral connection
      if append_twins:
        final_output_dim = output_forward['dimension']+output_backward['dimension']
        components.append("component name={0}_sum_twins type=NoOpComponent dim={1}".format(name, final_output_dim))
        component_nodes.append("component-node name={0}_renorm_sum component={0}_sum_twins input=Append({1}, Sum({0}_dropout_twin, {0}_dropout_regularize))".format(name, output_forward['descriptor']))
      else:
        assert(output_forward['dimension'] == output_backward['dimension'])
        final_output_dim = output_forward['dimension']
        components.append("component name={0}_sum_twins type=NoOpComponent dim={1}".format(name, final_output_dim))
        component_nodes.append("component-node name={0}_renorm_sum component={0}_sum_twins input=Sum({1}, Sum({0}_dropout_twin, {0}_dropout_regularize))".format(name, output_forward['descriptor']))
    else:
      # Dropout for X_twin
      components.append("component name={0}_dropout type=DropoutComponent dim={1} dropout-proportion=0.0".format(name, output_backward['dimension']))
      component_nodes.append("component-node name={0}_dropout component={0}_dropout input={1}".format(name, output_backward['descriptor']))
      if append_twins:
        final_output_dim = output_forward['dimension']+output_backward['dimension']
        components.append("component name={0}_sum_twins type=NoOpComponent dim={1}".format(name, final_output_dim))
        component_nodes.append("component-node name={0}_renorm_sum component={0}_sum_twins input=Append({1}, {0}_dropout)".format(name, output_forward['descriptor']))
      else:
        assert(output_forward['dimension'] == output_backward['dimension'])
        final_output_dim = output_forward['dimension']
        components.append("component name={0}_sum_twins type=NoOpComponent dim={1}".format(name, final_output_dim))
        component_nodes.append("component-node name={0}_renorm_sum component={0}_sum_twins input=Sum({1}, {0}_dropout)".format(name, output_forward['descriptor']))

    output_descriptor = 'Append({0}, {1})'.format(output_forward['descriptor'], output_backward['descriptor'])
    output_dim = output_forward['dimension'] + output_backward['dimension']



    return [{
            'descriptor': '{0}_renorm_sum'.format(name),
            'dimension': final_output_dim,
            'regularizer':('{0}_scaled_regularizer'.format(name) if add_reg is True else ''),
            },
            { 
            'descriptor': output_descriptor,
            'dimension': output_dim
            }
            ]

def AddTwinBLstmLayerRegularizer2(config_lines,
                  name, forward_input, backward_input, cell_dim,
                  recurrent_projection_dim = 0,
                  non_recurrent_projection_dim = 0,
                  clipping_threshold = 1.0,
                  norm_based_clipping = "false",
                  ng_per_element_scale_options = "",
                  ng_affine_options = "",
                  lstm_delay = [-1,1],
                  self_repair_scale_nonlinearity = None,
                  self_repair_scale_clipgradient = None,
                  add_reg = False,
                  append_twins = False,
                  reg_term=1.0):
    assert(len(lstm_delay) == 2 and lstm_delay[0] < 0 and lstm_delay[1] > 0)
    output_forward = AddLstmLayer(config_lines, "{0}_forward".format(name), forward_input, cell_dim,
                                  recurrent_projection_dim, non_recurrent_projection_dim,
                                  clipping_threshold, norm_based_clipping,
                                  ng_per_element_scale_options, ng_affine_options,
                                  lstm_delay = lstm_delay[0],
                                  self_repair_scale_nonlinearity = self_repair_scale_nonlinearity,
                                  self_repair_scale_clipgradient = self_repair_scale_clipgradient)
    output_backward = AddLstmLayer(config_lines, "{0}_backward".format(name), backward_input, cell_dim,
                                   recurrent_projection_dim, non_recurrent_projection_dim,
                                   clipping_threshold, norm_based_clipping,
                                   ng_per_element_scale_options, ng_affine_options,
                                   lstm_delay = lstm_delay[1],
                                   self_repair_scale_nonlinearity = self_repair_scale_nonlinearity,
                                   self_repair_scale_clipgradient = self_repair_scale_clipgradient)


    if add_reg:
      # add affine layer for regularizer term ReLU(HX)
      regularize_name = "{0}_regularize".format(name);
      regularize_output = AddAffineLayer(config_lines, regularize_name, output_forward, output_backward['dimension'], ng_affine_options = ng_affine_options)
      
      # add affine layer for twin as ReLU(H2*X_twin)
      twin_regularize_name = "{0}_regularize_twin".format(name);
      twin_regularize_output = AddAffineLayer(config_lines, twin_regularize_name, output_backward, output_backward['dimension'], ng_affine_options = ng_affine_options)


    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    
    
    if add_reg:
      # component to compute regularizer term l2_norm(HX - H2*X_twin)
      components.append("component name={0}_negate_regularize_twin type=FixedScaleComponent dim={1} scale={2}".format(name, output_backward['dimension'], -1.0))
      component_nodes.append("component-node name={0}_negate_regularize_twin component={0}_negate_regularize_twin input={1}".format(name, twin_regularize_output['descriptor']))
     
      components.append("component name={0}_regularizer type=NoOpComponent dim={1}".format(name, output_backward['dimension']))
      component_nodes.append("component-node name={0}_regularizer component={0}_regularizer input=Sum({1}, {0}_negate_regularize_twin)".format(name, regularize_output['descriptor']))

      components.append("component name={0}_scaled_regularizer type=FixedScaleComponent dim={1} scale={2}".format(name, output_backward['dimension'], reg_term/output_backward['dimension']))
      component_nodes.append("component-node name={0}_scaled_regularizer component={0}_scaled_regularizer input={0}_regularizer".format(name))


      # Dropout component for DP(Append(H2X_twin, HX), dp), which applies dp on X_twin and 1-dp and its complement on HX.
      components.append("component name={0}_dropout_twin_regularize type=DropoutComponent dim={1} dropout-proportion=0.0 complement=true".format(name, 2*output_backward['dimension']))
      component_nodes.append("component-node name={0}_dropout_twin_regularize component={0}_dropout_twin_regularize input=Append({1}, {2})".format(name, twin_regularize_output['descriptor'], regularize_output['descriptor']))
       
      # component node to sum x + DP(Hx, 1-dp) + DP(x_twin, dp)
      # dim-range nodes for two subset as regularization term Hx and twin part Y
      component_nodes.append("dim-range-node name={0}_dropout_twin input-node={0}_dropout_twin_regularize dim-offset=0 dim={1}".format(name, output_backward['dimension']))
      component_nodes.append("dim-range-node name={0}_dropout_regularize input-node={0}_dropout_twin_regularize dim-offset={1} dim={1}".format(name, output_backward['dimension']))

      # component to connect twins using ephemeral connection
      if append_twins:
        final_output_dim = output_forward['dimension']+output_backward['dimension']
        components.append("component name={0}_sum_twins type=NoOpComponent dim={1}".format(name, final_output_dim))
        component_nodes.append("component-node name={0}_renorm_sum component={0}_sum_twins input=Append({1}, Sum({0}_dropout_twin, {0}_dropout_regularize))".format(name, output_forward['descriptor']))
      else:
        assert(output_forward['dimension'] == output_backward['dimension'])
        final_output_dim = output_forward['dimension']
        components.append("component name={0}_sum_twins type=NoOpComponent dim={1}".format(name, final_output_dim))
        component_nodes.append("component-node name={0}_renorm_sum component={0}_sum_twins input=Sum({1}, Sum({0}_dropout_twin, {0}_dropout_regularize))".format(name, output_forward['descriptor']))
    else:
      # Dropout for X_twin
      components.append("component name={0}_dropout type=DropoutComponent dim={1} dropout-proportion=0.0".format(name, output_backward['dimension']))
      component_nodes.append("component-node name={0}_dropout component={0}_dropout input={1}".format(name, output_backward['descriptor']))
      if append_twins:
        final_output_dim = output_forward['dimension']+output_backward['dimension']
        components.append("component name={0}_sum_twins type=NoOpComponent dim={1}".format(name, final_output_dim))
        component_nodes.append("component-node name={0}_renorm_sum component={0}_sum_twins input=Append({1}, {0}_dropout)".format(name, output_forward['descriptor']))
      else:
        assert(output_forward['dimension'] == output_backward['dimension'])
        final_output_dim = output_forward['dimension']
        components.append("component name={0}_sum_twins type=NoOpComponent dim={1}".format(name, final_output_dim))
        component_nodes.append("component-node name={0}_renorm_sum component={0}_sum_twins input=Sum({1}, {0}_dropout)".format(name, output_forward['descriptor']))
 
    output_descriptor = 'Append({0}, {1})'.format(output_forward['descriptor'], output_backward['descriptor'])
    output_dim = output_forward['dimension'] + output_backward['dimension']

    return [{
            'descriptor': '{0}_renorm_sum'.format(name),
            'dimension': final_output_dim,
            'regularizer':('{0}_scaled_regularizer'.format(name) if add_reg is True else ''),
            },
            { 
            'descriptor': output_descriptor,
            'dimension': output_dim
            }
            ]

# If add_reg is true, it returns the regularizer term RelNorm(HX).X_twin and added term RelNorm(HX) to output. 
# If append_twins is true, HX + X_twin appended to output, otherwise HX + X_twin is added to output.
# In this function, regularizer defined as (HX)^T X_twin - (L1HX)^T (L1HX) - (L2X_twin)^T L2X_twin
# where L1 and L2 are like regularizers. 
def AddTwinBLstmLayerRegularizer3(config_lines,
                  name, forward_input, backward_input, cell_dim,
                  recurrent_projection_dim = 0,
                  non_recurrent_projection_dim = 0,
                  clipping_threshold = 1.0,
                  norm_based_clipping = "false",
                  ng_per_element_scale_options = "",
                  ng_affine_options = "",
                  lstm_delay = [-1,1],
                  self_repair_scale_nonlinearity = None,
                  self_repair_scale_clipgradient = None,
                  add_reg = False,
                  append_twins = False):
    assert(len(lstm_delay) == 2 and lstm_delay[0] < 0 and lstm_delay[1] > 0)
    output_forward = AddLstmLayer(config_lines, "{0}_forward".format(name), forward_input, cell_dim,
                                  recurrent_projection_dim, non_recurrent_projection_dim,
                                  clipping_threshold, norm_based_clipping,
                                  ng_per_element_scale_options, ng_affine_options,
                                  lstm_delay = lstm_delay[0],
                                  self_repair_scale_nonlinearity = self_repair_scale_nonlinearity,
                                  self_repair_scale_clipgradient = self_repair_scale_clipgradient)
    output_backward = AddLstmLayer(config_lines, "{0}_backward".format(name), backward_input, cell_dim,
                                   recurrent_projection_dim, non_recurrent_projection_dim,
                                   clipping_threshold, norm_based_clipping,
                                   ng_per_element_scale_options, ng_affine_options,
                                   lstm_delay = lstm_delay[1],
                                   self_repair_scale_nonlinearity = self_repair_scale_nonlinearity,
                                   self_repair_scale_clipgradient = self_repair_scale_clipgradient)


    if add_reg:
      # add affine layer for regularizer term ReLU(HX)
      regularize_name = "{0}_regularize".format(name)
      regularize_output = AddAffineLayer(config_lines, regularize_name, output_forward, output_backward['dimension'], ng_affine_options = ng_affine_options)
      # add L1, which is extra constraint to normalize H1X in regularizer.
      norm_regularize_name =  "{0}_regularize_norm".format(name)
      norm_regularize_output = AddAffineLayer(config_lines, norm_regularize_name, regularize_output, output_backward['dimension'], ng_affine_options = ng_affine_options)
      
      # add affine layer for twin as ReLU(H2*X_twin)
      twin_regularize_name = "{0}_regularize_twin".format(name);
      twin_regularize_output = AddAffineLayer(config_lines, twin_regularize_name, output_backward, output_backward['dimension'], ng_affine_options = ng_affine_options)
      # add affine transform L2 to normalize H2X_twin in regularizer.
      norm_twin_regularize_name = "{0}_regularize_twin_norm".format(name)
      norm_twin_regularize_output = AddAffineLayer(config_lines, twin_regularize_name, output_backward, output_backward['dimension'], ng_affine_options = ng_affine_options)

    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    
    
    if add_reg:
      # component to compute regularizer term as (H1X.H2X_twin - (L1H1X).(L1H1X) - (L2H2X_twin).(L2H2X_twin))
      ####
      # component to compute regularizer term (HX. X_twin)
      # scale HX.X_twin to minimize cos(HX, X_twin)
      # compute term H1X.H2X_twin
      components.append("component name={0}_regularizer_p1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2*output_dim, output_dim))
      component_nodes.append("component-node name={0}_regularizer_p1 component={0}_regularizer_p1 input=Append({1}, {2})".format(name, regularize_output['descriptor'], twin_regularize_output['descriptor']))
      # compute term L1H1X.(L1H1X)^T
      components.append("component name={0}_regularizer_p2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2*output_dim, output_dim))
      component_nodes.append("component-node name={0}_regularizer_p2 component={0}_regularizer input=Append({1}, {2})".format(name, norm_regularize_output['descriptor'], norm_regularize_output['descriptor']))
      components.append("component name={0}_regularizer_p2_negate type=FixedScaleComponent dim={1} scale=-1.0".format(name, output_dim))
      component_nodes.append("component-node name={0}_regularizer_p2_negate component={0}_regularizer_p2_negate input={0}_regularizer_p2_negate".format(name))

      # compute term L2H2X_twin (L2H2X_twin)^T
      components.append("component name={0}_regularizer_p3 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2*output_dim, output_dim))
      component_nodes.append("component-node name={0}_regularizer_p3 component={0}_regularizer input=Append({1}, {2})".format(name, norm_twin_regularize_output['descriptor'], norm_twin_regularize_output['descriptor']))
      components.append("component name={0}_regularizer_p3_negate type=FixedScaleComponent dim={1} scale=-1.0".format(name, output_dim))
      component_nodes.append("component-node name={0}_regularizer_p3_negate component={0}_regularizer_p3_negate input={0}_regularizer_p3_negate".format(name))


      # sum different parts of regularizer
      components.append("component name={0}_regularizer type=NoOpComponent input-dim={1} output-dim={1}".format(name, output_dim))
      component_nodes.append("component-node name={0}_regularizer component={0}_regularizer input=Sum({0}_regularizer_p1, Sum({0}_regularizer_p2_negate, {0}_regularizer_p3_negate))".format(name))

      components.append("component name={0}_negate_regularize_twin type=FixedScaleComponent dim={1} scale={2}".format(name, output_backward['dimension'], -1.0))
      component_nodes.append("component-node name={0}_negate_regularize_twin component={0}_negate_regularize_twin input={1}".format(name, twin_regularize_output['descriptor']))
     
      components.append("component name={0}_regularizer type=NoOpComponent dim={1}".format(name, output_backward['dimension']))
      component_nodes.append("component-node name={0}_regularizer component={0}_regularizer input=Sum({1}, {0}_negate_regularize_twin)".format(name, regularize_output['descriptor']))

      # scale the regularizer with 1/output-dim to compute mean. 
      components.append("component name={0}_scaled_regularizer type=FixedScaleComponent dim={1} scale={2}".format(name, output_dim, 1.0/output_dim))
      component_nodes.append("component-node name={0}_scaled_regularizer component={0}_scaled_regularizer input={0}_regularizer".format(name)) 


      # Dropout component for DP(Append(H2X_twin, HX), dp), which applies dp on X_twin and 1-dp and its complement on HX.
      components.append("component name={0}_dropout_twin_regularize type=DropoutComponent dim={1} dropout-proportion=0.0 complement=true".format(name, 2*output_backward['dimension']))
      component_nodes.append("component-node name={0}_dropout_twin_regularize component={0}_dropout_twin_regularize input=Append({1}, {2})".format(name, twin_regularize_output['descriptor'], regularize_output['descriptor']))
       
      # component node to sum x + DP(Hx, 1-dp) + DP(x_twin, dp)
      # dim-range nodes for two subset as regularization term Hx and twin part Y
      component_nodes.append("dim-range-node name={0}_dropout_twin input-node={0}_dropout_twin_regularize dim-offset=0 dim={1}".format(name, output_backward['dimension']))
      component_nodes.append("dim-range-node name={0}_dropout_regularize input-node={0}_dropout_twin_regularize dim-offset={1} dim={1}".format(name, output_backward['dimension']))

      # component to connect twins using ephemeral connection
      if append_twins:
        final_output_dim = output_forward['dimension']+output_backward['dimension']
        components.append("component name={0}_sum_twins type=NoOpComponent dim={1}".format(name, final_output_dim))
        component_nodes.append("component-node name={0}_sum component={0}_sum_twins input=Append({1}, Sum({0}_dropout_twin, {0}_dropout_regularize))".format(name, output_forward['descriptor']))
      else:
        assert(output_forward['dimension'] == output_backward['dimension'])
        final_output_dim = output_forward['dimension']
        components.append("component name={0}_sum_twins type=NoOpComponent dim={1}".format(name, final_output_dim))
        component_nodes.append("component-node name={0}_sum component={0}_sum_twins input=Sum({1}, Sum({0}_dropout_twin, {0}_dropout_regularize))".format(name, output_forward['descriptor']))
    else:
      # Dropout for X_twin
      components.append("component name={0}_dropout type=DropoutComponent dim={1} dropout-proportion=0.0".format(name, output_backward['dimension']))
      component_nodes.append("component-node name={0}_dropout component={0}_dropout input={1}".format(name, output_backward['descriptor']))
      if append_twins:
        final_output_dim = output_forward['dimension']+output_backward['dimension']
        components.append("component name={0}_sum_twins type=NoOpComponent dim={1}".format(name, final_output_dim))
        component_nodes.append("component-node name={0}_sum component={0}_sum_twins input=Append({1}, {0}_dropout)".format(name, output_forward['descriptor']))
      else:
        assert(output_forward['dimension'] == output_backward['dimension'])
        final_output_dim = output_forward['dimension']
        components.append("component name={0}_sum_twins type=NoOpComponent dim={1}".format(name, final_output_dim))
        component_nodes.append("component-node name={0}_sum component={0}_sum_twins input=Sum({1}, {0}_dropout)".format(name, output_forward['descriptor']))
 
    output_descriptor = 'Append({0}, {1})'.format(output_forward['descriptor'], output_backward['descriptor'])
    output_dim = output_forward['dimension'] + output_backward['dimension']

    return [{
            'descriptor': '{0}_sum'.format(name),
            'dimension': final_output_dim,
            'regularizer':('{0}_scaled_regularizer'.format(name) if add_reg is True else ''),
            },
            { 
            'descriptor': output_descriptor,
            'dimension': output_dim
            }
            ]

def AddTwinBLstmLayer(config_lines,
                  name, forward_input, backward_input, cell_dim,
                  recurrent_projection_dim = 0,
                  non_recurrent_projection_dim = 0,
                  clipping_threshold = 1.0,
                  norm_based_clipping = "false",
                  ng_per_element_scale_options = "",
                  ng_affine_options = "",
                  lstm_delay = [-1,1],
                  self_repair_scale_nonlinearity = None,
                  self_repair_scale_clipgradient = None):
    assert(len(lstm_delay) == 2 and lstm_delay[0] < 0 and lstm_delay[1] > 0)
    output_forward = AddLstmLayer(config_lines, "{0}_forward".format(name), forward_input, cell_dim,
                                  recurrent_projection_dim, non_recurrent_projection_dim,
                                  clipping_threshold, norm_based_clipping,
                                  ng_per_element_scale_options, ng_affine_options,
                                  lstm_delay = lstm_delay[0],
                                  self_repair_scale_nonlinearity = self_repair_scale_nonlinearity,
                                  self_repair_scale_clipgradient = self_repair_scale_clipgradient)
    output_backward = AddLstmLayer(config_lines, "{0}_backward".format(name), backward_input, cell_dim,
                                   recurrent_projection_dim, non_recurrent_projection_dim,
                                   clipping_threshold, norm_based_clipping,
                                   ng_per_element_scale_options, ng_affine_options,
                                   lstm_delay = lstm_delay[1],
                                   self_repair_scale_nonlinearity = self_repair_scale_nonlinearity,
                                   self_repair_scale_clipgradient = self_repair_scale_clipgradient)
    # connect forward and backward using gate as Lstm_forward_rt + Dropout(g(Lstm_backward_rt) . Lstm_backward_rt)
    # and Lstm_backward_rt + DP(sum(H * Lstm_backward_rt + Lstm_forward_rt))

    assert(output_forward['dimension'] == output_backward['dimension'])

    forward_output_gate = AddSigmoidGate(config_lines, "{0}_forward".format(name), output_forward, self_repair_scale_nonlinearity = self_repair_scale_nonlinearity)

    backward_output_gate = AddSigmoidGate(config_lines, "{0}_backward".format(name), output_backward, self_repair_scale_nonlinearity = self_repair_scale_nonlinearity)

    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_forward_append_backward_gate type=NoOpComponent dim={1}".format(name, output_forward['dimension']))
    component_nodes.append("component-node name={0}_forward_append_backward_gate component={0}_forward_append_backward_gate input=Sum({1}, {2})".format(name, output_forward['descriptor'], backward_output_gate['descriptor']))
    

    components.append("component name={0}_backward_append_forward_gate type=NoOpComponent dim={1}".format(name, output_backward['dimension']))
    component_nodes.append("component-node name={0}_backward_append_forward_gate component={0}_backward_append_forward_gate input=Sum({1}, {2})".format(name, output_backward['descriptor'], forward_output_gate['descriptor']))

    return [{
            'descriptor': '{0}_forward_append_backward_gate'.format(name),
            'dimension': output_forward['dimension']
            },
            {'descriptor': '{0}_backward_append_forward_gate'.format(name),
             'dimension': output_backward['dimension']
            }
            ]

def AddRegularizer(config_lines, input, suffix=None, supervision_type= 'unsupervised', objective_type = 'quadratic'):
  components = config_lines['components']
  component_nodes = config_lines['component-nodes']
  name = 'regularize-output'
  if suffix is not None:
    name = '{0}-{1}'.format(name, suffix)
  component_nodes.append('output-node name={0} input={1} objective={2} supervision={3}'.format(name, input['regularizer'], objective_type, supervision_type))

# The input for main network needs to modified to be independent of twin part 
# at end of training and orphan nodes and components removed using --edit option from network
def RemoveRegularizer(config_lines, name, input):
   component_nodes = config_lines['component-nodes']
   component_nodes.append("component-node name={1} component={0}_sum_twins input=Sum({0}_renorm, {0}_regularize_renorm)".format(name, input['descriptor']))


