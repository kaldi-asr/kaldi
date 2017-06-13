#!/usr/bin/env python

# Copyright      2015  Johns Hopkins University (Author: Vijayaditya Peddinti)
# Apache 2.0

# script to convert nnet3-am-info output to a dot graph


# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re
import os
import argparse
import sys
import math
import warnings
import descriptor_parser
import pprint

node_attributes = {
    'input-node':{
        'shape':'oval'
    },
    'output-node':{
        'shape':'oval'
    },
    'NaturalGradientAffineComponent':{
        'color':'lightgrey',
        'shape':'box',
        'style':'filled'
    },
    'NaturalGradientPerElementScaleComponent':{
        'color':'lightpink',
        'shape':'box',
        'style':'filled'
    },
    'ConvolutionComponent':{
        'color':'lightpink',
        'shape':'box',
        'style':'filled'
    },
    'FixedScaleComponent':{
        'color':'blueviolet',
        'shape':'box',
        'style':'filled'
    },
    'FixedAffineComponent':{
        'color':'darkolivegreen1',
        'shape':'box',
        'style':'filled'
    },
    'SigmoidComponent':{
        'color':'bisque',
        'shape':'rectangle',
        'style':'filled'
    },
    'TanhComponent':{
        'color':'bisque',
        'shape':'rectangle',
        'style':'filled'
    },
    'NormalizeComponent':{
        'color':'aquamarine',
        'shape':'rectangle',
        'style':'filled'
    },
    'RectifiedLinearComponent':{
        'color':'bisque',
        'shape':'rectangle',
        'style':'filled'
    },
    'ClipGradientComponent':{
        'color':'bisque',
        'shape':'rectangle',
        'style':'filled'
    },
    'ElementwiseProductComponent':{
        'color':'green',
        'shape':'rectangle',
        'style':'filled'
    },
    'LogSoftmaxComponent':{
        'color':'cyan',
        'shape':'rectangle',
        'style':'filled'
    }
}

def GetDotNodeName(name_string, is_component = False):
    # this function is required as dot does not allow all the component names
    # allowed by nnet3.
    # Identified incompatibilities :
    #   1. dot does not allow hyphen(-) and dot(.) in names
    #   2. Nnet3 names can be shared among components and component nodes
    #      dot does not allow common names
    #
    node_name_string = re.sub("-", "hyphen", name_string)
    node_name_string = re.sub("\.", "_dot_", node_name_string)
    if is_component:
        node_name_string += node_name_string.strip() + "_component"
    return {"label":name_string, "node":node_name_string}

def ProcessAppendDescriptor(segment, parent_node_name, affix, edge_attributes = None):
    dot_graph = []
    names = []
    desc_name = 'Append_{0}'.format(affix)
    for i in range(len(segment['sub_segments'])):
        sub_segment = segment['sub_segments'][i]
        part_name = "{0}{1}{2}".format(desc_name, sub_segment['name'], i)
        names.append("<{0}> part {1}".format(GetDotNodeName(part_name)['node'], i))
        dot_graph += DescriptorSegmentToDot(sub_segment, "{0}:{1}".format(desc_name, part_name), desc_name)

    part_index = len(segment['sub_segments'])
    for i in range(len(segment['arguments'])):
        part_name = "{0}{1}{2}".format(desc_name, segment['arguments'][i], part_index + i)
        names.append("<{0}> part {1}".format(GetDotNodeName(part_name)['node'], part_index + i))
        dot_graph.append("{0} -> {1}:{2}".format(GetDotNodeName(segment['arguments'][i])['node'], GetDotNodeName(desc_name)['node'], GetDotNodeName(part_name)['node']))

    label = "|".join(names)
    label = "{{"+label+"}|Append}"
    dot_graph.append('{0} [shape=Mrecord, label="{1}"];'.format(GetDotNodeName(desc_name)['node'], label))

    attr_string = ''
    if edge_attributes is not None:
        if edge_attributes.has_key('label'):
            attr_string += " label={0} ".format(edge_attributes['label'])
        if edge_attributes.has_key('style'):
            attr_string += ' style={0} '.format(edge_attributes['style'])

    dot_string = '{0} -> {1} [tailport=s]'.format(GetDotNodeName(desc_name)['node'], GetDotNodeName(parent_node_name)['node'])

    if attr_string != '':
        dot_string += ' [{0}] '.format(attr_string)
    dot_graph.append(dot_string)


    return dot_graph

def ProcessRoundDescriptor(segment, parent_node_name, affix, edge_attributes = None):
    dot_graph = []

    label = 'Round ({0})'.format(segment['arguments'][1])
    style = None
    if edge_attributes is not None:
        if edge_attributes.has_key('label'):
            label = "{0} {1}".format(edge_attributes['label'], label)
        if edge_attributes.has_key('style'):
            style  = 'style={0}'.format(edge_attributes['style'])

    attr_string = 'label="{0}"'.format(label)
    if style is not None:
        attr_string += ' {0}'.format(style)
    dot_graph.append('{0}->{1} [ {2} ]'.format(GetDotNodeName(segment['arguments'][0])['node'],
                                                                    GetDotNodeName(parent_node_name)['node'],
                                                                    attr_string))
    if segment['sub_segments']:
        raise Exception("Round can just deal with forwarding descriptor, no sub-segments allowed")
    return dot_graph


def ProcessOffsetDescriptor(segment, parent_node_name, affix, edge_attributes = None):
    dot_graph = []

    label = 'Offset ({0})'.format(segment['arguments'][1])
    style = None
    if edge_attributes is not None:
        if edge_attributes.has_key('label'):
            label = "{0} {1}".format(edge_attributes['label'], label)
        if edge_attributes.has_key('style'):
            style  = 'style={0}'.format(edge_attributes['style'])

    attr_string = 'label="{0}"'.format(label)
    if style is not None:
        attr_string += ' {0}'.format(style)

    dot_graph.append('{0}->{1} [ {2} ]'.format(GetDotNodeName(segment['arguments'][0])['node'],
                                                                    GetDotNodeName(parent_node_name)['node'],
                                                                    attr_string))
    if segment['sub_segments']:
        raise Exception("Offset can just deal with forwarding descriptor, no sub-segments allowed")
    return dot_graph

def ProcessSumDescriptor(segment, parent_node_name, affix, edge_attributes = None):
    dot_graph = []
    names = []
    desc_name = 'Sum_{0}'.format(affix)
    # create the sum node
    for i in range(len(segment['sub_segments'])):
        sub_segment = segment['sub_segments'][i]
        part_name = "{0}{1}{2}".format(desc_name, sub_segment['name'], i)
        names.append("<{0}> part {1}".format(GetDotNodeName(part_name)['node'], i))
        dot_graph += DescriptorSegmentToDot(sub_segment, "{0}:{1}".format(desc_name, part_name), desc_name+"_"+str(i))

    # link the sum node parts to corresponding segments
    part_index = len(segment['sub_segments'])
    for i in range(len(segment['arguments'])):
        part_name = "{0}{1}{2}".format(desc_name, segment['arguments'][i], part_index + i)
        names.append("<{0}> part {1}".format(GetDotNodeName(part_name)['node'], part_index + i))
        dot_graph.append("{0} -> {1}:{2}".format(GetDotNodeName(segment['arguments'][i])['node'], GetDotNodeName(desc_name)['node'], GetDotNodeName(part_name)['node']))

    label = "|".join(names)
    label = '{{'+label+'}|Sum}'
    dot_graph.append('{0} [shape=Mrecord, label="{1}", color=red];'.format(GetDotNodeName(desc_name)['node'], label))

    attr_string = ''
    if edge_attributes is not None:
        if edge_attributes.has_key('label'):
            attr_string += " label={0} ".format(edge_attributes['label'])
        if edge_attributes.has_key('style'):
            attr_string += ' style={0} '.format(edge_attributes['style'])

    dot_string = '{0} -> {1}'.format(GetDotNodeName(desc_name)['node'], GetDotNodeName(parent_node_name)['node'])

    dot_string += ' [{0} tailport=s ] '.format(attr_string)
    dot_graph.append(dot_string)
    return dot_graph

def ProcessReplaceIndexDescriptor(segment, parent_node_name, affix, edge_attributes = None):
    dot_graph = []

    label = 'ReplaceIndex({0}, {1})'.format(segment['arguments'][1], segment['arguments'][2])
    style = None
    if edge_attributes is not None:
        if edge_attributes.has_key('label'):
            label = "{0} {1}".format(edge_attributes['label'], label)
        if edge_attributes.has_key('style'):
            style  = 'style={0}'.format(edge_attributes['style'])

    attr_string = 'label="{0}"'.format(label)
    if style is not None:
        attr_string += ' {0}'.format(style)

    dot_graph.append('{0}->{1} [{2}]'.format(GetDotNodeName(segment['arguments'][0])['node'],
                                                                    GetDotNodeName(parent_node_name)['node'],
                                                                    attr_string))
    if segment['sub_segments']:
        raise Exception("ReplaceIndex can just deal with forwarding descriptor, no sub-segments allowed")
    return dot_graph

def ProcessIfDefinedDescriptor(segment, parent_node_name, affix, edge_attributes = None):
    # IfDefined adds attributes to the edges
    if edge_attributes is not None:
        raise Exception("edge_attributes was not None, this means an IfDefined descriptor was calling the current IfDefined descriptor. This is not allowed")
    dot_graph = []
    dot_graph.append('#ProcessIfDefinedDescriptor')
    names = []

    if segment['sub_segments']:
        sub_segment = segment['sub_segments'][0]
        dot_graph += DescriptorSegmentToDot(sub_segment, parent_node_name, parent_node_name, edge_attributes={'style':'dotted', 'label':'IfDefined'})

    if segment['arguments']:
        dot_graph.append('{0} -> {1} [style=dotted, label="IfDefined"]'.format(GetDotNodeName(segment['arguments'][0])['node'], GetDotNodeName(parent_node_name)['node']))

    return dot_graph

def DescriptorSegmentToDot(segment, parent_node_name, affix, edge_attributes = None):
    # segment is a dicionary which corresponds to a descriptor
    dot_graph = []
    if segment['name'] == "Append":
        dot_graph += ProcessAppendDescriptor(segment, parent_node_name, affix, edge_attributes)
    elif segment['name'] == "Offset":
        dot_graph += ProcessOffsetDescriptor(segment, parent_node_name, affix, edge_attributes)
    elif segment['name'] == "Sum":
        dot_graph += ProcessSumDescriptor(segment, parent_node_name, affix, edge_attributes)
    elif segment['name'] == "IfDefined":
        dot_graph += ProcessIfDefinedDescriptor(segment, parent_node_name, affix, edge_attributes)
    elif segment['name'] == "ReplaceIndex":
        dot_graph += ProcessReplaceIndexDescriptor(segment, parent_node_name, affix, edge_attributes)
    elif segment['name'] == "Round":
        dot_graph += ProcessRoundDescriptor(segment, parent_node_name, affix, edge_attributes)
    else:
        raise Exception('Descriptor {0}, is not recognized by this script. Please add Process{0}Descriptor method'.format(segment['name']))
    return dot_graph

def Nnet3DescriptorToDot(descriptor, parent_node_name):
    dot_lines = []
    [segments, arguments] = descriptor_parser.IdentifyNestedSegments(descriptor)
    if segments:
        for segment in segments:
            dot_lines += DescriptorSegmentToDot(segment, parent_node_name, parent_node_name)
    elif arguments:
        assert(len(arguments) == 1)
        dot_lines.append("{0} -> {1}".format(GetDotNodeName(arguments[0])['node'], GetDotNodeName(parent_node_name)['node']))
    return dot_lines

def ParseNnet3String(string):
    if re.search('^input-node|^component|^output-node|^component-node|^dim-range-node', string.strip()) is None:
        return [None, None]

    parts = string.split()
    config_type = parts[0]
    fields = []
    prev_field = ''
    for i in range(1, len(parts)):
        if re.search('=', parts[i]) is None:
            prev_field += ' '+parts[i]
        else:
            if not (prev_field.strip() == ''):
                fields.append(prev_field)
            sub_parts = parts[i].split('=')
            if (len(sub_parts) != 2):
                raise Exception('Malformed config line {0}'.format(string))
            fields.append(sub_parts[0])
            prev_field = sub_parts[1]
    fields.append(prev_field)

    parsed_string = {}
    try:
        while len(fields) > 0:
            value = re.sub(',$', '', fields.pop().strip())
            key = fields.pop()
            parsed_string[key.strip()] = value.strip()
    except IndexError:
        raise Exception('Malformed config line {0}'.format(string))
    return [config_type, parsed_string]

# sample component config line
# component name=L0_lda type=FixedAffineComponent, input-dim=300, output-dim=300, linear-params-stddev=0.00992724, bias-params-stddev=0.573973
def Nnet3ComponentToDot(component_config, component_attributes = None):
    label = ''
    if component_attributes is None:
        component_attributes = component_config.keys()
    attributes_to_print = set(component_attributes).intersection(component_config.keys())
    # process the known fields
    for key in attributes_to_print:
        if component_config.has_key(key):
            label += '{0} = {1}\\n'.format(key, component_config[key])

    attr_string = ''
    try:
        attributes = node_attributes[component_config['type']]
        for key in attributes.keys():
            attr_string += ' {0}={1} '.format(key, attributes[key])
    except KeyError:
        pass

    return ['{0} [label="{1}" {2}]'.format(GetDotNodeName(component_config['name'], is_component = True)['node'], label, attr_string)]


# input-node name=input dim=40
def Nnet3InputToDot(parsed_config):
    return ['{0} [ label="{1}\\ndim={2}"]'.format(GetDotNodeName(parsed_config['name'])['node'], parsed_config['name'], parsed_config['dim'] )]

# output-node name=output input=Final_log_softmax dim=3940 objective=linear
#output-node name=output input=Offset(Final_log_softmax, 5) dim=3940 objective=linear
def Nnet3OutputToDot(parsed_config):
    dot_graph = []
    dot_graph += Nnet3DescriptorToDot(parsed_config['input'], parsed_config['name'])
    dot_graph.append('{0} [ label="{1}\\nobjective={2}"]'.format(GetDotNodeName(parsed_config['name'])['node'], parsed_config['name'], parsed_config['objective']))
    return dot_graph

# dim-range-node name=Lstm1_r_t input-node=Lstm1_rp_t dim-offset=0 dim=256
def Nnet3DimrangeToDot(parsed_config):
    dot_graph = []
    dot_node = GetDotNodeName(parsed_config['name'])
    dot_graph.append('{0} [shape=rectangle, label="{1}"]'.format(dot_node['node'], dot_node['label']))
    dot_graph.append('{0} -> {1} [taillabel="dimrange({2}, {3})"]'.format(GetDotNodeName(parsed_config['input-node'])['node'],
                                                           GetDotNodeName(parsed_config['name'])['node'],
                                                           parsed_config['dim-offset'],
                                                           parsed_config['dim']))
    return dot_graph

def Nnet3ComponentNodeToDot(parsed_config):
    dot_graph = []
    dot_graph += Nnet3DescriptorToDot(parsed_config['input'], parsed_config['name'])
    dot_node = GetDotNodeName(parsed_config['name'])
    dot_graph.append('{0} [ label="{1}", shape=box ]'.format(dot_node['node'], dot_node['label']))
    dot_graph.append('{0} -> {1} [ weight=10 ]'.format(GetDotNodeName(parsed_config['component'], is_component = True)['node'],
                                                       GetDotNodeName(parsed_config['name'])['node']))
    return dot_graph

def GroupConfigs(configs, node_prefixes = []):
    # we make the assumption that nodes belonging to the same sub-graph have a
    # commong prefix.
    grouped_configs = {}
    for node_prefix in node_prefixes:
        group = []
        rest = []
        for config in configs:
            if re.search('^{0}'.format(node_prefix), config[1]['name']) is not None:
                group.append(config)
            else:
                rest.append(config)
        configs = rest
        grouped_configs[node_prefix] = group
    grouped_configs[None] = configs

    return grouped_configs

def ParseConfigLines(lines, node_prefixes = [], component_attributes = None ):
    config_lines = []
    dot_graph=[]
    configs = []
    for line in lines:
        config_type, parsed_config = ParseNnet3String(line)
        if config_type is not None:
            configs.append([config_type, parsed_config])

    # process the config lines
    grouped_configs = GroupConfigs(configs, node_prefixes)
    for group in grouped_configs.keys():
        configs = grouped_configs[group]
        if not configs:
            continue
        if group is not None:
            # subgraphs prefixed with cluster will be treated differently by
            # dot
            dot_graph.append('subgraph cluster_{0} '.format(group) + "{")
            dot_graph.append('color=blue')

        for config in configs:
            config_type = config[0]
            parsed_config = config[1]
            if config_type is None:
                continue
            if config_type == 'input-node':
                dot_graph += Nnet3InputToDot(parsed_config)
            elif config_type == 'output-node':
                dot_graph += Nnet3OutputToDot(parsed_config)
            elif config_type == 'component-node':
                dot_graph += Nnet3ComponentNodeToDot(parsed_config)
            elif config_type == 'dim-range-node':
                dot_graph += Nnet3DimrangeToDot(parsed_config)
            elif config_type == 'component':
                dot_graph += Nnet3ComponentToDot(parsed_config, component_attributes)

        if group is not None:
            dot_graph.append('label = "{0}"'.format(group))
            dot_graph.append('}')

    dot_graph.insert(0, 'digraph nnet3graph {')
    dot_graph.append('}')

    return dot_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts the output of nnet3-am-info "
                                                 "to dot graph. The output has to be compiled"
                                                 " with dot to generate a displayable graph",
                                    epilog="See steps/nnet3/nnet3_to_dot.sh for example.");
    parser.add_argument("--component-attributes", type=str,
                        help="Attributes of the components which should be displayed in the dot-graph "
                             "e.g. --component-attributes name,type,input-dim,output-dim", default=None)
    parser.add_argument("--node-prefixes", type=str,
                        help="list of prefixes. Nnet3 components/component-nodes with the same prefix"
                        " will be clustered together in the dot-graph"
                        " --node-prefixes Lstm1,Lstm2,Layer1", default=None)

    parser.add_argument("dotfile", help="name of the dot output file")

    print(' '.join(sys.argv), file=sys.stderr)

    args = parser.parse_args()
    component_attributes = None
    if args.component_attributes is not None:
        component_attributes = args.component_attributes.split(',')
    node_prefixes = []
    if args.node_prefixes is not None:
        node_prefixes = args.node_prefixes.split(',')

    lines = sys.stdin.readlines()
    dot_graph = ParseConfigLines(lines, component_attributes = component_attributes, node_prefixes = node_prefixes)

    dotfile_handle = open(args.dotfile, "w")
    dotfile_handle.write("\n".join(dot_graph))
    dotfile_handle.close()
