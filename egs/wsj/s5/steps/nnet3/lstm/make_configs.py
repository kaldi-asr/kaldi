#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
import sys
import warnings
import copy

# adds the input nodes and returns the descriptor
def AddInputLayer(config_lines, feat_dim, splice_indexes=[0], ivector_dim=0):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    output_dim = 0
    components.append('input-node name=input dim=' + str(feat_dim))
    list = [('Offset(input, {0})'.format(n) if n != 0 else 'input') for n in splice_indexes]
    output_dim += len(splice_indexes) * feat_dim
    if args.ivector_dim > 0:
        components.append('input-node name=ivector dim=' + str(ivector_dim))
        list.append('ReplaceIndex(ivector, t, 0)')
        output_dim += ivector_dim
    splice_descriptor = "Append({0})".format(", ".join(list))
    print(splice_descriptor)
    return {'descriptor': splice_descriptor,
            'dimension': output_dim}

def AddLdaLayer(config_lines, name, input, lda_file):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append('component name={0}_lda type=FixedAffineComponent matrix={1}'.format(name, lda_file))
    component_nodes.append('component-node name={0}_lda component={0}_lda input={1}'.format(name, input['descriptor']))

    return {'descriptor':  '{0}_lda'.format(name),
            'dimension': input['dimension']}

def AddAffineLayer(config_lines, name, input, output_dim, ng_affine_options = ""):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))

    return {'descriptor':  '{0}_affine'.format(name),
            'dimension': output_dim}

def AddAffRelNormLayer(config_lines, name, input, output_dim, ng_affine_options = ""):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1}".format(name, output_dim))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1}".format(name, output_dim))

    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_relu component={0}_relu input={0}_affine".format(name))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_relu".format(name))

    return {'descriptor':  '{0}_renorm'.format(name),
            'dimension': output_dim}



def AddSoftmaxLayer(config_lines, name, input):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_log_softmax type=LogSoftmaxComponent dim={1}".format(name, input['dimension']))
    component_nodes.append("component-node name={0}_log_softmax component={0}_log_softmax input={1}".format(name, input['descriptor']))

    return {'descriptor':  '{0}_log_softmax'.format(name),
            'dimension': input['dimension']}

def AddOutputNode(config_lines, input):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    component_nodes.append('output-node name=output input={0}'.format(input['descriptor']))


def AddOutputNodeWithDelay(config_lines, input, label_delay):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    component_nodes.append('output-node name=output input=Offset({0},{1})'.format(input['descriptor'], label_delay))

def AddFinalLayer(config_lines, input, output_dim, ng_affine_options = ""):
    prev_layer_output = AddAffineLayer(config_lines, "Final", input, output_dim, ng_affine_options)
    prev_layer_output = AddSoftmaxLayer(config_lines, "Final", prev_layer_output)
    AddOutputNodeWithDelay(config_lines, prev_layer_output, args.label_delay)


def AddLstmLayer(config_lines,
                 name, input, cell_dim,
                 recurrent_projection_dim = 0,
                 non_recurrent_projection_dim = 0,
                 clipping_threshold = 1.0,
                 norm_based_clipping = "false",
                 ng_per_element_scale_options = "",
                 ng_affine_options = ""):
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
    components.append("component name={0}_i type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_f type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_o type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_g type=TanhComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_h type=TanhComponent dim={1}".format(name, cell_dim))

    components.append("# Defining the cell computations")
    components.append("component name={0}_c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_m type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, cell_dim, clipping_threshold, norm_based_clipping))

    # c1_t and c2_t defined below
    component_nodes.append("component-node name={0}_c_t component={0}_c input=Sum({0}_c1_t, {0}_c2_t)".format(name))
    c_tminus1_descriptor = "IfDefined(Offset({0}_c_t, -1))".format(name)

    component_nodes.append("# i_t")
    component_nodes.append("component-node name={0}_i1 component={0}_W_i-xr input=Append({1}, IfDefined(Offset({0}_{2}, -1)))".format(name, input_descriptor, recurrent_connection))
    component_nodes.append("component-node name={0}_i2 component={0}_w_ic  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_i_t component={0}_i input=Sum({0}_i1, {0}_i2)".format(name))

    component_nodes.append("# f_t")
    component_nodes.append("component-node name={0}_f1 component={0}_W_f-xr input=Append({1}, IfDefined(Offset({0}_{2}, -1)))".format(name, input_descriptor, recurrent_connection))
    component_nodes.append("component-node name={0}_f2 component={0}_w_fc  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_f_t component={0}_f input=Sum({0}_f1,{0}_f2)".format(name))

    component_nodes.append("# o_t")
    component_nodes.append("component-node name={0}_o1 component={0}_W_o-xr input=Append({1}, IfDefined(Offset({0}_{2}, -1)))".format(name, input_descriptor, recurrent_connection))
    component_nodes.append("component-node name={0}_o2 component={0}_w_oc input={0}_c_t".format(name))
    component_nodes.append("component-node name={0}_o_t component={0}_o input=Sum({0}_o1, {0}_o2)".format(name))

    component_nodes.append("# h_t")
    component_nodes.append("component-node name={0}_h_t component={0}_h input={0}_c_t".format(name))

    component_nodes.append("# g_t")
    component_nodes.append("component-node name={0}_g1 component={0}_W_c-xr input=Append({1}, IfDefined(Offset({0}_{2}, -1)))".format(name, input_descriptor, recurrent_connection))
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
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("# r_t and p_t")
        component_nodes.append("component-node name={0}_rp_t component={0}_W-m input={0}_m_t".format(name))
        component_nodes.append("dim-range-node name={0}_r_t_preclip input-node={0}_rp_t dim-offset=0 dim={1}".format(name, recurrent_projection_dim))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_rp_t'.format(name)
        output_dim = recurrent_projection_dim + non_recurrent_projection_dim

    elif add_recurrent_projection:
        components.append("# projection matrices : Wrm")
        components.append("component name={0}_Wrm type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim, ng_affine_options))
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("# r_t")
        component_nodes.append("component-node name={0}_r_t_preclip component={0}_Wrm input={0}_m_t".format(name))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = recurrent_projection_dim

    else:
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, cell_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_m_t".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = cell_dim

    return {
            'descriptor': output_descriptor,
            'dimension':output_dim
            }

def PrintConfig(file_name, config_lines):
    f = open(file_name, 'w')
    f.write("\n".join(config_lines['components'])+"\n")
    f.write("\n#Component nodes\n")
    f.write("\n".join(config_lines['component-nodes']))
    f.close()

def ParseSpliceString(splice_indexes, label_delay):
    ## Work out splice_array e.g. splice_array = [ [ -3,-2,...3 ], [0], [-2,2], .. [ -8,8 ] ]
    split1 = splice_indexes.split(" ");  # we already checked the string is nonempty.
    if len(split1) < 1:
        splice_indexes = "0"

    left_context = -label_delay
    right_context = label_delay
    splice_array = []
    try:
        for i in range(len(split1)):
            indexes = map(lambda x: int(x), split1[i].strip().split(","))
            print(indexes)
            if len(indexes) < 1:
                raise ValueError("invalid --splice-indexes argument, too-short element: "
                                + splice_indexes)

            if (i > 0)  and ((len(indexes) != 1) or (indexes[0] != 0)):
                raise ValueError("elements of --splice-indexes splicing is only allowed initial layer.")

            if not indexes == sorted(indexes):
                raise ValueError("elements of --splice-indexes must be sorted: "
                                + splice_indexes)
            left_context += -indexes[0]
            right_context += indexes[-1]
            splice_array.append(indexes)
    except ValueError as e:
        raise ValueError("invalid --splice-indexes argument " + splice_indexes + str(e))

    left_context = max(0, left_context)
    right_context = max(0, right_context)

    return {'left_context':left_context,
            'right_context':right_context,
            'splice_indexes':splice_array,
            'num_hidden_layers':len(splice_array)
            }

if __name__ == "__main__":
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="Writes config files and variables "
                                                 "for LSTMs creation and training",
                                     epilog="See steps/nnet3/lstm/train.sh for example.")
    # General neural network options
    parser.add_argument("--splice-indexes", type=str,
                        help="Splice indexes at input layer, e.g. '-3,-2,-1,0,1,2,3' [compulsary argument]", default="0")
    parser.add_argument("--feat-dim", type=int,
                        help="Raw feature dimension, e.g. 13")
    parser.add_argument("--ivector-dim", type=int,
                        help="iVector dimension, e.g. 100", default=0)

    # LSTM options
    parser.add_argument("--num-lstm-layers", type=int,
                        help="Number of LSTM layers to be stacked", default=1)
    parser.add_argument("--cell-dim", type=int,
                        help="dimension of lstm-cell")
    parser.add_argument("--recurrent-projection-dim", type=int,
                        help="dimension of recurrent projection")
    parser.add_argument("--non-recurrent-projection-dim", type=int,
                        help="dimension of non-recurrent projection")
    parser.add_argument("--hidden-dim", type=int,
                        help="dimension of fully-connected layers")

    # Natural gradient options
    parser.add_argument("--ng-per-element-scale-options", type=str,
                        help="options to be supplied to NaturalGradientPerElementScaleComponent", default="")
    parser.add_argument("--ng-affine-options", type=str,
                        help="options to be supplied to NaturalGradientAffineComponent", default="")

    # Gradient clipper options
    parser.add_argument("--norm-based-clipping", type=str,
                        help="use norm based clipping in ClipGradient components ", default="false", choices = ["false", "true"])
    parser.add_argument("--clipping-threshold", type=float,
                        help="clipping threshold used in ClipGradient components, if clipping-threshold=0 no clipping is done", default=15)

    parser.add_argument("--num-targets", type=int,
                        help="number of network targets (e.g. num-pdf-ids/num-leaves)")
    parser.add_argument("config_dir",
                        help="Directory to write config files and variables")

    # Delay options
    parser.add_argument("--label-delay", type=int,
                        help="option to delay the labels to make the lstm robust")


    print(' '.join(sys.argv))

    args = parser.parse_args()

    if not os.path.exists(args.config_dir):
        os.makedirs(args.config_dir)

    ## Check arguments.
    if args.splice_indexes is None:
        sys.exit("--splice-indexes argument is required")
    if args.feat_dim is None or not (args.feat_dim > 0):
        sys.exit("--feat-dim argument is required")
    if args.num_targets is None or not (args.num_targets > 0):
        sys.exit("--feat-dim argument is required")
    if (args.num_lstm_layers < 1):
        sys.exit("--num-lstm-layers has to be a positive integer")
    if (args.clipping_threshold < 0):
        sys.exit("--clipping-threshold has to be a non-negative")



    parsed_splice_output = ParseSpliceString(args.splice_indexes.strip(),args.label_delay)
    left_context = parsed_splice_output['left_context']
    right_context = parsed_splice_output['right_context']
    num_hidden_layers = parsed_splice_output['num_hidden_layers']
    splice_indexes = parsed_splice_output['splice_indexes']

    if (num_hidden_layers < args.num_lstm_layers):
        sys.exit("--num-lstm-layers : number of lstm layers has to be greater than number of layers, decided based on splice-indexes")

    # write the files used by other scripts like steps/nnet3/get_egs.sh
    f = open(args.config_dir + "/vars", "w")
    print('model_left_context=' + str(left_context), file=f)
    print('model_right_context=' + str(right_context), file=f)
    print('num_hidden_layers=' + str(num_hidden_layers), file=f)
    # print('initial_right_context=' + str(splice_array[0][-1]), file=f)
    f.close()

    config_lines = {'components':[], 'component-nodes':[]}

    config_files={}
    prev_layer_output = AddInputLayer(config_lines, args.feat_dim, splice_indexes[0], args.ivector_dim)

    # Add the init config lines for estimating the preconditioning matrices
    init_config_lines = copy.deepcopy(config_lines)
    init_config_lines['components'].insert(0, '# Config file for initializing neural network prior to')
    init_config_lines['components'].insert(0, '# preconditioning matrix computation')
    AddOutputNode(init_config_lines, prev_layer_output)
    config_files[args.config_dir + '/init.config'] = init_config_lines

    prev_layer_output = AddLdaLayer(config_lines, "L0", prev_layer_output, args.config_dir + '/lda.mat')

    for i in range(args.num_lstm_layers):
        prev_layer_output = AddLstmLayer(config_lines, "Lstm{0}".format(i+1), prev_layer_output, args.cell_dim,
                                         args.recurrent_projection_dim, args.non_recurrent_projection_dim,
                                         args.clipping_threshold, args.norm_based_clipping,
                                         args.ng_per_element_scale_options, args.ng_affine_options)
        # make the intermediate config file for layerwise discriminative
        # training
        AddFinalLayer(config_lines, prev_layer_output, args.num_targets, args.ng_affine_options)
        config_files['{0}/layer{1}.config'.format(args.config_dir, i+1)] = config_lines
        config_lines = {'components':[], 'component-nodes':[]}

    for i in range(args.num_lstm_layers, num_hidden_layers):
        prev_layer_output = AddAffRelNormLayer(config_lines, "L{0}".format(i+1),
                                               prev_layer_output, args.hidden_dim,
                                               args.ng_affine_options)
        # make the intermediate config file for layerwise discriminative
        # training
        AddFinalLayer(config_lines, prev_layer_output, args.num_targets, args.ng_affine_options)
        config_files['{0}/layer{1}.config'.format(args.config_dir, i+1)] = config_lines
        config_lines = {'components':[], 'component-nodes':[]}

    # printing out the configs
    # init.config used to train lda-mllt train
    for key in config_files.keys():
        PrintConfig(key, config_files[key])
