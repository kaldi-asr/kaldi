#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import os
import argparse
import shlex
import sys
import warnings
import copy
import imp
import ast

nodes = imp.load_source('', 'steps/nnet3/components.py')
sys.path.insert(0, 'steps')
import libs.common as common_lib

def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="Writes config files and variables "
                                                 "for TDNNs creation and training",
                                     epilog="See steps/nnet3/tdnn/train.sh for example.")

    # Only one of these arguments can be specified, and one of them has to
    # be compulsarily specified
    feat_group = parser.add_mutually_exclusive_group(required = True)
    feat_group.add_argument("--feat-dim", type=int,
                            help="Raw feature dimension, e.g. 13")
    feat_group.add_argument("--feat-dir", type=str,
                            help="Feature directory, from which we derive the feat-dim")

    # only one of these arguments can be specified
    ivector_group = parser.add_mutually_exclusive_group(required = False)
    ivector_group.add_argument("--ivector-dim", type=int,
                                help="iVector dimension, e.g. 100", default=0)
    ivector_group.add_argument("--ivector-dir", type=str,
                                help="iVector dir, which will be used to derive the ivector-dim  ", default=None)

    num_target_group = parser.add_mutually_exclusive_group(required = True)
    num_target_group.add_argument("--num-targets", type=int,
                                  help="number of network targets (e.g. num-pdf-ids/num-leaves)")
    num_target_group.add_argument("--ali-dir", type=str,
                                  help="alignment directory, from which we derive the num-targets")
    num_target_group.add_argument("--tree-dir", type=str,
                                  help="directory with final.mdl, from which we derive the num-targets")

    # CNN options
    parser.add_argument('--cnn.layer', type=str, action='append', dest = "cnn_layer",
                        help="CNN parameters at each CNN layer, e.g. --filt-x-dim=3 --filt-y-dim=8 "
                        "--filt-x-step=1 --filt-y-step=1 --num-filters=256 --pool-x-size=1 --pool-y-size=3 "
                        "--pool-z-size=1 --pool-x-step=1 --pool-y-step=3 --pool-z-step=1, "
                        "when CNN layers are used, no LDA will be added", default = None)
    parser.add_argument("--cnn.bottleneck-dim", type=int, dest = "cnn_bottleneck_dim",
                        help="Output dimension of the linear layer at the CNN output "
                        "for dimension reduction, e.g. 256."
                        "The default zero means this layer is not needed.", default=0)
    parser.add_argument("--cnn.cepstral-lifter", type=float, dest = "cepstral_lifter",
                        help="The factor used for determining the liftering vector in the production of MFCC. "
                        "User has to ensure that it matches the lifter used in MFCC generation, "
                        "e.g. 22.0", default=22.0)

    # General neural network options
    parser.add_argument("--splice-indexes", type=str, required = True,
                        help="Splice indexes at each layer, e.g. '-3,-2,-1,0,1,2,3' "
                        "If CNN layers are used the first set of splice indexes will be used as input "
                        "to the first CNN layer and later splice indexes will be interpreted as indexes "
                        "for the TDNNs.")
    parser.add_argument("--add-lda", type=str, action=common_lib.StrToBoolAction,
                        help="If \"true\" an LDA matrix computed from the input features "
                        "(spliced according to the first set of splice-indexes) will be used as "
                        "the first Affine layer. This affine layer's parameters are fixed during training. "
                        "If --cnn.layer is specified this option will be forced to \"false\".",
                        default=True, choices = ["false", "true"])

    parser.add_argument("--include-log-softmax", type=str, action=common_lib.StrToBoolAction,
                        help="add the final softmax layer ", default=True, choices = ["false", "true"])
    parser.add_argument("--add-final-sigmoid", type=str, action=common_lib.StrToBoolAction,
                        help="add a final sigmoid layer as alternate to log-softmax-layer. "
                        "Can only be used if include-log-softmax is false. "
                        "This is useful in cases where you want the output to be "
                        "like probabilities between 0 and 1. Typically the nnet "
                        "is trained with an objective such as quadratic",
                        default=False, choices = ["false", "true"])

    parser.add_argument("--objective-type", type=str,
                        help = "the type of objective; i.e. quadratic or linear",
                        default="linear", choices = ["linear", "quadratic"])
    parser.add_argument("--xent-regularize", type=float,
                        help="For chain models, if nonzero, add a separate output for cross-entropy "
                        "regularization (with learning-rate-factor equal to the inverse of this)",
                        default=0.0)
    parser.add_argument("--xent-separate-forward-affine", type=str, action=common_lib.StrToBoolAction,
                        help="if using --xent-regularize, gives it separate last-but-one weight matrix",
                        default=False, choices = ["false", "true"])
    parser.add_argument("--final-layer-normalize-target", type=float,
                        help="RMS target for final layer (set to <1 if final layer learns too fast",
                        default=1.0)
    parser.add_argument("--max-change-per-component", type=float,
                        help="Enforces per-component max change (except for the final affine layer). "
                        "if 0 it would not be enforced.", default=0.75)
    parser.add_argument("--max-change-per-component-final", type=float,
                        help="Enforces per-component max change for the final affine layer. "
                        "if 0 it would not be enforced.", default=1.5)
    parser.add_argument("--subset-dim", type=int, default=0,
                        help="dimension of the subset of units to be sent to the central frame")
    parser.add_argument("--pnorm-input-dim", type=int,
                        help="input dimension to p-norm nonlinearities")
    parser.add_argument("--pnorm-output-dim", type=int,
                        help="output dimension of p-norm nonlinearities")
    relu_dim_group = parser.add_mutually_exclusive_group(required = False)
    relu_dim_group.add_argument("--relu-dim", type=int,
                        help="dimension of all ReLU nonlinearity layers")
    relu_dim_group.add_argument("--relu-dim-final", type=int,
                        help="dimension of the last ReLU nonlinearity layer. Dimensions increase geometrically from the first through the last ReLU layer.", default=None)
    parser.add_argument("--relu-dim-init", type=int,
                        help="dimension of the first ReLU nonlinearity layer. Dimensions increase geometrically from the first through the last ReLU layer.", default=None)

    parser.add_argument("--self-repair-scale-nonlinearity", type=float,
                        help="A non-zero value activates the self-repair mechanism in the sigmoid and tanh non-linearities of the LSTM", default=None)


    parser.add_argument("--use-presoftmax-prior-scale", type=str, action=common_lib.StrToBoolAction,
                        help="if true, a presoftmax-prior-scale is added",
                        choices=['true', 'false'], default = True)
    parser.add_argument("config_dir",
                        help="Directory to write config files and variables")

    print(' '.join(sys.argv))

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if not os.path.exists(args.config_dir):
        os.makedirs(args.config_dir)

    ## Check arguments.
    if args.feat_dir is not None:
        args.feat_dim = common_lib.get_feat_dim(args.feat_dir)

    if args.ali_dir is not None:
        args.num_targets = common_lib.get_number_of_leaves_from_tree(args.ali_dir)
    elif args.tree_dir is not None:
        args.num_targets = common_lib.get_number_of_leaves_from_tree(args.tree_dir)

    if args.ivector_dir is not None:
        args.ivector_dim = common_lib.get_ivector_dim(args.ivector_dir)

    if not args.feat_dim > 0:
        raise Exception("feat-dim has to be postive")

    if not args.num_targets > 0:
        print(args.num_targets)
        raise Exception("num_targets has to be positive")

    if not args.ivector_dim >= 0:
        raise Exception("ivector-dim has to be non-negative")

    if (args.subset_dim < 0):
        raise Exception("--subset-dim has to be non-negative")

    if not args.relu_dim is None:
        if not args.pnorm_input_dim is None or not args.pnorm_output_dim is None or not args.relu_dim_init is None:
            raise Exception("--relu-dim argument not compatible with "
                            "--pnorm-input-dim or --pnorm-output-dim or --relu-dim-init options");
        args.nonlin_input_dim = args.relu_dim
        args.nonlin_output_dim = args.relu_dim
        args.nonlin_output_dim_final = None
        args.nonlin_output_dim_init = None
        args.nonlin_type = 'relu'

    elif not args.relu_dim_final is None:
        if not args.pnorm_input_dim is None or not args.pnorm_output_dim is None:
            raise Exception("--relu-dim-final argument not compatible with "
                            "--pnorm-input-dim or --pnorm-output-dim options")
        if args.relu_dim_init is None:
            raise Exception("--relu-dim-init argument should also be provided with --relu-dim-final")
        if args.relu_dim_init > args.relu_dim_final:
            raise Exception("--relu-dim-init has to be no larger than --relu-dim-final")
        args.nonlin_input_dim = None
        args.nonlin_output_dim = None
        args.nonlin_output_dim_final = args.relu_dim_final
        args.nonlin_output_dim_init = args.relu_dim_init
        args.nonlin_type = 'relu'

    else:
        if not args.relu_dim_init is None:
            raise Exception("--relu-dim-final argument not compatible with "
                            "--pnorm-input-dim or --pnorm-output-dim options")
        if not args.pnorm_input_dim > 0 or not args.pnorm_output_dim > 0:
            raise Exception("--relu-dim not set, so expected --pnorm-input-dim and "
                            "--pnorm-output-dim to be provided.");
        args.nonlin_input_dim = args.pnorm_input_dim
        args.nonlin_output_dim = args.pnorm_output_dim
        if (args.nonlin_input_dim < args.nonlin_output_dim) or (args.nonlin_input_dim % args.nonlin_output_dim != 0):
            raise Exception("Invalid --pnorm-input-dim {0} and --pnorm-output-dim {1}".format(args.nonlin_input_dim, args.nonlin_output_dim))
        args.nonlin_output_dim_final = None
        args.nonlin_output_dim_init = None
        args.nonlin_type = 'pnorm'

    if args.add_final_sigmoid and args.include_log_softmax:
        raise Exception("--include-log-softmax and --add-final-sigmoid cannot both be true.")

    if args.xent_separate_forward_affine and args.add_final_sigmoid:
        raise Exception("It does not make sense to have --add-final-sigmoid=true when xent-separate-forward-affine is true")

    if args.add_lda and args.cnn_layer is not None:
        args.add_lda = False
        warnings.warn("--add-lda is set to false as CNN layers are used.")

    if not args.max_change_per_component >= 0 or not args.max_change_per_component_final >= 0:
        raise Exception("max-change-per-component and max_change-per-component-final should be non-negative")

    return args

def AddConvMaxpLayer(config_lines, name, input, args):
    if '3d-dim' not in input:
        raise Exception("The input to AddConvMaxpLayer() needs '3d-dim' parameters.")

    input = nodes.AddConvolutionLayer(config_lines, name, input,
                              input['3d-dim'][0], input['3d-dim'][1], input['3d-dim'][2],
                              args.filt_x_dim, args.filt_y_dim,
                              args.filt_x_step, args.filt_y_step,
                              args.num_filters, input['vectorization'])

    if args.pool_x_size > 1 or args.pool_y_size > 1 or args.pool_z_size > 1:
      input = nodes.AddMaxpoolingLayer(config_lines, name, input,
                                input['3d-dim'][0], input['3d-dim'][1], input['3d-dim'][2],
                                args.pool_x_size, args.pool_y_size, args.pool_z_size,
                                args.pool_x_step, args.pool_y_step, args.pool_z_step)

    return input

# The ivectors are processed through an affine layer parallel to the CNN layers,
# then concatenated with the CNN output and passed to the deeper part of the network.
def AddCnnLayers(config_lines, cnn_layer, cnn_bottleneck_dim, cepstral_lifter, config_dir, feat_dim, splice_indexes=[0], ivector_dim=0):
    cnn_args = ParseCnnString(cnn_layer)
    num_cnn_layers = len(cnn_args)
    # We use an Idct layer here to convert MFCC to FBANK features
    common_lib.write_idct_matrix(feat_dim, cepstral_lifter, config_dir.strip() + "/idct.mat")
    prev_layer_output = {'descriptor':  "input",
                         'dimension': feat_dim}
    prev_layer_output = nodes.AddFixedAffineLayer(config_lines, "Idct", prev_layer_output, config_dir.strip() + '/idct.mat')

    list = [('Offset({0}, {1})'.format(prev_layer_output['descriptor'],n) if n != 0 else prev_layer_output['descriptor']) for n in splice_indexes]
    splice_descriptor = "Append({0})".format(", ".join(list))
    cnn_input_dim = len(splice_indexes) * feat_dim
    prev_layer_output = {'descriptor':  splice_descriptor,
                         'dimension': cnn_input_dim,
                         '3d-dim': [len(splice_indexes), feat_dim, 1],
                         'vectorization': 'yzx'}

    for cl in range(0, num_cnn_layers):
        prev_layer_output = AddConvMaxpLayer(config_lines, "L{0}".format(cl), prev_layer_output, cnn_args[cl])

    if cnn_bottleneck_dim > 0:
        prev_layer_output = nodes.AddAffineLayer(config_lines, "cnn-bottleneck", prev_layer_output, cnn_bottleneck_dim, "")

    if ivector_dim > 0:
        iv_layer_output = {'descriptor':  'ReplaceIndex(ivector, t, 0)',
                           'dimension': ivector_dim}
        iv_layer_output = nodes.AddAffineLayer(config_lines, "ivector", iv_layer_output, ivector_dim, "")
        prev_layer_output['descriptor'] = 'Append({0}, {1})'.format(prev_layer_output['descriptor'], iv_layer_output['descriptor'])
        prev_layer_output['dimension'] = prev_layer_output['dimension'] + iv_layer_output['dimension']

    return prev_layer_output

def PrintConfig(file_name, config_lines):
    f = open(file_name, 'w')
    f.write("\n".join(config_lines['components'])+"\n")
    f.write("\n#Component nodes\n")
    f.write("\n".join(config_lines['component-nodes'])+"\n")
    f.close()

def ParseCnnString(cnn_param_string_list):
    cnn_parser = argparse.ArgumentParser(description="cnn argument parser")

    cnn_parser.add_argument("--filt-x-dim", required=True, type=int)
    cnn_parser.add_argument("--filt-y-dim", required=True, type=int)
    cnn_parser.add_argument("--filt-x-step", type=int, default = 1)
    cnn_parser.add_argument("--filt-y-step", type=int, default = 1)
    cnn_parser.add_argument("--num-filters", required=True, type=int)
    cnn_parser.add_argument("--pool-x-size", type=int, default = 1)
    cnn_parser.add_argument("--pool-y-size", type=int, default = 1)
    cnn_parser.add_argument("--pool-z-size", type=int, default = 1)
    cnn_parser.add_argument("--pool-x-step", type=int, default = 1)
    cnn_parser.add_argument("--pool-y-step", type=int, default = 1)
    cnn_parser.add_argument("--pool-z-step", type=int, default = 1)

    cnn_args = []
    for cl in range(0, len(cnn_param_string_list)):
         cnn_args.append(cnn_parser.parse_args(shlex.split(cnn_param_string_list[cl])))

    return cnn_args

def ParseSpliceString(splice_indexes):
    splice_array = []
    left_context = 0
    right_context = 0
    split1 = splice_indexes.split();  # we already checked the string is nonempty.
    if len(split1) < 1:
        raise Exception("invalid splice-indexes argument, too short: "
                 + splice_indexes)
    try:
        for string in split1:
            split2 = string.split(",")
            if len(split2) < 1:
                raise Exception("invalid splice-indexes argument, too-short element: "
                         + splice_indexes)
            int_list = []
            for int_str in split2:
                int_list.append(int(int_str))
            if not int_list == sorted(int_list):
                raise Exception("elements of splice-indexes must be sorted: "
                         + splice_indexes)
            left_context += -int_list[0]
            right_context += int_list[-1]
            splice_array.append(int_list)
    except ValueError as e:
        raise Exception("invalid splice-indexes argument " + splice_indexes + str(e))
    left_context = max(0, left_context)
    right_context = max(0, right_context)

    return {'left_context':left_context,
            'right_context':right_context,
            'splice_indexes':splice_array,
            'num_hidden_layers':len(splice_array)
            }

# The function signature of MakeConfigs is changed frequently as it is intended for local use in this script.
def MakeConfigs(config_dir, splice_indexes_string,
                cnn_layer, cnn_bottleneck_dim, cepstral_lifter,
                feat_dim, ivector_dim, num_targets, add_lda,
                nonlin_type, nonlin_input_dim, nonlin_output_dim, subset_dim,
                nonlin_output_dim_init, nonlin_output_dim_final,
                use_presoftmax_prior_scale,
                final_layer_normalize_target,
                include_log_softmax,
                add_final_sigmoid,
                xent_regularize,
                xent_separate_forward_affine,
                self_repair_scale,
                max_change_per_component, max_change_per_component_final,
                objective_type):

    parsed_splice_output = ParseSpliceString(splice_indexes_string.strip())

    left_context = parsed_splice_output['left_context']
    right_context = parsed_splice_output['right_context']
    num_hidden_layers = parsed_splice_output['num_hidden_layers']
    splice_indexes = parsed_splice_output['splice_indexes']
    input_dim = len(parsed_splice_output['splice_indexes'][0]) + feat_dim + ivector_dim

    if xent_separate_forward_affine:
        if splice_indexes[-1] != [0]:
            raise Exception("--xent-separate-forward-affine option is supported only if the last-hidden layer has no splicing before it. Please use a splice-indexes with just 0 as the final splicing config.")

    prior_scale_file = '{0}/presoftmax_prior_scale.vec'.format(config_dir)

    config_lines = {'components':[], 'component-nodes':[]}

    config_files={}
    prev_layer_output = nodes.AddInputLayer(config_lines, feat_dim, splice_indexes[0], ivector_dim)

    # Add the init config lines for estimating the preconditioning matrices
    init_config_lines = copy.deepcopy(config_lines)
    init_config_lines['components'].insert(0, '# Config file for initializing neural network prior to')
    init_config_lines['components'].insert(0, '# preconditioning matrix computation')
    nodes.AddOutputLayer(init_config_lines, prev_layer_output)
    config_files[config_dir + '/init.config'] = init_config_lines

    if cnn_layer is not None:
        prev_layer_output = AddCnnLayers(config_lines, cnn_layer, cnn_bottleneck_dim, cepstral_lifter, config_dir,
                                         feat_dim, splice_indexes[0], ivector_dim)

    if add_lda:
        prev_layer_output = nodes.AddLdaLayer(config_lines, "L0", prev_layer_output, config_dir + '/lda.mat')

    left_context = 0
    right_context = 0
    # we moved the first splice layer to before the LDA..
    # so the input to the first affine layer is going to [0] index
    splice_indexes[0] = [0]

    if not nonlin_output_dim is None:
        nonlin_output_dims = [nonlin_output_dim] * num_hidden_layers
    elif nonlin_output_dim_init < nonlin_output_dim_final and num_hidden_layers == 1:
        raise Exception("num-hidden-layers has to be greater than 1 if relu-dim-init and relu-dim-final is different.")
    else:
        # computes relu-dim for each hidden layer. They increase geometrically across layers
        factor = pow(float(nonlin_output_dim_final) / nonlin_output_dim_init, 1.0 / (num_hidden_layers - 1)) if num_hidden_layers > 1 else 1
        nonlin_output_dims = [int(round(nonlin_output_dim_init * pow(factor, i))) for i in range(0, num_hidden_layers)]
        assert(nonlin_output_dims[-1] >= nonlin_output_dim_final - 1 and nonlin_output_dims[-1] <= nonlin_output_dim_final + 1) # due to rounding error
        nonlin_output_dims[-1] = nonlin_output_dim_final # It ensures that the dim of the last hidden layer is exactly the same as what is specified

    for i in range(0, num_hidden_layers):
        # make the intermediate config file for layerwise discriminative training

        # prepare the spliced input
        if not (len(splice_indexes[i]) == 1 and splice_indexes[i][0] == 0):
            try:
                zero_index = splice_indexes[i].index(0)
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
            for j in range(len(splice_indexes[i])):
                if j == zero_index:
                    appended_descriptors.append(prev_layer_output['descriptor'])
                    appended_dimension += prev_layer_output['dimension']
                    continue
                appended_descriptors.append('Offset({0}, {1})'.format(subset_output['descriptor'], splice_indexes[i][j]))
                appended_dimension += subset_output['dimension']
            prev_layer_output = {'descriptor' : "Append({0})".format(" , ".join(appended_descriptors)),
                                 'dimension'  : appended_dimension}
        else:
            # this is a normal affine node
            pass

        if xent_separate_forward_affine and i == num_hidden_layers - 1:
            if xent_regularize == 0.0:
                raise Exception("xent-separate-forward-affine=True is valid only if xent-regularize is non-zero")

            if nonlin_type == "relu" :
                prev_layer_output_chain = nodes.AddAffRelNormLayer(config_lines, "Tdnn_pre_final_chain",
                                                                   prev_layer_output, nonlin_output_dim,
                                                                   norm_target_rms = final_layer_normalize_target,
                                                                   self_repair_scale = self_repair_scale,
                                                                   max_change_per_component = max_change_per_component)

                prev_layer_output_xent = nodes.AddAffRelNormLayer(config_lines, "Tdnn_pre_final_xent",
                                                                  prev_layer_output, nonlin_output_dim,
                                                                  norm_target_rms = final_layer_normalize_target,
                                                                  self_repair_scale = self_repair_scale,
                                                                  max_change_per_component = max_change_per_component)
            elif nonlin_type == "pnorm" :
                prev_layer_output_chain = nodes.AddAffPnormLayer(config_lines, "Tdnn_pre_final_chain",
                                                                 prev_layer_output, nonlin_input_dim, nonlin_output_dim,
                                                                 norm_target_rms = final_layer_normalize_target)

                prev_layer_output_xent = nodes.AddAffPnormLayer(config_lines, "Tdnn_pre_final_xent",
                                                                prev_layer_output, nonlin_input_dim, nonlin_output_dim,
                                                                norm_target_rms = final_layer_normalize_target)
            else:
                raise Exception("Unknown nonlinearity type")

            nodes.AddFinalLayer(config_lines, prev_layer_output_chain, num_targets,
                               max_change_per_component = max_change_per_component_final,
                               use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                               prior_scale_file = prior_scale_file,
                               include_log_softmax = include_log_softmax)

            nodes.AddFinalLayer(config_lines, prev_layer_output_xent, num_targets,
                                ng_affine_options = " param-stddev=0 bias-stddev=0 learning-rate-factor={0} ".format(
                                    0.5 / xent_regularize),
                                max_change_per_component = max_change_per_component_final,
                                use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                                prior_scale_file = prior_scale_file,
                                include_log_softmax = True,
                                name_affix = 'xent')
        else:
            if nonlin_type == "relu":
                prev_layer_output = nodes.AddAffRelNormLayer(config_lines, "Tdnn_{0}".format(i),
                                                            prev_layer_output, nonlin_output_dims[i],
                                                            norm_target_rms = 1.0 if i < num_hidden_layers -1 else final_layer_normalize_target,
                                                            self_repair_scale = self_repair_scale,
                                                            max_change_per_component = max_change_per_component)
            elif nonlin_type == "pnorm":
                prev_layer_output = nodes.AddAffPnormLayer(config_lines, "Tdnn_{0}".format(i),
                                                           prev_layer_output, nonlin_input_dim, nonlin_output_dim,
                                                           norm_target_rms = 1.0 if i < num_hidden_layers -1 else final_layer_normalize_target)
            else:
                raise Exception("Unknown nonlinearity type")
            # a final layer is added after each new layer as we are generating
            # configs for layer-wise discriminative training

            # add_final_sigmoid adds a sigmoid as a final layer as alternative
            # to log-softmax layer.
            # http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression#Softmax_Regression_vs._k_Binary_Classifiers
            # This is useful when you need the final outputs to be probabilities between 0 and 1.
            # Usually used with an objective-type such as "quadratic".
            # Applications are k-binary classification such Ideal Ratio Mask prediction.
            nodes.AddFinalLayer(config_lines, prev_layer_output, num_targets,
                               max_change_per_component = max_change_per_component_final,
                               use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                               prior_scale_file = prior_scale_file,
                               include_log_softmax = include_log_softmax,
                               add_final_sigmoid = add_final_sigmoid,
                               objective_type = objective_type)
            if xent_regularize != 0.0:
                nodes.AddFinalLayer(config_lines, prev_layer_output, num_targets,
                                    ng_affine_options = " param-stddev=0 bias-stddev=0 learning-rate-factor={0} ".format(
                                          0.5 / xent_regularize),
                                    max_change_per_component = max_change_per_component_final,
                                    use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                                    prior_scale_file = prior_scale_file,
                                    include_log_softmax = True,
                                    name_affix = 'xent')

        config_files['{0}/layer{1}.config'.format(config_dir, i+1)] = config_lines
        config_lines = {'components':[], 'component-nodes':[]}

    left_context += int(parsed_splice_output['left_context'])
    right_context += int(parsed_splice_output['right_context'])

    # write the files used by other scripts like steps/nnet3/get_egs.sh
    f = open(config_dir + "/vars", "w")
    print('model_left_context=' + str(left_context), file=f)
    print('model_right_context=' + str(right_context), file=f)
    print('num_hidden_layers=' + str(num_hidden_layers), file=f)
    print('num_targets=' + str(num_targets), file=f)
    print('add_lda=' + ('true' if add_lda else 'false'), file=f)
    print('include_log_softmax=' + ('true' if include_log_softmax else 'false'), file=f)
    print('objective_type=' + objective_type, file=f)
    f.close()

    # printing out the configs
    # init.config used to train lda-mllt train
    for key in config_files.keys():
        PrintConfig(key, config_files[key])

def Main():
    args = GetArgs()

    MakeConfigs(config_dir = args.config_dir,
                splice_indexes_string = args.splice_indexes,
                feat_dim = args.feat_dim, ivector_dim = args.ivector_dim,
                num_targets = args.num_targets,
                add_lda = args.add_lda,
                cnn_layer = args.cnn_layer,
                cnn_bottleneck_dim = args.cnn_bottleneck_dim,
                cepstral_lifter = args.cepstral_lifter,
                nonlin_type = args.nonlin_type,
                nonlin_input_dim = args.nonlin_input_dim,
                nonlin_output_dim = args.nonlin_output_dim,
                subset_dim = args.subset_dim,
                nonlin_output_dim_init = args.nonlin_output_dim_init,
                nonlin_output_dim_final = args.nonlin_output_dim_final,
                use_presoftmax_prior_scale = args.use_presoftmax_prior_scale,
                final_layer_normalize_target = args.final_layer_normalize_target,
                include_log_softmax = args.include_log_softmax,
                add_final_sigmoid = args.add_final_sigmoid,
                xent_regularize = args.xent_regularize,
                xent_separate_forward_affine = args.xent_separate_forward_affine,
                self_repair_scale = args.self_repair_scale_nonlinearity,
                max_change_per_component = args.max_change_per_component,
                max_change_per_component_final = args.max_change_per_component_final,
                objective_type = args.objective_type)

if __name__ == "__main__":
    Main()

