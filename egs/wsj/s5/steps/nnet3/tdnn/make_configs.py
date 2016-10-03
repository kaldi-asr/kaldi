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
nnet3_train_lib = imp.load_source('ntl', 'steps/nnet3/nnet3_train_lib.py')
chain_lib = imp.load_source('ncl', 'steps/nnet3/chain/nnet3_chain_lib.py')

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
    parser.add_argument("--add-lda", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="If \"true\" an LDA matrix computed from the input features "
                        "(spliced according to the first set of splice-indexes) will be used as "
                        "the first Affine layer. This affine layer's parameters are fixed during training. "
                        "If --cnn.layer is specified this option will be forced to \"false\".",
                        default=True, choices = ["false", "true"])

    parser.add_argument("--include-log-softmax", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="add the final softmax layer ", default=True, choices = ["false", "true"])
    parser.add_argument("--add-final-sigmoid", type=str, action=nnet3_train_lib.StrToBoolAction,
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
    parser.add_argument("--xent-separate-forward-affine", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="if using --xent-regularize, gives it separate last-but-one weight matrix",
                        default=False, choices = ["false", "true"])
    parser.add_argument("--final-layer-normalize-target", type=float,
                        help="RMS target for final layer (set to <1 if final layer learns too fast",
                        default=1.0)
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


    parser.add_argument("--use-presoftmax-prior-scale", type=str, action=nnet3_train_lib.StrToBoolAction,
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
        args.feat_dim = nnet3_train_lib.GetFeatDim(args.feat_dir)

    if args.ali_dir is not None:
        args.num_targets = nnet3_train_lib.GetNumberOfLeaves(args.ali_dir)
    elif args.tree_dir is not None:
        args.num_targets = chain_lib.GetNumberOfLeaves(args.tree_dir)

    if args.ivector_dir is not None:
        args.ivector_dim = nnet3_train_lib.GetIvectorDim(args.ivector_dir)

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
    num_learnable_params = 0
    cnn_args = ParseCnnString(cnn_layer)
    num_cnn_layers = len(cnn_args)
    # We use an Idct layer here to convert MFCC to FBANK features
    nnet3_train_lib.WriteIdctMatrix(feat_dim, cepstral_lifter, config_dir.strip() + "/idct.mat")
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
        prev_layer = AddConvMaxpLayer(config_lines, "L{0}".format(cl), prev_layer_output, cnn_args[cl])
        prev_layer_output = prev_layer['output']
        num_learnable_params = prev_layer['num_learnable_params']

    if cnn_bottleneck_dim > 0:
        prev_layer = nodes.AddAffineLayer(config_lines, "cnn-bottleneck", prev_layer_output, cnn_bottleneck_dim, "")
        prev_layer_output = prev_layer['output']
        num_learnable_params = prev_layer['num_learnable_params']

    if ivector_dim > 0:
        iv_layer_output = {'descriptor':  'ReplaceIndex(ivector, t, 0)',
                           'dimension': ivector_dim}
        iv_layer = nodes.AddAffineLayer(config_lines, "ivector", iv_layer_output, ivector_dim, "")
        iv_layer_output = iv_layer['output']
        num_learnable_params += iv_layer['num_learnable_params']

        prev_layer_output['descriptor'] = 'Append({0}, {1})'.format(prev_layer_output['descriptor'], iv_layer_output['descriptor'])
        prev_layer_output['dimension'] = prev_layer_output['dimension'] + iv_layer_output['dimension']

    return {'output' : prev_layer_output,
            'num_learnable_params' : num_learnable_params}

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
                objective_type):

    parsed_splice_output = ParseSpliceString(splice_indexes_string.strip())

    num_hidden_layers = parsed_splice_output['num_hidden_layers']
    splice_indexes = parsed_splice_output['splice_indexes']
    input_dim = len(parsed_splice_output['splice_indexes'][0]) + feat_dim + ivector_dim

    if xent_separate_forward_affine:
        if splice_indexes[-1] != [0]:
            raise Exception("--xent-separate-forward-affine option is supported only if the last-hidden layer has no splicing before it. Please use a splice-indexes with just 0 as the final splicing config.")

    prior_scale_file = '{0}/presoftmax_prior_scale.vec'.format(config_dir)

    # start the config generation process
    config_lines = {'components':[], 'component-nodes':[]}
    config_files={}

    num_learnable_params = 0
    num_learnable_params_xent = 0   # number of parameters in xent-branch of chain models

    prev_layer = nodes.AddInputLayer(config_lines, feat_dim, splice_indexes[0], ivector_dim)
    prev_layer_output = prev_layer['output']
    # we moved the first splice layer to before the LDA..
    # so the input to the first affine layer is going to [0] index
    splice_indexes[0] = [0]

    # Adding the init config lines for estimating the preconditioning matrices
    init_config_lines = copy.deepcopy(config_lines)
    init_config_lines['components'].insert(0, '# Config file for initializing neural network prior to')
    init_config_lines['components'].insert(0, '# preconditioning matrix computation')
    nodes.AddOutputLayer(init_config_lines, prev_layer_output)
    config_files[config_dir + '/init.config'] = init_config_lines

    if cnn_layer is not None:
        prev_layer = AddCnnLayers(config_lines, cnn_layer, cnn_bottleneck_dim, cepstral_lifter, config_dir,
                                         feat_dim, splice_indexes[0], ivector_dim)
        prev_layer_output = prev_layer['output']
        num_learnable_params += prev_layer['num_learnable_params']

    if add_lda:
        prev_layer = nodes.AddLdaLayer(config_lines, "L0", prev_layer_output, config_dir + '/lda.mat')
        prev_layer_output = prev_layer['output']

    # generating the output-dims for each layer
    if nonlin_type == "pnorm":
        # we don't increase the output dimension for each layer
        # we might support this in the future
        nonlin_output_dims = [nonlin_output_dim] * num_hidden_layers
    elif not nonlin_output_dim is None:
        nonlin_output_dims = [nonlin_output_dim] * num_hidden_layers
    elif nonlin_output_dim_init < nonlin_output_dim_final and num_hidden_layers == 1:
        raise Exception("num-hidden-layers has to be greater than 1 if relu-dim-init and relu-dim-final is different.")
    else:
        # computes relu-dim for each hidden layer. They increase geometrically across layers
        factor = pow(float(nonlin_output_dim_final) / nonlin_output_dim_init, 1.0 / (num_hidden_layers - 1)) if num_hidden_layers > 1 else 1
        nonlin_output_dims = [int(round(nonlin_output_dim_init * pow(factor, i))) for i in range(0, num_hidden_layers)]
        assert(nonlin_output_dims[-1] >= nonlin_output_dim_final - 1 and nonlin_output_dims[-1] <= nonlin_output_dim_final + 1) # due to rounding error
        nonlin_output_dims[-1] = nonlin_output_dim_final # It ensures that the dim of the last hidden layer is exactly the same as what is specified


    # Adding the TDNN layers
    for i in range(0, num_hidden_layers):
        if xent_separate_forward_affine and i == num_hidden_layers - 1:
            # xent_separate_forward_affine is only done when adding the final hidden layer
            # this is the final layer so assert that splice index is [0]
            assert(splice_indexes[i] == [0])
            if xent_regularize == 0.0:
                raise Exception("xent-separate-forward-affine=True is valid only if xent-regularize is non-zero")

            # we use named arguments as we do not want argument offset errors
            num_params_fin, num_params_fin_xent = nodes.AddFinalLayersWithXentSeperateForwardAffineRegularizer(config_lines,
                                                                                                             input = prev_layer_output,
                                                                                                             num_targets = num_targets,
                                                                                                             nonlin_type = nonlin_type,
                                                                                                             nonlin_input_dim = nonlin_input_dim,
                                                                                                             nonlin_output_dim = nonlin_output_dim,
                                                                                                             use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                                                                                                             prior_scale_file = prior_scale_file,
                                                                                                             include_log_softmax = include_log_softmax,
                                                                                                             self_repair_scale = self_repair_scale,
                                                                                                             xent_regularize = xent_regularize,
                                                                                                             final_layer_normalize_target = final_layer_normalize_target)



        else:
            if splice_indexes[i] == ]:
                # add a normal affine layer
                prev_layer = nodes.AddAffineNonlinLayer(config_lines, 'Affine{0}'.format(i),
                                                          prev_layer_output,
                                                          nonlin_type, nonlin_input_dim, nonlin_output_dims[i],
                                                          self_repair_scale = self_repair_scale,
                                                          norm_target_rms = 1.0 if i < num_hidden_layers -1 else final_layer_normalize_target)
            else:
                prev_layer = nodes.AddTdnnLayer(config_lines, 'Tdnn{0}'.format(i+1), prev_layer_output,
                                                splice_indexes[i],
                                                nonlin_type, nonlin_input_dim, nonlin_output_dims[i],
                                                subset_dim = subset_dim,
                                                self_repair_scale = self_repair_scale,
                                                norm_target_rms = 1.0 if i < num_hidden_layers -1 else final_layer_normalize_target)


            # a final layer is added after each new layer as we are generating
            # configs for layer-wise discriminative training
            num_params_fin, num_params_fin_xent = nodes.AddFinalLayerWithXentRegularizer(config_lines,
                                                                                         input = prev_layer_output,
                                                                                         num_targets = num_targets,
                                                                                         use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                                                                                         prior_scale_file = prior_scale_file,
                                                                                         include_log_softmax = include_log_softmax,
                                                                                         self_repair_scale = self_repair_scale,
                                                                                         xent_regularize = xent_regularize,
                                                                                         add_final_sigmoid = add_final_sigmoid,
                                                                                         objective_type = objective_type)




        config_files['{0}/layer{1}.config'.format(config_dir, i+1)] = config_lines
        config_lines = {'components':[], 'component-nodes':[]}

    num_learnable_params += num_params_fin
    num_learnable_params_xent = num_params_fin_xent

    left_context = int(parsed_splice_output['left_context'])
    right_context = int(parsed_splice_output['right_context'])
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

    print('This model has num_learnable_params={0:,} and num_learnable_params_xent={1:,}'.format(num_learnable_params, num_learnable_params_xent))

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
                objective_type = args.objective_type)

if __name__ == "__main__":
    Main()

