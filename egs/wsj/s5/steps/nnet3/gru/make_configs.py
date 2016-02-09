#!/usr/bin/env python

# Gated Recurrent Unit(GRU) is a kind of recurrent neural network similar to LSTM, but faster and less likely to diverge than LSTM.
# See http://arxiv.org/pdf/1512.02595v1.pdf for more info about the network.

from __future__ import print_function
import os
import argparse
import sys
import warnings
import copy
import imp

nodes = imp.load_source('nodes', 'steps/nnet3/components.py')
nnet3_train_lib = imp.load_source('ntl', 'steps/nnet3/nnet3_train_lib.py')
chain_lib = imp.load_source('ncl', 'steps/nnet3/chain/nnet3_chain_lib.py')

def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="Writes config files and variables "
                                                 "for GRU`s creation and training",
                                     epilog="See steps/nnet3/train_rnn.py for example.")

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

    # General neural network options
    parser.add_argument("--splice-indexes", type=str,
                        help="Splice indexes at input layer, e.g. '-3,-2,-1,0,1,2,3'", required = True, default="0")
    parser.add_argument("--xent-regularize", type=float,
                        help="For chain models, if nonzero, add a separate output for cross-entropy "
                        "regularization (with learning-rate-factor equal to the inverse of this)",
                        default=0.0)
    parser.add_argument("--include-log-softmax", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="add the final softmax layer ", default=True, choices = ["false", "true"])

    # GRU options
    parser.add_argument("--num-gru-layers", type=int,
                        help="Number of GRU layers to be stacked", default=1)
    parser.add_argument("--recurrent-projection-dim", type=int,
                        help="dimension of recurrent projection")
    parser.add_argument("--non-recurrent-projection-dim", type=int,
                        help="dimension of non-recurrent projection")
    parser.add_argument("--hidden-dim", type=int,
                        help="dimension of fully-connected layers")

    # Natural gradient options
    parser.add_argument("--ng-affine-options", type=str,
                        help="options to be supplied to NaturalGradientAffineComponent", default="")

    # Gradient clipper options
    parser.add_argument("--norm-based-clipping", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="use norm based clipping in ClipGradient components ", default=True, choices = ["false", "true"])
    parser.add_argument("--clipping-threshold", type=float,
                        help="clipping threshold used in ClipGradient components, if clipping-threshold=0 no clipping is done", default=30)
    parser.add_argument("--self-repair-scale", type=float,
                        help="A non-zero value activates the self-repair mechanism in the sigmoid and tanh non-linearities of the GRU", default=None)

    # Delay options
    parser.add_argument("--label-delay", type=int, default=None,
                        help="option to delay the labels to make the gru robust")

    parser.add_argument("--gru-delay", type=str, default=None,
                        help="option to have different delays in recurrence for each gru")

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

    if (args.num_gru_layers < 1):
        sys.exit("--num-gru-layers has to be a positive integer")
    if (args.clipping_threshold < 0):
        sys.exit("--clipping-threshold has to be a non-negative")
    if args.gru_delay is None:
        args.gru_delay = [[-1]] * args.num_gru_layers
    else:
        try:
            args.gru_delay = ParseGruDelayString(args.gru_delay.strip())
        except ValueError:
            sys.exit("--gru-delay has incorrect format value. Provided value is '{0}'".format(args.gru_delay))
        if len(args.gru_delay) != args.num_gru_layers:
            sys.exit("--gru-delay: Number of delays provided has to match --num-gru-layers")

    return args

def PrintConfig(file_name, config_lines):
    f = open(file_name, 'w')
    f.write("\n".join(config_lines['components'])+"\n")
    f.write("\n#Component nodes\n")
    f.write("\n".join(config_lines['component-nodes'])+"\n")
    f.close()

def WriteScaleMinusOne(file_name, recurrent_projection_dim):
    f = open(file_name, 'w')
    f.write(" [ ")
    for i in range(recurrent_projection_dim):
        f.write("-1 ")
    f.write("]\n")
    f.close()

def WriteBiasOne(file_name, recurrent_projection_dim):
    f = open(file_name, 'w')
    f.write(" [ ")
    for i in range(recurrent_projection_dim):
        f.write("1 ")
    f.write("]\n")
    f.close()

def ParseSpliceString(splice_indexes, label_delay=None):
    ## Work out splice_array e.g. splice_array = [ [ -3,-2,...3 ], [0], [-2,2], .. [ -8,8 ] ]
    split1 = splice_indexes.split(" ");  # we already checked the string is nonempty.
    if len(split1) < 1:
        splice_indexes = "0"

    left_context=0
    right_context=0
    if label_delay is not None:
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

def ParseGruDelayString(gru_delay):
    ## Work out gru_delay e.g. "-1 [-1,1] -2" -> list([ [-1], [-1, 1], [-2] ])
    split1 = gru_delay.split(" ");
    gru_delay_array = []
    try:
        for i in range(len(split1)):
            indexes = map(lambda x: int(x), split1[i].strip().lstrip('[').rstrip(']').strip().split(","))
            if len(indexes) < 1:
                raise ValueError("invalid --gru-delay argument, too-short element: "
                                + gru_delay)
	    elif len(indexes) == 2 and indexes[0] * indexes[1] >= 0:
                raise ValueError('Warning: ' + str(indexes) + ' is not a standard BGRU mode. There should be a negative delay for the forward, and a postive delay for the backward.')
            gru_delay_array.append(indexes)
    except ValueError as e:
        raise ValueError("invalid --gru-delay argument " + gru_delay + str(e))

    return gru_delay_array


def MakeConfigs(config_dir, feat_dim, ivector_dim, num_targets,
                splice_indexes, gru_delay,
                recurrent_projection_dim, non_recurrent_projection_dim,
                num_gru_layers, num_hidden_layers,
                norm_based_clipping, clipping_threshold,
                ng_affine_options,
                label_delay, include_log_softmax, xent_regularize, self_repair_scale):

    WriteScaleMinusOne(config_dir + '/scale_minus_one.vec', recurrent_projection_dim)
    WriteBiasOne(config_dir + '/bias_one.vec', recurrent_projection_dim)

    config_lines = {'components':[], 'component-nodes':[]}

    config_files={}
    prev_layer_output = nodes.AddInputLayer(config_lines, feat_dim, splice_indexes[0], ivector_dim)

    # Add the init config lines for estimating the preconditioning matrices
    init_config_lines = copy.deepcopy(config_lines)
    init_config_lines['components'].insert(0, '# Config file for initializing neural network prior to')
    init_config_lines['components'].insert(0, '# preconditioning matrix computation')
    nodes.AddOutputLayer(init_config_lines, prev_layer_output)
    config_files[config_dir + '/init.config'] = init_config_lines

    prev_layer_output = nodes.AddLdaLayer(config_lines, "L0", prev_layer_output, config_dir + '/lda.mat')

    for i in range(num_gru_layers):
        if len(gru_delay[i]) == 2: # BGRU layer case, add both forward and backward
            prev_layer_output1 = nodes.AddGruLayer(config_lines, "BGru{0}_forward".format(i+1), prev_layer_output,
                                             recurrent_projection_dim, non_recurrent_projection_dim,
                                             config_dir + '/scale_minus_one.vec', config_dir + '/bias_one.vec',
                                             clipping_threshold, norm_based_clipping, ng_affine_options,
                                             gru_delay = gru_delay[i][0], self_repair_scale = self_repair_scale)
            prev_layer_output2 = nodes.AddGruLayer(config_lines, "BGru{0}_backward".format(i+1), prev_layer_output,
                                             recurrent_projection_dim, non_recurrent_projection_dim,
                                             config_dir + '/scale_minus_one.vec', config_dir + '/bias_one.vec',
                                             clipping_threshold, norm_based_clipping, ng_affine_options,
                                             gru_delay = gru_delay[i][1], self_repair_scale = self_repair_scale)
            prev_layer_output['descriptor'] = 'Append({0}, {1})'.format(prev_layer_output1['descriptor'], prev_layer_output2['descriptor'])
            prev_layer_output['dimension'] = prev_layer_output1['dimension'] + prev_layer_output2['dimension']
        else: # GRU layer case
            prev_layer_output = nodes.AddGruLayer(config_lines, "Gru{0}".format(i+1), prev_layer_output,
                                            recurrent_projection_dim, non_recurrent_projection_dim,
                                            config_dir + '/scale_minus_one.vec', config_dir + '/bias_one.vec',
                                            clipping_threshold, norm_based_clipping, ng_affine_options,
                                            gru_delay = gru_delay[i][0], self_repair_scale = self_repair_scale)
        # make the intermediate config file for layerwise discriminative
        # training
        nodes.AddFinalLayer(config_lines, prev_layer_output, num_targets, ng_affine_options, label_delay = label_delay, include_log_softmax = include_log_softmax)


        if xent_regularize != 0.0:
            nodes.AddFinalLayer(config_lines, prev_layer_output, num_targets,
                                include_log_softmax = True,
                                name_affix = 'xent')

        config_files['{0}/layer{1}.config'.format(config_dir, i+1)] = config_lines
        config_lines = {'components':[], 'component-nodes':[]}

    for i in range(num_gru_layers, num_hidden_layers):
        prev_layer_output = nodes.AddAffRelNormLayer(config_lines, "L{0}".format(i+1),
                                               prev_layer_output, hidden_dim,
                                               ng_affine_options, self_repair_scale = self_repair_scale)
        # make the intermediate config file for layerwise discriminative
        # training
        nodes.AddFinalLayer(config_lines, prev_layer_output, num_targets, ng_affine_options, label_delay = label_delay, include_log_softmax = include_log_softmax)

        if xent_regularize != 0.0:
            nodes.AddFinalLayer(config_lines, prev_layer_output, num_targets,
                                include_log_softmax = True,
                                name_affix = 'xent')

        config_files['{0}/layer{1}.config'.format(config_dir, i+1)] = config_lines
        config_lines = {'components':[], 'component-nodes':[]}

    # printing out the configs
    # init.config used to train lda-mllt train
    for key in config_files.keys():
        PrintConfig(key, config_files[key])




def ProcessSpliceIndexes(config_dir, splice_indexes, label_delay, num_gru_layers):
    parsed_splice_output = ParseSpliceString(splice_indexes.strip(), label_delay)
    left_context = parsed_splice_output['left_context']
    right_context = parsed_splice_output['right_context']
    num_hidden_layers = parsed_splice_output['num_hidden_layers']
    splice_indexes = parsed_splice_output['splice_indexes']

    if (num_hidden_layers < num_gru_layers):
        raise Exception("num-gru-layers : number of gru layers has to be greater than number of layers, decided based on splice-indexes")

    # write the files used by other scripts like steps/nnet3/get_egs.sh
    f = open(config_dir + "/vars", "w")
    print('model_left_context=' + str(left_context), file=f)
    print('model_right_context=' + str(right_context), file=f)
    print('num_hidden_layers=' + str(num_hidden_layers), file=f)
    # print('initial_right_context=' + str(splice_array[0][-1]), file=f)
    f.close()

    return [left_context, right_context, num_hidden_layers, splice_indexes]


def Main():
    args = GetArgs()
    [left_context, right_context, num_hidden_layers, splice_indexes] = ProcessSpliceIndexes(args.config_dir, args.splice_indexes, args.label_delay, args.num_gru_layers)

    MakeConfigs(args.config_dir,
                args.feat_dim, args.ivector_dim, args.num_targets,
                splice_indexes, args.gru_delay,
                args.recurrent_projection_dim, args.non_recurrent_projection_dim,
                args.num_gru_layers, num_hidden_layers,
                args.norm_based_clipping,
                args.clipping_threshold,
                args.ng_affine_options,
                args.label_delay, args.include_log_softmax, args.xent_regularize,
                args.self_repair_scale)

if __name__ == "__main__":
    Main()
