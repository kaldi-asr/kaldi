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
from collections import defaultdict

sys.path.insert(0, 'steps/nnet3/libs/')
from xconfig_lib import *
from xconfig_layers import *


def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="Reads an xconfig file and creates config files "
                                     "for neural net creation and training",
                                     epilog="Search egs/*/*/local/nnet3/*sh for examples")

    parser.add_argument("--self-repair-scale-nonlinearity", type=float,
                        help="A non-zero value activates the self-repair mechanism in "
                        "nonlinearities (larger -> faster self-repair)", default=1.0e-05)
    parser.add_argument("xconfig_file",
                        help="Filename of input xconfig file")
    parser.add_argument("config_dir",
                        help="Directory to write config files and variables")

    print(' '.join(sys.argv))

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if not os.path.exists(args.config_dir):
        os.makedirs(args.config_dir)
    if args.self_repair_scale_nonlinearity < 0.0 or args.self_repair_scale_nonlinearity > 0.1:
        sys.exit("{0}: invalid option --self-repair-scale-nonlinearity={1}".format(
            sys.argv[0], args.self_repair_scale_nonlinearity))

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
                                                                   self_repair_scale = self_repair_scale,
                                                                   norm_target_rms = final_layer_normalize_target)

                prev_layer_output_xent = nodes.AddAffRelNormLayer(config_lines, "Tdnn_pre_final_xent",
                                                                  prev_layer_output, nonlin_output_dim,
                                                                  self_repair_scale = self_repair_scale,
                                                                  norm_target_rms = final_layer_normalize_target)
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
                               use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                               prior_scale_file = prior_scale_file,
                               include_log_softmax = include_log_softmax)

            nodes.AddFinalLayer(config_lines, prev_layer_output_xent, num_targets,
                                ng_affine_options = " param-stddev=0 bias-stddev=0 learning-rate-factor={0} ".format(
                                    0.5 / xent_regularize),
                                use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                                prior_scale_file = prior_scale_file,
                                include_log_softmax = True,
                                name_affix = 'xent')
        else:
            if nonlin_type == "relu":
                prev_layer_output = nodes.AddAffRelNormLayer(config_lines, "Tdnn_{0}".format(i),
                                                            prev_layer_output, nonlin_output_dims[i],
                                                            self_repair_scale = self_repair_scale,
                                                            norm_target_rms = 1.0 if i < num_hidden_layers -1 else final_layer_normalize_target)
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
                               use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                               prior_scale_file = prior_scale_file,
                               include_log_softmax = include_log_softmax,
                               add_final_sigmoid = add_final_sigmoid,
                               objective_type = objective_type)
            if xent_regularize != 0.0:
                nodes.AddFinalLayer(config_lines, prev_layer_output, num_targets,
                                    ng_affine_options = " param-stddev=0 bias-stddev=0 learning-rate-factor={0} ".format(
                                          0.5 / xent_regularize),
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


def BackUpXconfigFile(xconfig_file, config_dir):
    # we write a copy of the xconfig file just to have a record of the original
    # input.
    try:
        xconfig_file_out = open(config_dir + "/xconfig")
    except:
        sys.exit("{0}: error opening file {1}/xconfig for output".format(
            sys.argv[0], config_dir))
    try:
        xconfig_file_in = open(xconfig_file)
    except:
        sys.exit("{0}: error opening file {1} for input".format(sys.argv[0], config_dir))

    print("# This file was created by the command:\n"
          "# {0}\n"
          "# It is a copy of the source from which the config files in "
          "# this directory were generated.\n".format(' '.join(sys.argv)),
          file=xconfig_file_out)

    while True:
        line = xconfig_file_in.readline()
        if line == '':
            break
        print(line.strip(), file=xconfig_file_out)
    xconfig_file_out.close()
    xconfig_file_in.close()


def WriteExpandedXconfigFile(config_dir, all_layers):
    try:
        xconfig_file_out = open(config_dir + "/xconfig.expanded")
    except:
        sys.exit("{0}: error opening file {1}/xconfig.expanded for output".format(
            sys.argv[0], config_dir))

    print("# This file was created by {0}.  It contains the same content as\n"
          "# ./xconfig but it was parsed, default config values were set, and\n"
          "# it was printed from the internal representation.\n".format(sys.argv[0]),
          file=xconfig_file_out)

    for layer in all_layers:
        print(str(layer), file=xconfig_file_out)
    xconfig_file_out.close()


# This function returns a map from config-file basename
# e.g. 'init', 'ref', 'layer1' to a documentation string that goes
# at the top of the file.
def GetConfigHeaders():
    ans = defaultdict(str)  # resulting dict will default to the empty string
                            # for any config files not explicitly listed here.
    ans['init'] = ("# This file was created by the command:\n"
                   "# " + ' '.join(sys.argv) + "\n"
                   "# It contains the input of the network and is used in\n"
                   "# accumulating stats for an LDA-like transform of the\n"
                   "# input features.\n");
    ans['ref'] = ("# This file was created by the command:\n"
                  "# " + ' '.join(sys.argv) + "\n"
                  "# It contains the entire neural network, but with those\n"
                  "# components that would normally require fixed vectors/matrices\n"
                  "# read from disk, replaced with random initialization\n"
                  "# (this applies to the LDA-like transform and the\n"
                  "# presoftmax-prior-scale, if applicable).  This file\n"
                  "# is used only to work out the left-context and right-context\n"
                  "# of the network.\n");
    ans['all'] = ("# This file was created by the command:\n"
                  "# " + ' '.join(sys.argv) + "\n"
                  "# It contains the entire neural network.  It might not be used\n"
                  "# in the current scripts; it's provided for forward compatibility\n"
                  "# to possible future changes.\n")

    # Note: currently we just copy all lines that were going to go to 'all', into
    # 'layer1', to avoid propagating this nastiness to the code in xconfig_layers.py
    ans['layer1'] = ("# This file was created by the command:\n"
                     "# " + ' '.join(sys.argv) + "\n"
                     "# It contains the configuration of the entire neural network.\n"
                     "# The contents are the same\n"
                     "# as 'all.config'.  The reason this file is named this way (and\n"
                     "# that the config file `num_hidden_layers` contains 1, even though\n"
                     "# this file may really contain more than 1 hidden layer), is\n"
                     "# historical... we used to create networks by adding hidden layers\n"
                     "# one by one (discriminative pretraining), but more recently we\n"
                     "# have found that it's better to add them all at once.  This file\n"
                     "# exists to enable the older training scripts to work.  Note:\n"
                     "# it contains the inputs of the neural network even though it doesn't\n"
                     "# have to (since they are included in 'init.config').  This will\n"
                     "# give us the flexibility to change the scripts in future.\n");
    return ans;




# This is where most of the work of this program happens.
def WriteConfigFiles(config_dir, all_layers):
    config_basename_to_lines = defaultdict(list)2

    config_basename_to_header = GetConfigHeaders()





def Main():
    args = GetArgs()

    BackUpXconfigFile(args.xconfig_file, args.config_dir)

    all_layers = ReadXconfigFile(args.xconfig_file)

    WriteExpandedXconfigFile(args.config_dir all_layers)

    try:
        f =
        shutil.copyfile(args.xconfig_file, args.xconfig_dir

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
