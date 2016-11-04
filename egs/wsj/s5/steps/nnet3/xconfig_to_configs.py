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
# the following is in case we weren't running this from the normal directory.
sys.path.insert(0, os.path.realpath(os.path.dirname(sys.argv[0])) + '/libs/')

import xconfig_utils
import xconfig_layers


def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description='Reads an xconfig file and creates config files '
                                     'for neural net creation and training',
                                     epilog='Search egs/*/*/local/nnet3/*sh for examples')
    parser.add_argument('xconfig_file',
                        help='Filename of input xconfig file')
    parser.add_argument('config_dir',
                        help='Directory to write config files and variables')

    print(' '.join(sys.argv))

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if not os.path.exists(args.config_dir):
        os.makedirs(args.config_dir)
    return args


#     # write the files used by other scripts like steps/nnet3/get_egs.sh
#     f = open(config_dir + 'vars', 'w')
#     print('model_left_context=' + str(left_context), file=f)
#     print('model_right_context=' + str(right_context), file=f)
#     print('num_hidden_layers=' + str(num_hidden_layers), file=f)
#     print('num_targets=' + str(num_targets), file=f)
#     print('add_lda=' + ('true' if add_lda else 'false'), file=f)
#     print('include_log_softmax=' + ('true' if include_log_softmax else 'false'), file=f)
#     print('objective_type=' + objective_type, file=f)
#     f.close()



def BackUpXconfigFile(xconfig_file, config_dir):
    # we write a copy of the xconfig file just to have a record of the original
    # input.
    try:
        xconfig_file_out = open(config_dir + '/xconfig', 'w')
    except:
        sys.exit('{0}: error opening file {1}/xconfig for output'.format(
            sys.argv[0], config_dir))
    try:
        xconfig_file_in = open(xconfig_file)
    except:
        sys.exit('{0}: error opening file {1} for input'.format(sys.argv[0], config_dir))

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


# This functions writes config_dir/xconfig.expanded.1 and
# config_dir/xconfig.expanded.2, showing some of the internal stages of
# processing the xconfig file before turning it into config files.
def WriteExpandedXconfigFiles(config_dir, all_layers):
    try:
        xconfig_file_out = open(config_dir + '/xconfig.expanded.1', 'w')
    except:
        sys.exit('{0}: error opening file {1}/xconfig.expanded.1 for output'.format(
            sys.argv[0], config_dir))


    print('# This file was created by the command:\n'
          '# ' + ' '.join(sys.argv) + '\n'
          '#It contains the same content as ./xconfig but it was parsed and\n'
          '#default config values were set.\n'
          '# See also ./xconfig.expanded.2\n', file=xconfig_file_out)

    for layer in all_layers:
        print(str(layer), file=xconfig_file_out)
    xconfig_file_out.close()

    try:
        xconfig_file_out = open(config_dir + '/xconfig.expanded.2', 'w')
    except:
        sys.exit('{0}: error opening file {1}/xconfig.expanded.2 for output'.format(
                sys.argv[0], config_dir))

    print('# This file was created by the command:\n'
          '# ' + ' '.join(sys.argv) + '\n'
          '# It contains the same content as ./xconfig but it was parsed,\n'
          '# default config values were set, and Descriptors (input=xxx) were normalized.\n'
          '# See also ./xconfig.expanded.1\n\n',
          file=xconfig_file_out)

    for layer in all_layers:
        layer.NormalizeDescriptors()
        print(str(layer), file=xconfig_file_out)
    xconfig_file_out.close()




# This function returns a map from config-file basename
# e.g. 'init', 'ref', 'layer1' to a documentation string that goes
# at the top of the file.
def GetConfigHeaders():
    ans = defaultdict(str)  # resulting dict will default to the empty string
                            # for any config files not explicitly listed here.
    ans['init'] = ('# This file was created by the command:\n'
                   '# ' + ' '.join(sys.argv) + '\n'
                   '# It contains the input of the network and is used in\n'
                   '# accumulating stats for an LDA-like transform of the\n'
                   '# input features.\n');
    ans['ref'] = ('# This file was created by the command:\n'
                  '# ' + ' '.join(sys.argv) + '\n'
                  '# It contains the entire neural network, but with those\n'
                  '# components that would normally require fixed vectors/matrices\n'
                  '# read from disk, replaced with random initialization\n'
                  '# (this applies to the LDA-like transform and the\n'
                  '# presoftmax-prior-scale, if applicable).  This file\n'
                  '# is used only to work out the left-context and right-context\n'
                  '# of the network.\n');
    ans['all'] = ('# This file was created by the command:\n'
                  '# ' + ' '.join(sys.argv) + '\n'
                  '# It contains the entire neural network.  It might not be used\n'
                  '# in the current scripts; it\'s provided for forward compatibility\n'
                  '# to possible future changes.\n')

    # Note: currently we just copy all lines that were going to go to 'all', into
    # 'layer1', to avoid propagating this nastiness to the code in xconfig_layers.py
    ans['layer1'] = ('# This file was created by the command:\n'
                     '# ' + ' '.join(sys.argv) + '\n'
                     '# It contains the configuration of the entire neural network.\n'
                     '# The contents are the same\n'
                     '# as \'all.config\'.  The reason this file is named this way (and\n'
                     '# that the config file `num_hidden_layers` contains 1, even though\n'
                     '# this file may really contain more than 1 hidden layer), is\n'
                     '# historical... we used to create networks by adding hidden layers\n'
                     '# one by one (discriminative pretraining), but more recently we\n'
                     '# have found that it\'s better to add them all at once.  This file\n'
                     '# exists to enable the older training scripts to work.  Note:\n'
                     '# it contains the inputs of the neural network even though it doesn\'t\n'
                     '# have to (since they are included in \'init.config\').  This will\n'
                     '# give us the flexibility to change the scripts in future.\n');
    return ans;




# This is where most of the work of this program happens.
def WriteConfigFiles(config_dir, all_layers):
    # config_basename_to_lines is map from the basename of the
    # config, as a string (i.e. 'ref', 'all', 'init') to a list of
    # strings representing lines to put in the config file.
    config_basename_to_lines = defaultdict(list)

    config_basename_to_header = GetConfigHeaders()

    for layer in all_layers:
        try:
            pairs = layer.GetFullConfig()
            for config_basename, line in pairs:
                config_basename_to_lines[config_basename].append(line)
        except Exception as e:
            print('{0}: error producing config lines from xconfig '
                     'line \'{1}\': error was: {2}'.format(sys.argv[0], str(layer),
                                                         repr(e)), file=sys.stderr)
            raise(e)

    # currently we don't expect any of the GetFullConfig functions to output to
    # config-basename 'layer1'... currently we just copy this from
    # config-basename 'all', for back-compatibility to older scripts.
    assert not 'layer1' in config_basename_to_lines
    config_basename_to_lines['layer1'] = config_basename_to_lines['all']

    for basename,lines in config_basename_to_lines.items():
        header = config_basename_to_header[basename]
        filename = '{0}/{1}.config'.format(config_dir, basename)
        try:
            f = open(filename, 'w')
            print(header, file=f)
            for line in lines:
                print(line, file=f)
            f.close()
        except Exception as e:
            print('{0}: error writing to config file {1}: error is {2}'.format(
                    sys.argv[0], filename, repr(e)), file=sys.stderr)
            raise e





def Main():
    args = GetArgs()
    BackUpXconfigFile(args.xconfig_file, args.config_dir)
    all_layers = xconfig_layers.ReadXconfigFile(args.xconfig_file)
    WriteExpandedXconfigFiles(args.config_dir, all_layers)
    WriteConfigFiles(args.config_dir, all_layers)



if __name__ == '__main__':
    Main()


# test:
# mkdir -p foo; (echo 'input dim=40 name=input'; echo 'output name=output input=Append(-1,0,1)')  >xconfig; ./xconfig_to_configs.py xconfig foo
#  mkdir -p foo; (echo 'input dim=40 name=input'; echo 'output-layer name=output dim=1924 input=Append(-1,0,1)')  >xconfig; ./xconfig_to_configs.py xconfig foo

# mkdir -p foo; (echo 'input dim=40 name=input'; echo 'relu-renorm-layer name=affine1 dim=1024'; echo 'output-layer name=output dim=1924 input=Append(-1,0,1)')  >xconfig; ./xconfig_to_configs.py xconfig foo

# mkdir -p foo; (echo 'input dim=100 name=ivector'; echo 'input dim=40 name=input'; echo 'fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=foo/bar/lda.mat'; echo 'output-layer name=output dim=1924 input=Append(-1,0,1)')  >xconfig; ./xconfig_to_configs.py xconfig foo


