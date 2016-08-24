#!/usr/bin/env python

# tdnn or RNN with 'jesus layer'

#  inputs to jesus layer:
#      - for each spliced version of the previous layer the output (of dim  --jesus-forward-output-dim)

#  outputs of jesus layer:
#     for all layers:
#       --jesus-forward-output-dim


# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings
import imp

nnet3_train_lib = imp.load_source('ntl', 'steps/nnet3/nnet3_train_lib.py')
chain_lib = imp.load_source('ncl', 'steps/nnet3/chain/nnet3_chain_lib.py')

parser = argparse.ArgumentParser(description="Writes config files and variables "
                                 "for TDNNs creation and training",
                                 epilog="See steps/nnet3/train_tdnn.sh for example.");
parser.add_argument("--splice-indexes", type=str, required = True,
                    help="Splice[:recurrence] indexes at each hidden layer, e.g. '-3,-2,-1,0,1,2,3 -3,0:-3 -3,0:-3 -6,-3,0:-6,-3'. "
                    "Note: recurrence indexes are optional, may not appear in 1st layer, and must be "
                    "either all negative or all positive for any given layer.")

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

parser.add_argument("--include-log-softmax", type=str,
                    help="add the final softmax layer ", default="true", choices = ["false", "true"])
parser.add_argument("--xent-regularize", type=float,
                    help="For chain models, if nonzero, add a separate output for cross-entropy "
                    "regularization (with learning-rate-factor equal to the inverse of this)",
                    default=0.0)
parser.add_argument("--xent-separate-forward-affine", type=str,
                    help="if using --xent-regularize, gives it separate last-but-one weight matrix",
                    default="false", choices = ["false", "true"])
parser.add_argument("--use-repeated-affine", type=str,
                    help="if true use RepeatedAffineComponent, else BlockAffineComponent (i.e. no sharing)",
                    default="true", choices = ["false", "true"])
parser.add_argument("--final-layer-learning-rate-factor", type=float,
                    help="Learning-rate factor for final affine component",
                    default=1.0)
parser.add_argument("--self-repair-scale-nonlinearity", type=float,
                    help="Small scale involved in fixing derivatives, if supplied (e.g. try 0.00001)",
                    default=0.0)
parser.add_argument("--jesus-hidden-dim", type=int,
                    help="hidden dimension of Jesus layer.", default=10000)
parser.add_argument("--jesus-forward-output-dim", type=int,
                    help="part of output dimension of Jesus layer that goes to next layer",
                    default=1000)
parser.add_argument("--jesus-forward-input-dim", type=int,
                    help="Input dimension of Jesus layer that comes from affine projection "
                    "from the previous layer (same as output dim of forward affine transform)",
                    default=1000)
parser.add_argument("--final-hidden-dim", type=int,
                    help="Final hidden layer dimension-- or if <0, the same as "
                    "--jesus-forward-input-dim", default=-1)
parser.add_argument("--num-jesus-blocks", type=int,
                    help="number of blocks in Jesus layer.  All configs of the form "
                    "--jesus-*-dim will be rounded up to be a multiple of this.",
                    default=100);
parser.add_argument("--jesus-stddev-scale", type=float,
                    help="Scaling factor on parameter stddev of Jesus layer (smaller->jesus layer learns faster)",
                    default=1.0)
parser.add_argument("--clipping-threshold", type=float,
                    help="clipping threshold used in ClipGradient components (only relevant if "
                    "recurrence indexes are specified).  If clipping-threshold=0 no clipping is done",
                    default=15)
parser.add_argument("config_dir",
                    help="Directory to write config files and variables");

print(' '.join(sys.argv))

args = parser.parse_args()

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


## Check arguments.
if args.num_jesus_blocks < 1:
    sys.exit("invalid --num-jesus-blocks value");
if args.final_hidden_dim < 0:
    args.final_hidden_dim = args.jesus_forward_input_dim

for name in [ "jesus_hidden_dim", "jesus_forward_output_dim", "jesus_forward_input_dim",
              "final_hidden_dim" ]:
    old_val = getattr(args, name)
    if old_val % args.num_jesus_blocks != 0:
        new_val = old_val + args.num_jesus_blocks - (old_val % args.num_jesus_blocks)
        printable_name = '--' + name.replace('_', '-')
        print('Rounding up {0} from {1} to {2} to be a multiple of --num-jesus-blocks={3} '.format(
                printable_name, old_val, new_val, args.num_jesus_blocks))
        setattr(args, name, new_val);

# this is a bit like a struct, initialized from a string, which describes how to
# set up the statistics-pooling and statistics-extraction components.
# An example string is 'mean(-99:3:9::99)', which means, compute the mean of
# data within a window of -99 to +99, with distinct means computed every 9 frames
# (we round to get the appropriate one), and with the input extracted on multiples
# of 3 frames (so this will force the input to this layer to be evaluated
# every 3 frames).  Another example string is 'mean+stddev(-99:3:9:99)',
# which will also cause the standard deviation to be computed.
class StatisticsConfig:
    # e.g. c = StatisticsConfig('mean+stddev(-99:3:9:99)', 400, 'jesus1-forward-output-affine')
    def __init__(self, config_string, input_dim, input_name):
        self.input_dim = input_dim
        self.input_name = input_name

        m = re.search("(mean|mean\+stddev)\((-?\d+):(-?\d+):(-?\d+):(-?\d+)\)",
                      config_string)
        if m == None:
            sys.exit("Invalid splice-index or statistics-config string: " + config_string)
        self.output_stddev = (m.group(1) != 'mean')
        self.left_context = -int(m.group(2))
        self.input_period = int(m.group(3))
        self.stats_period = int(m.group(4))
        self.right_context = int(m.group(5))
        if not (self.left_context > 0 and self.right_context > 0 and
                self.input_period > 0 and self.stats_period > 0 and
                self.left_context % self.stats_period == 0 and
                self.right_context % self.stats_period == 0 and
                self.stats_period % self.input_period == 0):
            sys.exit("Invalid configuration of statistics-extraction: " + config_string)

    # OutputDim() returns the output dimension of the node that this produces.
    def OutputDim(self):
        return self.input_dim * (2 if self.output_stddev else 1)

    # OutputDims() returns an array of output dimensions, consisting of
    # [ input-dim ] if just "mean" was specified, otherwise
    # [ input-dim input-dim ]
    def OutputDims(self):
        return [ self.input_dim, self.input_dim ] if self.output_stddev else [ self.input_dim ]

    # Descriptor() returns the textual form of the descriptor by which the
    # output of this node is to be accessed.
    def Descriptor(self):
        return 'Round({0}-pooling-{1}-{2}, {3})'.format(self.input_name, self.left_context, self.right_context,
                                                       self.stats_period)

    # This function writes the configuration lines need to compute the specified
    # statistics, to the file f.
    def WriteConfigs(self, f):
        print('component name={0}-extraction-{1}-{2} type=StatisticsExtractionComponent input-dim={3} '
              'input-period={4} output-period={5} include-variance={6} '.format(
                self.input_name, self.left_context, self.right_context,
                self.input_dim, self.input_period, self.stats_period,
                ('true' if self.output_stddev else 'false')), file=f)
        print('component-node name={0}-extraction-{1}-{2} component={0}-extraction-{1}-{2} input={0} '.format(
                self.input_name, self.left_context, self.right_context), file=f)
        stats_dim = 1 + self.input_dim * (2 if self.output_stddev else 1)
        print('component name={0}-pooling-{1}-{2} type=StatisticsPoolingComponent input-dim={3} '
              'input-period={4} left-context={1} right-context={2} num-log-count-features=0 '
              'output-stddevs={5} '.format(self.input_name, self.left_context, self.right_context,
                                           stats_dim, self.stats_period,
                                           ('true' if self.output_stddev else 'false')),
              file=f)
        print('component-node name={0}-pooling-{1}-{2} component={0}-pooling-{1}-{2} input={0}-extraction-{1}-{2} '.format(
                self.input_name, self.left_context, self.right_context), file=f)




## Work out splice_array
## e.g. for
## args.splice_indexes == '-3,-2,-1,0,1,2,3 -3,0:-3 -3,0:-3 -6,-3,0:-6,-3'
## we would have
##   splice_array = [ [ -3,-2,...3 ], [-3,0] [-3,0] [-6,-3,0]


splice_array = []
left_context = 0
right_context = 0
split_on_spaces = args.splice_indexes.split(" ");  # we already checked the string is nonempty.
if len(split_on_spaces) < 2:
    sys.exit("invalid --splice-indexes argument, too short: "
             + args.splice_indexes)
try:
    for string in split_on_spaces:
        this_layer = len(splice_array)

        this_splices = string.split(",")
        splice_array.append(this_splices)
        # the rest of this block updates left_context and right_context, and
        # does some checking.
        leftmost_splice = 10000
        rightmost_splice = -10000
        for s in this_splices:
            try:
                n = int(s)
                if n < leftmost_splice:
                    leftmost_splice = n
                if n > rightmost_splice:
                    rightmost_splice = n
            except:
                if len(splice_array) == 1:
                    sys.exit("First dimension of splicing array must not have averaging [yet]")
                try:
                    x = StatisticsConfig(s, 100, 'foo')
                except:
                    sys.exit("The following element of the splicing array is not a valid specifier "
                    "of statistics: " + s)

        if leftmost_splice == 10000 or rightmost_splice == -10000:
            sys.exit("invalid element of --splice-indexes: " + string)
        left_context += -leftmost_splice
        right_context += rightmost_splice
except ValueError as e:
    sys.exit("invalid --splice-indexes argument " + args.splice_indexes + " " + str(e))
left_context = max(0, left_context)
right_context = max(0, right_context)
num_hidden_layers = len(splice_array)
input_dim = len(splice_array[0]) * args.feat_dim  +  args.ivector_dim

f = open(args.config_dir + "/vars", "w")
print('left_context=' + str(left_context), file=f)
print('right_context=' + str(right_context), file=f)
print('num_hidden_layers=' + str(num_hidden_layers), file=f)
f.close()


f = open(args.config_dir + "/init.config", "w")
print('# Config file for initializing neural network prior to', file=f)
print('# preconditioning matrix computation', file=f)
print('input-node name=input dim=' + str(args.feat_dim), file=f)
list=[ ('Offset(input, {0})'.format(n) if n != 0 else 'input' ) for n in splice_array[0] ]
if args.ivector_dim > 0:
    print('input-node name=ivector dim=' + str(args.ivector_dim), file=f)
    list.append('ReplaceIndex(ivector, t, 0)')
# example of next line:
# output-node name=output input="Append(Offset(input, -3), Offset(input, -2), Offset(input, -1), ... , Offset(input, 3), ReplaceIndex(ivector, t, 0))"
print('output-node name=output input=Append({0})'.format(", ".join(list)), file=f)
f.close()


for l in range(1, num_hidden_layers + 1):
    # the following summarizes the structure of the layers:  Here, the Jesus component includes ReLU at its input and output, and renormalize
    #   at its output after the ReLU.
    # layer1: splice + LDA-transform + affine + ReLU + renormalize
    # layerX: splice + Jesus + affine + ReLU

    # Inside the jesus component is:
    #  [permute +] ReLU + repeated-affine + ReLU + repeated-affine
    # [we make the repeated-affine the last one so we don't have to redo that in backprop].
    # We follow this with a post-jesus composite component containing the operations:
    #  [permute +] ReLU + renormalize
    # call this post-jesusN.
    # After this we use dim-range nodes to split up the output into
    # [ jesusN-forward-output, jesusN-direct-output and jesusN-projected-output ]
    # parts;
    # and nodes for the jesusN-forward-affine.

    f = open(args.config_dir + "/layer{0}.config".format(l), "w")
    print('# Config file for layer {0} of the network'.format(l), file=f)
    if l == 1:
        print('component name=lda type=FixedAffineComponent matrix={0}/lda.mat'.
              format(args.config_dir), file=f)
        splices = [ ('Offset(input, {0})'.format(n) if n != 0 else 'input') for n in splice_array[l-1] ]
        if args.ivector_dim > 0: splices.append('ReplaceIndex(ivector, t, 0)')
        orig_input='Append({0})'.format(', '.join(splices))
        # e.g. orig_input = 'Append(Offset(input, -2), ... Offset(input, 2), ivector)'
        print('component-node name=lda component=lda input={0}'.format(orig_input),
              file=f)
        # after the initial LDA transform, put a trainable affine layer and a ReLU, followed
        # by a NormalizeComponent.
        print('component name=affine1 type=NaturalGradientAffineComponent '
              'input-dim={0} output-dim={1} bias-stddev=0'.format(
                input_dim, args.jesus_forward_input_dim), file=f)
        print('component-node name=affine1 component=affine1 input=lda',
              file=f)
        # the ReLU after the affine
        print('component name=relu1 type=RectifiedLinearComponent dim={1} self-repair-scale={2}'.format(
                l, args.jesus_forward_input_dim, args.self_repair_scale_nonlinearity), file=f)
        print('component-node name=relu1 component=relu1 input=affine1', file=f)
        # the renormalize component after the ReLU
        print ('component name=renorm1 type=NormalizeComponent dim={0} '.format(
                args.jesus_forward_input_dim), file=f)
        print('component-node name=renorm1 component=renorm1 input=relu1', file=f)
        cur_output = 'renorm1'
        cur_affine_output_dim = args.jesus_forward_input_dim
    else:
        splices = []
        spliced_dims = []
        for s in splice_array[l-1]:
            # the connection from the previous layer
            try:
                offset = int(s)
                # it's an integer offset.
                splices.append('Offset({0}, {1})'.format(cur_output, offset))
                spliced_dims.append(cur_affine_output_dim)
            except:
                # it's not an integer offset, so assume it specifies the
                # statistics-extraction.
                stats = StatisticsConfig(s, cur_affine_output_dim, cur_output)
                stats.WriteConfigs(f)
                splices.append(stats.Descriptor())
                spliced_dims.extend(stats.OutputDims())

        # get the input to the Jesus layer.
        cur_input = 'Append({0})'.format(', '.join(splices))
        cur_dim = sum(spliced_dims)

        this_jesus_output_dim = args.jesus_forward_output_dim

        # As input to the Jesus component we'll append the spliced input and any
        # mean/stddev-stats input, and the first thing inside the component that
        # we do is rearrange the dimensions so that things pertaining to a
        # particular block stay together.

        column_map = []
        for x in range(0, args.num_jesus_blocks):
            dim_offset = 0
            for src_splice in spliced_dims:
                src_block_size = src_splice / args.num_jesus_blocks
                for y in range(0, src_block_size):
                    column_map.append(dim_offset + (x * src_block_size) + y)
                dim_offset += src_splice
        if sorted(column_map) != range(0, sum(spliced_dims)):
            print("column_map is " + str(column_map))
            print("num_jesus_blocks is " + str(args.num_jesus_blocks))
            print("spliced_dims is " + str(spliced_dims))
            sys.exit("code error creating new column order")

        need_input_permute_component = (column_map != range(0, sum(spliced_dims)))

        # Now add the jesus component.

        permute_offset = (1 if need_input_permute_component else 0)

        if args.jesus_hidden_dim > 0: # normal case where we have jesus-hidden-dim.
            num_sub_components = 4 + permute_offset
            hidden_else_output_dim = args.jesus_hidden_dim
        else: # no hidden part in jesus layer.
            num_sub_components = 2 + permute_offset
            hidden_else_output_dim = args.jesus_forward_output_dim
        print('component name=jesus{0} type=CompositeComponent num-components={1}'.format(
                l, num_sub_components), file=f, end='')
        # print the sub-components of the CompositeComopnent on the same line.
        # this CompositeComponent has the same effect as a sequence of
        # components, but saves memory.
        if need_input_permute_component:
            print(" component1='type=PermuteComponent column-map={1}'".format(
                    l, ','.join([str(x) for x in column_map])), file=f, end='')
        print(" component{0}='type=RectifiedLinearComponent dim={1} self-repair-scale={2}'".format(
                1 + permute_offset,
                cur_dim, args.self_repair_scale_nonlinearity), file=f, end='')

        if args.use_repeated_affine == "true":
            print(" component{0}='type=NaturalGradientRepeatedAffineComponent input-dim={1} output-dim={2} "
                  "num-repeats={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(
                    2 + permute_offset,
                    cur_dim, hidden_else_output_dim,
                    args.num_jesus_blocks,
                    args.jesus_stddev_scale / math.sqrt(cur_dim / args.num_jesus_blocks),
                    0.5 * args.jesus_stddev_scale),
                  file=f, end='')
        else:
            print(" component{0}='type=BlockAffineComponent input-dim={1} output-dim={2} "
                  "num-blocks={3} param-stddev={4} bias-stddev=0'".format(
                    2 + permute_offset,
                    cur_dim, hidden_else_output_dim,
                    args.num_jesus_blocks,
                    args.jesus_stddev_scale / math.sqrt(cur_dim / args.num_jesus_blocks)),
                  file=f, end='')

        if args.jesus_hidden_dim > 0: # normal case where we have jesus-hidden-dim.
            print(" component{0}='type=RectifiedLinearComponent dim={1} self-repair-scale={2}'".format(
                    3 + permute_offset, hidden_else_output_dim,
                    args.self_repair_scale_nonlinearity), file=f, end='')

            if args.use_repeated_affine == "true":
                print(" component{0}='type=NaturalGradientRepeatedAffineComponent input-dim={1} output-dim={2} "
                      "num-repeats={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(
                        4 + permute_offset,
                        args.jesus_hidden_dim,
                        this_jesus_output_dim,
                        args.num_jesus_blocks,
                        args.jesus_stddev_scale / math.sqrt(args.jesus_hidden_dim / args.num_jesus_blocks),
                        0.5 * args.jesus_stddev_scale),
                      file=f, end='')
            else:
                print(" component{0}='type=BlockAffineComponent input-dim={1} output-dim={2} "
                      "num-blocks={3} param-stddev={4} bias-stddev=0'".format(
                        4 + permute_offset,
                        args.jesus_hidden_dim,
                        this_jesus_output_dim,
                        args.num_jesus_blocks,
                        args.jesus_stddev_scale / math.sqrt((args.jesus_hidden_dim / args.num_jesus_blocks))),
                      file=f, end='')

        print("", file=f) # print newline.
        print('component-node name=jesus{0} component=jesus{0} input={1}'.format(
                l, cur_input), file=f)

        # now print the post-Jesus component which consists of ReLU +
        # renormalize.

        num_sub_components = 2
        print('component name=post-jesus{0} type=CompositeComponent num-components=2'.format(l),
              file=f, end='')

        # still within the post-Jesus component, print the ReLU
        print(" component1='type=RectifiedLinearComponent dim={0} self-repair-scale={1}'".format(
                this_jesus_output_dim, args.self_repair_scale_nonlinearity), file=f, end='')
        # still within the post-Jesus component, print the NormalizeComponent
        print(" component2='type=NormalizeComponent dim={0} '".format(
                this_jesus_output_dim), file=f, end='')
        print("", file=f) # print newline.
        print('component-node name=post-jesus{0} component=post-jesus{0} input=jesus{0}'.format(l),
              file=f)

        # handle the forward output, we need an affine node for this:
        cur_affine_output_dim = (args.jesus_forward_input_dim if l < num_hidden_layers else args.final_hidden_dim)
        print('component name=forward-affine{0} type=NaturalGradientAffineComponent '
              'input-dim={1} output-dim={2} bias-stddev=0'.
              format(l, args.jesus_forward_output_dim, cur_affine_output_dim), file=f)
        print('component-node name=jesus{0}-forward-output-affine component=forward-affine{0} input=post-jesus{0}'.format(
            l), file=f)
        # for each recurrence delay, create an affine node followed by a
        # clip-gradient node.  [if there are multiple recurrences in the same layer,
        # each one gets its own affine projection.]

        # The reason we set the param-stddev to 0 is out of concern that if we
        # initialize to nonzero, this will encourage the corresponding inputs at
        # the jesus layer to become small (to remove this random input), which
        # in turn will make this component learn slowly (due to small
        # derivatives).  we set the bias-mean to 0.001 so that the ReLUs on the
        # input of the Jesus layer are in the part of the activation that has a
        # nonzero derivative- otherwise with this setup it would never learn.

        cur_output = 'jesus{0}-forward-output-affine'.format(l)


    # with each new layer we regenerate the final-affine component, with a ReLU before it
    # because the layers we printed don't end with a nonlinearity.
    print('component name=final-relu type=RectifiedLinearComponent dim={0} self-repair-scale={1}'.format(
            cur_affine_output_dim, args.self_repair_scale_nonlinearity), file=f)
    print('component-node name=final-relu component=final-relu input={0}'.format(cur_output),
          file=f)
    print('component name=final-affine type=NaturalGradientAffineComponent '
          'input-dim={0} output-dim={1} learning-rate-factor={2} param-stddev=0.0 bias-stddev=0'.format(
            cur_affine_output_dim, args.num_targets,
            args.final_layer_learning_rate_factor), file=f)
    print('component-node name=final-affine component=final-affine input=final-relu',
          file=f)
    # printing out the next two, and their component-nodes, for l > 1 is not
    # really necessary as they will already exist, but it doesn't hurt and makes
    # the structure clearer.
    if args.include_log_softmax == "true":
        print('component name=final-log-softmax type=LogSoftmaxComponent dim={0}'.format(
                args.num_targets), file=f)
        print('component-node name=final-log-softmax component=final-log-softmax '
              'input=final-affine', file=f)
        print('output-node name=output input=final-log-softmax', file=f)
    else:
        print('output-node name=output input=final-affine', file=f)

    if args.xent_regularize != 0.0:
        xent_input = 'final-relu'
        if l == num_hidden_layers and args.xent_separate_forward_affine == "true":
            print('component name=forward-affine{0}-xent type=NaturalGradientAffineComponent '
                  'input-dim={1} output-dim={2} bias-stddev=0'.
                  format(l, args.jesus_forward_output_dim, args.final_hidden_dim), file=f)
            print('component-node name=jesus{0}-forward-output-affine-xent component=forward-affine{0}-xent input=post-jesus{0}'.format(
                    l), file=f)
            print('component name=final-relu-xent type=RectifiedLinearComponent dim={0} self-repair-scale={1}'.format(
                    args.final_hidden_dim, args.self_repair_scale_nonlinearity), file=f)
            print('component-node name=final-relu-xent component=final-relu-xent '
                  'input=jesus{0}-forward-output-affine-xent'.format(l), file=f)
            xent_input = 'final-relu-xent'

        # This block prints the configs for a separate output that will be
        # trained with a cross-entropy objective in the 'chain' models... this
        # has the effect of regularizing the hidden parts of the model.  we use
        # 0.5 / args.xent_regularize as the learning rate factor- the factor of
        # 1.0 / args.xent_regularize is suitable as it means the xent
        # final-layer learns at a rate independent of the regularization
        # constant; and the 0.5 was tuned so as to make the relative progress
        # similar in the xent and regular final layers.
        print('component name=final-affine-xent type=NaturalGradientAffineComponent '
              'input-dim={0} output-dim={1} param-stddev=0.0 bias-stddev=0 learning-rate-factor={2}'.format(
                cur_affine_output_dim, args.num_targets, 0.5 / args.xent_regularize), file=f)
        print('component-node name=final-affine-xent component=final-affine-xent input={0}'.format(
                xent_input), file=f)
        print('component name=final-log-softmax-xent type=LogSoftmaxComponent dim={0}'.format(
                args.num_targets), file=f)
        print('component-node name=final-log-softmax-xent component=final-log-softmax-xent '
              'input=final-affine-xent', file=f)
        print('output-node name=output-xent input=final-log-softmax-xent', file=f)

    f.close()
