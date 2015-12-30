#!/usr/bin/env python

# tdnn or RNN with 'jesus layer'

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings


parser = argparse.ArgumentParser(description="Writes config files and variables "
                                 "for TDNNs creation and training",
                                 epilog="See steps/nnet3/train_tdnn.sh for example.");
parser.add_argument("--splice-indexes", type=str,
                    help="Splice[:recurrence] indexes at each hidden layer, e.g. '-3,-2,-1,0,1,2,3 -3,0:-3 -3,0:-3 -6,-3,0:-6,-3'. "
                    "Note: recurrence indexes are optional, may not appear in 1st layer, may not include zero, and must be "
                    "either all negative or all positive for any given layer.")
parser.add_argument("--feat-dim", type=int,
                    help="Raw feature dimension, e.g. 13")
parser.add_argument("--ivector-dim", type=int,
                    help="iVector dimension, e.g. 100", default=0)
parser.add_argument("--include-log-softmax", type=str,
                    help="add the final softmax layer ", default="true", choices = ["false", "true"])
parser.add_argument("--final-layer-normalize-target", type=float,
                    help="RMS target for final layer (set to <1 if final layer learns too fast",
                    default=1.0)
parser.add_argument("--jesus-hidden-dim", type=int,
                    help="hidden dimension of Jesus layer.  Its output dim is --affine-input-dim "
                    "and its input dim is determined by --affine-output-dim together with "
                    "splicing and recurrence.", default=10000)
parser.add_argument("--jesus-output-dim", type=int,
                    help="output dimension of Jesus layer", default=1000)
parser.add_argument("--affine-output-dim", type=int,
                    help="Output dimension of affine components (their input dim "
                    "is --jesus-output-dim).", default=1000)
parser.add_argument("--include-relu", type=str,
                    help="If true, add ReLU nonlinearity after the Jesus layer",
                    default="true", choices = ["false", "true"])
parser.add_argument("--num-jesus-blocks", type=int,
                    help="number of blocks in Jesus layer.  --jesus-output-dim, "
                    "--jesus-hidden-dim and --affine-output-dim will be rounded up to "
                    "be a multiple of this.", default=100);
parser.add_argument("--clipping-threshold", type=float,
                    help="clipping threshold used in ClipGradient components (only relevant if "
                    "recurrence indexes are specified).  If clipping-threshold=0 no clipping is done",
                    default=15)
parser.add_argument("--num-targets", type=int,
                    help="number of network targets (e.g. num-pdf-ids/num-leaves)")
parser.add_argument("config_dir",
                    help="Directory to write config files and variables");

print(' '.join(sys.argv))

args = parser.parse_args()

if not os.path.exists(args.config_dir):
    os.makedirs(args.config_dir)

## Check arguments.
if args.splice_indexes is None:
    sys.exit("--splice-indexes argument is required");
if args.feat_dim is None or not (args.feat_dim > 0):
    sys.exit("--feat-dim argument is required");
if args.num_targets is None or not (args.num_targets > 0):
    sys.exit("--num-targets argument is required");
if args.num_jesus_blocks < 1:
    sys.exit("invalid --num-jesus-blocks value");
if args.jesus_output_dim % args.num_jesus_blocks != 0:
    args.jesus_output_dim += args.num_jesus_blocks - (args.jesus_output_dim % args.num_jesus_blocks)
    print('Rounding up --jesus-output-dim to {0} to be a multiple of --num-jesus-blocks={1}: ',
          args.jesus_output_dim, args.num_jesus_blocks)
if args.jesus_hidden_dim % args.num_jesus_blocks != 0:
    args.jesus_hidden_dim += args.num_jesus_blocks - (args.jesus_hidden_dim % args.num_jesus_blocks)
    print('Rounding up --jesus-hidden-dim to {0} to be a multiple of --num-jesus-blocks={1}: ',
          args.jesus_hidden_dim, args.num_jesus_blocks)
if args.affine_output_dim % args.num_jesus_blocks != 0:
    args.affine_output_dim += args.num_jesus_blocks - (args.jesus_hidden_dim % args.num_jesus_blocks)
    print('Rounding up --jesus-hidden-dim to {0} to be a multiple of --num-jesus-blocks={1}: ',
          args.affine_output_dim, args.num_jesus_blocks)

## Work out splice_array and recurrence_array,
## e.g. for
## args.splice_indexes == '-3,-2,-1,0,1,2,3 -3,0:-3 -3,0:-3 -6,-3,0:-6,-3'
## we would have
##   splice_array = [ [ -3,-2,...3 ], [-3,0] [-3,0] [-6,-3,0]
## and
##  recurrence_array = [ [], [-3], [-3], [-6,-3] ]
## Note, recurrence_array[0] must be empty; and any element of recurrence_array
## may be empty.  Also it cannot contain zeros, or both positive and negative elements
## at the same layer.
splice_array = []
recurrence_array = []
left_context = 0
right_context = 0
split_on_spaces = args.splice_indexes.split(" ");  # we already checked the string is nonempty.
if len(split_on_spaces) < 2:
    sys.exit("invalid --splice-indexes argument, too short: "
             + args.splice_indexes)
try:
    for string in split_on_spaces:
        this_layer = len(splice_array)
        split_on_colon = string.split(":")  # there will only be a colon if
                                            # there is recurrence at this layer.
        if len(split_on_colon) < 1 or len(split_on_colon) > 2 or (this_layer == 0 and len(split_on_colon) > 1):
            sys.exit("invalid --splice-indexes argument: " + args.splice_indexes)
        if len(split_on_colon) == 1:
            split_on_colon.append("")
        int_list = []
        this_splices = [ int(x) for x in split_on_colon[0].split(",") ]
        this_recurrence = [ int(x) for x in split_on_colon[1].split(",") if x ]
        splice_array.append(this_splices)
        recurrence_array.append(this_recurrence)
        if (len(this_splices) < 1):
            sys.exit("invalid --splice-indexes argument [empty splices]: " + args.splice_indexes)
        if len(this_recurrence) > 1 and this_recurrence[0] * this_recurrence[-1] <= 0:
            sys.exit("invalid --splice-indexes argument [invalid recurrence indexes; would not be computable."
                     + args.splice_indexes)
        if not this_splices == sorted(this_splices):
            sys.exit("elements of --splice-indexes must be sorted: "
                     + args.splice_indexes)
        left_context += -this_splices[0]
        right_context += this_splices[-1]
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
    # the following summarizes the structure of the layers:
    # layer1: splice + LDA-transform + affine + ReLU + renormalize
    # layerX: splice + Jesus [+ ReLU +] affine + ReLU + renormalize.

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
                input_dim, args.affine_output_dim), file=f)
        print('component-node name=affine1 component=affine1 input=lda',
              file=f)
    else:
        splices = []
        spliced_dims = []
        for offset in splice_array[l-1]:
            splices.append('Offset(renorm{0}, {1})'.format(l-1, offset))
            spliced_dims.append(args.affine_output_dim)
        # if this layer has recurrence, add a ClipGradientComponent for use
        # by its recurrent connections
        # TODO: test where it's best to have the recurrence from- maybe the output
        # of the jesus layer itself, before the affine layer?
        if len(recurrence_array[l-1]) > 0:
            print('component type=ClipGradientComponent name=clip-gradient{0} dim={1} clipping-threshold={2} '
                  'norm-based-clipping=true '.format(
                    l, args.affine_output_dim, args.clipping_threshold), file=f)
            print('component-node name=clip-gradient{0} component=clip-gradient{0} '
                  'input=renorm{0}'.format(l), file=f)
        for offset in recurrence_array[l-1]:
            splices.append('IfDefined(Offset(clip-gradient{0}, {1}))'.format(l, offset))
            spliced_dims.append(args.affine_output_dim)


        cur_input = 'Append({0})'.format(', '.join(splices))
        cur_dim = sum(spliced_dims)

        # As input to the Jesus component we'll append the spliced input and
        # recurrent input, but we first need to rearrange the dimensions so that
        # things pertaining to a particular block stay together.
        new_column_order = []
        for x in range(0, args.num_jesus_blocks):
            dim_offset = 0
            for src_splice in spliced_dims:
                src_block_size = src_splice / args.num_jesus_blocks
                for y in range(0, src_block_size):
                    new_column_order.append(dim_offset + (x * src_block_size) + y)
                dim_offset += src_splice
        if sorted(new_column_order) != range(0, sum(spliced_dims)):
            print("new_column_order is " + str(new_column_order))
            print("num_jesus_blocks is " + str(args.num_jesus_blocks))
            print("spliced_dims is " + str(spliced_dims))
            sys.exit("code error creating new column order")

        if new_column_order != range(0, sum(spliced_dims)):
            print('component name=permute{0} type=PermuteComponent new-column-order={1}'.format(
                    l, ','.join([str(x) for x in new_column_order])), file=f)
            print('component-node name=permute{0} component=permute{0} input={1}'.format(
                    l, cur_input), file=f)
            cur_input = 'permute{0}'.format(l)

        # Now add the jesus component.
        num_sub_components = (4 if args.include_relu == "true"  else 3);
        print('component name=jesus{0} type=CompositeComponent num-components={1}'.format(
                l, num_sub_components), file=f, end='')
        # print the sub-components of the CompositeComopnent on the same line.
        # this has the same effect as a sequence of components, but saves memory.
        print(" component1='type=RepeatedAffineComponent bias-stddev=0 input-dim={0} "
              "output-dim={1} num-repeats={2}'".format(cur_dim, args.jesus_hidden_dim,
                                                       args.num_jesus_blocks),
              file=f, end='')
        print(" component2='type=RectifiedLinearComponent dim={0}'".format(
                args.jesus_hidden_dim), file=f, end='')
        print(" component3='type=RepeatedAffineComponent bias-stddev=0 input-dim={0} "
              "output-dim={1} num-repeats={2}'".format(args.jesus_hidden_dim,
                                                       args.jesus_output_dim,
                                                       args.num_jesus_blocks),
              file=f, end='')
        if args.include_relu == "true":
            print(" component4='type=RectifiedLinearComponent dim={0}'".format(
                    args.jesus_output_dim), file=f, end='')
        print("", file=f) # newline.
        print('component-node name=jesus{0} component=jesus{0} input={1}'.format(
                l, cur_input), file=f)

        # Now print the affine component
        print('# Note: param-stddev in next component defaults to 1/sqrt(input-dim).', file=f)
        print('component name=affine{0} type=NaturalGradientAffineComponent '
              'input-dim={1} output-dim={2} bias-stddev=0'.
              format(l, args.jesus_output_dim, args.affine_output_dim), file=f)
        print('component-node name=affine{0} component=affine{0} input=jesus{0}'.format(
                l), file=f)

    # the ReLU after the affine
    print('component name=relu{0} type=RectifiedLinearComponent dim={1}'.format(
            l, args.affine_output_dim), file=f)
    print('component-node name=relu{0} component=relu{0} input=affine{0}'.format(l),
          file=f)
    # the renormalize component after the ReLU
    print ('component name=renorm{0} type=NormalizeComponent dim={1} target-rms={2}'.format(
            l, args.affine_output_dim,
            (1.0 if l < num_hidden_layers else args.final_layer_normalize_target)), file=f)
    print('component-node name=renorm{0} component=renorm{0} input=relu{0}'.format(
            l), file=f)

    # with each new layer we regenerate the final-affine component
    print('component name=final-affine type=NaturalGradientAffineComponent '
          'input-dim={0} output-dim={1} param-stddev=0.001 bias-stddev=0'.format(
            args.affine_output_dim, args.num_targets), file=f)
    print('component-node name=final-affine component=final-affine input=renorm{0}'.format(l),
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

    f.close()


