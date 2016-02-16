#!/usr/bin/env python

# tdnn or RNN with 'jesus layer'


# notes on jesus layer with recurrence:

#  inputs to jesus layer:
#      - for each previous layer in regular splicing, the output of dim  --jesus-forward-output-dim
#      - for each recurrent connection:
#      - direct input from the recurrence                            --jesus-direct-recurrence-dim
#      - indirect [projected] input from recurrence.                 --jesus-projected-recurrence-input-dim
#  outputs of jesus layer:
#     for all layers:
#       --jesus-forward-output-dim
#     for recurrent layers:
#       --jesus-direct-recurrence-dim
#       --jesus-projected-recurrence-output-dim


# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings


parser = argparse.ArgumentParser(description="Writes config files and variables "
                                 "for TDNNs creation and training",
                                 epilog="See steps/nnet3/train_tdnn.sh for example.");
parser.add_argument("--splice-indexes", type=str,
                    help="Splice[:recurrence] indexes at each hidden layer, e.g. '-3,-2,-1,0,1,2,3 -3,0:-3 -3,0:-3 -6,-3,0:-6,-3'. "
                    "Note: recurrence indexes are optional, may not appear in 1st layer, and must be "
                    "either all negative or all positive for any given layer.")
parser.add_argument("--feat-dim", type=int,
                    help="Raw feature dimension, e.g. 13")
parser.add_argument("--ivector-dim", type=int,
                    help="iVector dimension, e.g. 100", default=0)
parser.add_argument("--include-log-softmax", type=str,
                    help="add the final softmax layer ", default="true", choices = ["false", "true"])
parser.add_argument("--xent-regularize", type=float,
                    help="For chain models, if nonzero, add a separate output for cross-entropy "
                    "regularization (with learning-rate-factor equal to the inverse of this)",
                    default=0.0)
parser.add_argument("--use-repeated-affine", type=str,
                    help="if true use RepeatedAffineComponent, else BlockAffineComponent (i.e. no sharing)",
                    default="true", choices = ["false", "true"])
parser.add_argument("--final-layer-learning-rate-factor", type=float,
                    help="Learning-rate factor for final affine component",
                    default=1.0)
parser.add_argument("--recurrent-projection-learning-rate-factor", type=float,
                    help="Learning-rate factor for recurrent projections",
                    default=10.0)
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
parser.add_argument("--jesus-direct-recurrence-dim", type=int,
                    help="part of output dimension of Jesus layer that comes directly from "
                    "different time instance of the same Jesus layer", default=1000)
parser.add_argument("--jesus-projected-recurrence-output-dim", type=int,
                    help="part of output dimension of Jesus layer (in recurrent layers) "
                    "that is destined for projection to dimension "
                    "--jesus-projected-recurrence-input-dim", default=500)
parser.add_argument("--jesus-projected-recurrence-input-dim", type=int,
                    help="part of input dimension of Jesus layer that comes via "
                    "projection from the output of the same Jesus layer at different time",
                    default=200)
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
if args.final_hidden_dim < 0:
    args.final_hidden_dim = args.jesus_forward_input_dim

for name in [ "jesus_hidden_dim", "jesus_forward_output_dim", "jesus_forward_input_dim",
              "jesus_direct_recurrence_dim", "jesus_projected_recurrence_output_dim",
              "jesus_projected_recurrence_input_dim", "final_hidden_dim" ]:
    old_val = getattr(args, name)
    if old_val % args.num_jesus_blocks != 0:
        new_val = old_val + args.num_jesus_blocks - (old_val % args.num_jesus_blocks)
        printable_name = '--' + name.replace('_', '-')
        print('Rounding up {0} from {1} to {2} to be a multiple of --num-jesus-blocks={3}: '.format(
                printable_name, old_val, new_val, args.num_jesus_blocks))
        setattr(args, name, new_val);


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
    # the following summarizes the structure of the layers:  Here, the Jesus component includes ReLU at its input and output, and renormalize
    #   at its output after the ReLU.
    # layer1: splice + LDA-transform + affine + ReLU + renormalize
    # layerX [non-recurrent]: splice + Jesus + affine + ReLU
    # layerX [recurrent]: splice + Jesus + renormalize + split up:  -> [forward] affine + ReLU
    #                                                               -> [direct-recurrent]
    #                                                               -> [projected-recurrent, one per delay]: affine + ReLU
    # Inside the jesus component is:
    #  [permute +] ReLU + repeated-affine + ReLU + repeated-affine
    # [we make the repeated-affine the last one so we don't have to redo that in backprop].
    # We follow this with a post-jesus composite component containing the operations:
    #  [permute +] ReLU + renormalize
    # call this post-jesusN.
    # After this we use dim-range nodes to split up the output into
    # [ jesusN-forward-output, jesusN-direct-output and jesusN-projected-output ]
    # parts;
    # and nodes for the jesusN-forward-affine and jesusN-recurrent-affine-offsetN
    # and jesusN-recurrent-affine-offsetN-clip
    # computations.

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
        print('component name=relu1 type=RectifiedLinearComponent dim={1}'.format(
                l, args.jesus_forward_input_dim), file=f)
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
        for offset in splice_array[l-1]:
            # the connection from the previous layer
            if l == 2:
                splices.append('Offset(renorm1, {0})'.format(offset))
            else:
                splices.append('Offset(jesus{0}-forward-output-affine, {1})'.format(l-1, offset))
            spliced_dims.append(args.jesus_forward_input_dim)
        for offset in recurrence_array[l-1]:
            # the direct recurrence
            splices.append('IfDefined(Offset(jesus{0}-direct-output, {1}))'.format(l, offset))
            spliced_dims.append(args.jesus_direct_recurrence_dim)
            # the indirect recurrence (via projection)
            splices.append('IfDefined(Offset(jesus{0}-recurrent-affine-offset{1}-clip, {1}))'.format(l, offset))
            spliced_dims.append(args.jesus_projected_recurrence_input_dim)

        # get the input to the Jesus layer.
        cur_input = 'Append({0})'.format(', '.join(splices))
        cur_dim = sum(spliced_dims)

        this_layer_is_recurrent = (len(recurrence_array[l-1]) != 0)
        this_jesus_output_dim = args.jesus_forward_output_dim + (
            (args.jesus_projected_recurrence_output_dim +
             args.jesus_direct_recurrence_dim) if this_layer_is_recurrent else 0)

        # As input to the Jesus component we'll append the spliced input and
        # recurrent input, and the first thing inside the component that we do
        # is rearrange the dimensions so that things pertaining to a particular
        # block stay together.

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
        num_sub_components = (5 if need_input_permute_component else 4);
        print('component name=jesus{0} type=CompositeComponent num-components={1}'.format(
                l, num_sub_components), file=f, end='')
        # print the sub-components of the CompositeComopnent on the same line.
        # this CompositeComponent has the same effect as a sequence of
        # components, but saves memory.
        if need_input_permute_component:
            print(" component1='type=PermuteComponent column-map={1}'".format(
                    l, ','.join([str(x) for x in column_map])), file=f, end='')
        print(" component{0}='type=RectifiedLinearComponent dim={1}'".format(
                (2 if need_input_permute_component else 1),
                cur_dim), file=f, end='')

        if args.use_repeated_affine == "true":
            print(" component{0}='type=NaturalGradientRepeatedAffineComponent input-dim={1} output-dim={2} "
                  "num-repeats={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(
                    (3 if need_input_permute_component else 2),
                    cur_dim, args.jesus_hidden_dim,
                    args.num_jesus_blocks,
                    args.jesus_stddev_scale / math.sqrt(cur_dim / args.num_jesus_blocks),
                    0.5 * args.jesus_stddev_scale),
                  file=f, end='')
        else:
            print(" component{0}='type=BlockAffineComponent input-dim={1} output-dim={2} "
                  "num-blocks={3} param-stddev={4} bias-stddev=0'".format(
                    (3 if need_input_permute_component else 2),
                    cur_dim, args.jesus_hidden_dim,
                    args.num_jesus_blocks,
                    args.jesus_stddev_scale / math.sqrt(cur_dim / args.num_jesus_blocks)),
                  file=f, end='')


        print(" component{0}='type=RectifiedLinearComponent dim={1}'".format(
                (4 if need_input_permute_component else 3),
                args.jesus_hidden_dim), file=f, end='')



        if args.use_repeated_affine == "true":
            print(" component{0}='type=NaturalGradientRepeatedAffineComponent input-dim={1} output-dim={2} "
                  "num-repeats={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(
                    (5 if need_input_permute_component else 4),
                    args.jesus_hidden_dim,
                    this_jesus_output_dim,
                    args.num_jesus_blocks,
                    args.jesus_stddev_scale / math.sqrt(args.jesus_hidden_dim / args.num_jesus_blocks),
                    0.5 * args.jesus_stddev_scale),
                  file=f, end='')
        else:
            print(" component{0}='type=BlockAffineComponent input-dim={1} output-dim={2} "
                  "num-blocks={3} param-stddev={4} bias-stddev=0'".format(
                    (5 if need_input_permute_component else 4),
                    args.jesus_hidden_dim,
                    this_jesus_output_dim,
                    args.num_jesus_blocks,
                    args.jesus_stddev_scale / math.sqrt((args.jesus_hidden_dim / args.num_jesus_blocks))),
                  file=f, end='')

        print("", file=f) # print newline.
        print('component-node name=jesus{0} component=jesus{0} input={1}'.format(
                l, cur_input), file=f)

        # now print the post-Jesus component which consists of [permute +] ReLU
        # + renormalize.  we only need the permute component if this is a
        # recurrent layer.

        num_sub_components = (3 if this_layer_is_recurrent else 2);
        print('component name=post-jesus{0} type=CompositeComponent num-components={1}'.format(
                l, num_sub_components), file=f, end='')
        if this_layer_is_recurrent:
            column_map = []
            output_part_dims = [ args.jesus_forward_output_dim,
                                 args.jesus_direct_recurrence_dim,
                                 args.jesus_projected_recurrence_output_dim ]
            if sum(output_part_dims) != this_jesus_output_dim:
                sys.exit("code error")
            total_block_size = this_jesus_output_dim / args.num_jesus_blocks
            previous_part_dims_sum = 0
            for part_dim in output_part_dims:
                within_block_offset = previous_part_dims_sum / args.num_jesus_blocks
                within_block_dim = part_dim / args.num_jesus_blocks
                for x in range(0, args.num_jesus_blocks):
                    for y in range(0, within_block_dim):
                        column_map.append(x * total_block_size + within_block_offset + y)
                previous_part_dims_sum += part_dim
            if sorted(column_map) != range(0, this_jesus_output_dim):
                print("column_map is " + str(column_map))
                print("output_part_dims is " + str(output_part_dims))
                sys.exit("code error creating new column order")
            print(" component1='type=PermuteComponent column-map={1}'".format(
                    l, ','.join([str(x) for x in column_map ])), file=f, end='')

        # still within the post-Jesus component, print the ReLU
        print(" component{0}='type=RectifiedLinearComponent dim={1}'".format(
                (2 if this_layer_is_recurrent else 1),
                this_jesus_output_dim), file=f, end='')
        # still within the post-Jesus component, print the NormalizeComponent
        print(" component{0}='type=NormalizeComponent dim={1} '".format(
                (3 if this_layer_is_recurrent else 2),
                this_jesus_output_dim), file=f, end='')
        print("", file=f) # print newline.
        print('component-node name=post-jesus{0} component=post-jesus{0} input=jesus{0}'.format(l),
              file=f)

        if len(recurrence_array[l-1]) != 0:
            # This is a recurrent layer -> print the dim-range nodes.
            dim_offset = 0
            print('dim-range-node name=jesus{0}-forward-output input-node=post-jesus{0} '
                  'dim={1} dim-offset={2}'.format(l, args.jesus_forward_output_dim, dim_offset), file=f)
            dim_offset += args.jesus_forward_output_dim
            print('dim-range-node name=jesus{0}-direct-output input-node=post-jesus{0} '
                  'dim={1} dim-offset={2}'.format(l, args.jesus_direct_recurrence_dim, dim_offset), file=f)
            dim_offset += args.jesus_direct_recurrence_dim
            print('dim-range-node name=jesus{0}-projected-output input-node=post-jesus{0} '
                  'dim={1} dim-offset={2}'.format(l, args.jesus_projected_recurrence_output_dim,
                                                  dim_offset), file=f)
            input_to_forward_affine = 'jesus{0}-forward-output'.format(l)
        else:
            input_to_forward_affine = 'post-jesus{0}'.format(l)

        # handle the forward output, we need an affine node for this:
        cur_affine_output_dim = (args.jesus_forward_input_dim if l < num_hidden_layers else args.final_hidden_dim)
        print('component name=forward-affine{0} type=NaturalGradientAffineComponent '
              'input-dim={1} output-dim={2} bias-stddev=0'.
              format(l, args.jesus_forward_output_dim, cur_affine_output_dim), file=f)
        print('component-node name=jesus{0}-forward-output-affine component=forward-affine{0} input={1}'.format(
                l, input_to_forward_affine), file=f)
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
        for delay in recurrence_array[l-1]:
            print('component name=jesus{0}-recurrent-affine-offset{1} type=NaturalGradientAffineComponent '
                  'input-dim={2} output-dim={3} learning-rate-factor={4} param-stddev=0 bias-stddev=0 bias-mean=0.001'.
                  format(l, delay,
                         args.jesus_projected_recurrence_output_dim,
                         args.jesus_projected_recurrence_input_dim,
                         args.recurrent_projection_learning_rate_factor), file=f)
            print('component-node name=jesus{0}-recurrent-affine-offset{1} component=jesus{0}-recurrent-affine-offset{1} '
                  'input=jesus{0}-projected-output'.format(l, delay), file=f)
            print('component name=jesus{0}-recurrent-affine-offset{1}-clip type=ClipGradientComponent '
                  'dim={2} clipping-threshold={3} '.format(l, delay, args.jesus_projected_recurrence_input_dim,
                                                           args.clipping_threshold), file=f)
            print('component-node name=jesus{0}-recurrent-affine-offset{1}-clip component=jesus{0}-recurrent-affine-offset{1}-clip '
                  'input=jesus{0}-recurrent-affine-offset{1}'.format(l, delay), file=f)

        cur_output = 'jesus{0}-forward-output-affine'.format(l)


    # with each new layer we regenerate the final-affine component, with a ReLU before it
    # because the layers we printed don't end with a nonlinearity.
    print('component name=final-relu type=RectifiedLinearComponent dim={0}'.format(
            cur_affine_output_dim), file=f)
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
        print('component-node name=final-affine-xent component=final-affine-xent input=final-relu',
              file=f)
        print('component name=final-log-softmax-xent type=LogSoftmaxComponent dim={0}'.format(
                args.num_targets), file=f)
        print('component-node name=final-log-softmax-xent component=final-log-softmax-xent '
              'input=final-affine-xent', file=f)
        print('output-node name=output-xent input=final-log-softmax-xent', file=f)

    f.close()
