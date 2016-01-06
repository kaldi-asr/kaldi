#!/usr/bin/env python

# tdnn with 'jesus layer' including block-affine connecctions.
# search below for 'notes here'.

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings

parser = argparse.ArgumentParser(description="Writes config files and variables "
                                 "for TDNNs creation and training",
                                 epilog="See steps/nnet3/train_tdnn.sh for example.");
parser.add_argument("--splice-indexes", type=str,
                    help="Splice indexes at each hidden layer, e.g. '-3,-2,-1,0,1,2,3 -3,0 -3,0 -6,-3,0'. ");
parser.add_argument("--feat-dim", type=int,
                    help="Raw feature dimension, e.g. 13")
parser.add_argument("--ivector-dim", type=int,
                    help="iVector dimension, e.g. 100", default=0)
parser.add_argument("--include-log-softmax", type=str,
                    help="add the final softmax layer ", default="true", choices = ["false", "true"])
parser.add_argument("--final-layer-target-rms", type=float,
                    help="Target RMS value prior to final affine component (e.g. set 0.5 for chain)",
                    default=1.0)
parser.add_argument("--jesus-hidden-dim", type=int,
                    help="hidden dimension of Jesus layer.", default=10000)
parser.add_argument("--jesus-full-output-dim", type=int,
                    help="part of output dimension of Jesus layer that goes via full matrix "
                    "to next layer (same as input-dim of that full matrix)", default=600)
parser.add_argument("--jesus-full-input-dim", type=int,
                    help="Input dimension of Jesus layer (per splice) that comes from full-matrix "
                    "affine projection from the previous layer (same as output dim of forward "
                    "affine transform)",  default=600)
parser.add_argument("--jesus-block-output-dim", type=int,
                    help="Output dimension of Jesus layer that goes via block-matrix affine projection "
                    "to the next layer (same as input dim of block affine transform)",
                    default=2400)
parser.add_argument("--jesus-block-input-dim", type=int,
                    help="Input dimension of Jesus layer (per splice) that comes from block-matrix "
                    "affine projection from the previous layer (same as output dim of block "
                    "affine transform)",
                    default=2400)
parser.add_argument("--jesus-final-output-dim", type=int,
                    help="Output dimension for the final Jesus layer, if >0; else "
                    "the same as jesus-full-output-dim.", default=-1);
parser.add_argument("--jesus-first-layer-input-dim", type=int,
                    help="Input dimension for the final Jesus layer, if >0; else "
                    "the same as jesus-full-input-dim.", default=-1);
parser.add_argument("--block-learning-rate-factor", type=float,
                    help="Learning-rate factor for block-diagonal affine matrices",
                    default=1.0)
parser.add_argument("--num-jesus-blocks", type=int,
                    help="number of blocks in Jesus layer and in block affine projections "
                    "between Jesus layers.  All configs of the form "
                    "--jesus-*-dim will be rounded up to be a multiple of this.",
                    default=100);
parser.add_argument("--num-affine-blocks", type=int,
                    help="number of blocks in block-affine layer between Jesus layers, if >0. "
                    "Otherwise defaults to num-jesus-blocks.  Must divide num-jesus-blocks.",
                    default=-1);
parser.add_argument("--jesus-stddev-scale", type=float,
                    help="Scaling factor on parameter stddev of Jesus layer (smaller->jesus layer learns faster)",
                    default=1.0)
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

# some defaults:
if args.jesus_first_layer_input_dim < 1:
    args.jesus_first_layer_input_dim = args.jesus_full_input_dim
if args.jesus_final_output_dim < 1:
    args.jesus_final_output_dim = args.jesus_full_output_dim
if args.num_affine_blocks <= 0:
    args.num_affine_blocks = args.num_jesus_blocks
if args.num_jesus_blocks % args.num_affine_blocks != 0:
    sys.exit("invalid --num-affine-blocks value, does not divide --num-jesus-blocks.");


for name in [ "jesus_hidden_dim", "jesus_full_output_dim", "jesus_full_input_dim",
              "jesus_block_output_dim", "jesus_block_input_dim",
              "jesus_final_output_dim", "jesus_first_layer_input_dim" ]:
    old_val = getattr(args, name)
    if old_val % args.num_jesus_blocks != 0:
        new_val = old_val + args.num_jesus_blocks - (old_val % args.num_jesus_blocks)
        printable_name = '--' + name.replace('_', '-')
        print('Rounding up {0} from {1} to {2} to be a multiple of --num-jesus-blocks={3}: '.format(
                printable_name, old_val, new_val, args.num_jesus_blocks))
        setattr(args, name, new_val);


## Work out splice_array.
## e.g. for
## args.splice_indexes == '-3,-2,-1,0,1,2,3 -3,0 -3,0 -6,-3,0'
## we would have
##   splice_array = [ [ -3,-2,...3 ], [-3,0] [-3,0] [-6,-3,0] ]
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
        this_splices = [ int(x) for x in string.split(",") ]
        splice_array.append(this_splices)
        if (len(this_splices) < 1):
            sys.exit("invalid --splice-indexes argument [empty splices]: " + args.splice_indexes)
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
    # notes here.
    # the following summarizes the structure of the layers: Here, the Jesus
    #   component includes ReLU at its input and output, and renormalize at its
    #   output after the ReLU.  Note, the Jesus layer includes permutation
    #   components.  For a reason relating to saving memory, we put the
    # ReLU + renormalize at the beginning of each layer, inside the Jesus
    # component, instead of at the end of the layers.

    # layer1: splice + LDA-transform + affine
    #
    # layerX [middle layers]: splice + Jesus + post-Jesus + [full transform; block transform]
    # layerN [last hidden layer]: Jesus + post-Jesus + full transform
    # final-layer:  ReLU + renormalize + final-affine [+ log-softmax].

    # Inside the jesus component is:
    #   [permute +] ReLU + renormalize + repeated-affine + ReLU + repeated-affine
    # Inside the post-Jesus component is:
    #  [permute +] ReLU + renormalize.
    # to save some repeated computation caused by the store-stats operation,
    # we put the final ReLU outside of the Jesus component.
    #
    # After the Jesus component we use dim-range nodes to split up the output into
    # [ jesusN-full-output, jesusN-block-output ], then project these  to
    #  jesus [ jesusN-full-affine, jesusN-block-affine ].

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
        output_dim = args.jesus_first_layer_input_dim
        print('component name=affine1 type=NaturalGradientAffineComponent '
              'input-dim={0} output-dim={1} bias-mean=0.1 bias-stddev=0'.format(
                input_dim, output_dim), file=f)
        print('component-node name=affine1 component=affine1 input=lda',
              file=f)
        print('component name=renorm1 type=NormalizeComponent dim={0}'.format(output_dim),
              file=f)
        print('component-node name=renorm1 component=renorm1 input=affine1',
              file=f)

        cur_dims = [ args.jesus_first_layer_input_dim ]
        cur_outputs = [ 'renorm1' ]
        cur_dim_if_final = args.jesus_first_layer_input_dim
        cur_output_if_final = 'renorm1'
    else:
        # Take care of splicing:
        next_outputs = []
        next_dims = []
        for offset in splice_array[l-1]:
            for s in cur_outputs:
                next_outputs.append('Offset({0}, {1})'.format(s, offset))
            for i in cur_dims:
                next_dims.append(i)
        cur_outputs = next_outputs
        cur_dims = next_dims

        # get the input to the Jesus layer.
        cur_input = 'Append({0})'.format(', '.join(cur_outputs))
        cur_dim = sum(cur_dims)

        if l == num_hidden_layers:
            this_jesus_output_dim = args.jesus_final_output_dim
        else:
            this_jesus_output_dim = args.jesus_full_output_dim + args.jesus_block_output_dim


        # As input to the Jesus component we'll append the spliced input and
        # recurrent input, and the first thing inside the component that we do
        # is rearrange the dimensions so that things pertaining to a particular
        # block stay together.
        column_map = []
        for x in range(0, args.num_jesus_blocks):
            dim_offset = 0
            for src_dim in cur_dims:
                src_block_size = src_dim / args.num_jesus_blocks
                for y in range(0, src_block_size):
                    column_map.append(dim_offset + (x * src_block_size) + y)
                dim_offset += src_dim
        if sorted(column_map) != range(0, sum(cur_dims)):
            print("column_map is " + str(column_map))
            print("num_jesus_blocks is " + str(args.num_jesus_blocks))
            print("cur_dims is " + str(cur_dims))
            sys.exit("code error creating new column order")

        need_input_permute_component = (column_map != range(0, sum(cur_dims)))

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

        print(" component{0}='type=RepeatedAffineComponent input-dim={1} output-dim={2} "
              "num-repeats={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(
                (3 if need_input_permute_component else 2),
                cur_dim, args.jesus_hidden_dim,
                args.num_jesus_blocks,
                args.jesus_stddev_scale / math.sqrt(cur_dim / args.num_jesus_blocks),
                0.5 * args.jesus_stddev_scale),
              file=f, end='')

        print(" component{0}='type=RectifiedLinearComponent dim={1}'".format(
                (4 if need_input_permute_component else 3),
                args.jesus_hidden_dim), file=f, end='')

        print(" component{0}='type=RepeatedAffineComponent input-dim={1} output-dim={2} "
              "num-repeats={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(
                (5 if need_input_permute_component else 4),
                args.jesus_hidden_dim,
                this_jesus_output_dim,
                args.num_jesus_blocks,
                args.jesus_stddev_scale / math.sqrt(args.jesus_hidden_dim / args.num_jesus_blocks),
                0.5 * args.jesus_stddev_scale),
              file=f, end='')

        print("", file=f) # print newline.
        print('component-node name=jesus{0} component=jesus{0} input={1}'.format(
                l, cur_input), file=f)

        # now print the post-Jesus component which consists of [permute +] ReLU
        # + renormalize.  we only need the permute component if this is a
        # recurrent layer.

        num_sub_components = (3 if l < num_hidden_layers else 2);
        print('component name=post-jesus{0} type=CompositeComponent num-components={1}'.format(
                l, num_sub_components), file=f, end='')
        if l < num_hidden_layers:
            column_map = []
            output_part_dims = [ args.jesus_full_output_dim,
                                 args.jesus_block_output_dim ]
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
                (2 if l < num_hidden_layers else 1),
                this_jesus_output_dim), file=f, end='')
        # still within the post-Jesus component, print the NormalizeComponent
        print(" component{0}='type=NormalizeComponent dim={1} '".format(
                (3 if l < num_hidden_layers else 2),
                this_jesus_output_dim), file=f, end='')
        print("", file=f) # print newline.
        print('component-node name=post-jesus{0} component=post-jesus{0} input=jesus{0}'.format(l),
              file=f)

        if l < num_hidden_layers:
            # This is a non-final hidden layer -> print the dim-range nodes for the full
            # and block affine transforms, and then the affine components.
            dim_offset = 0
            print('dim-range-node name=jesus{0}-full-output input-node=post-jesus{0} '
                  'dim={1} dim-offset={2}'.format(l, args.jesus_full_output_dim,
                                                  dim_offset), file=f)
            dim_offset += args.jesus_full_output_dim
            print('dim-range-node name=jesus{0}-block-output input-node=post-jesus{0} '
                  'dim={1} dim-offset={2}'.format(l, args.jesus_block_output_dim,
                                                  dim_offset), file=f)

            # print an affine node for the full output.
            print('component name=jesus{0}-full-affine type=NaturalGradientAffineComponent '
                  'input-dim={1} output-dim={2} bias-mean=0.1 bias-stddev=0'.
                  format(l, args.jesus_full_output_dim, args.jesus_full_input_dim), file=f)
            print('component-node name=jesus{0}-full-affine component=jesus{0}-full-affine '
                  'input=jesus{0}-full-output'.format(l), file=f)
            # ... and print a block-affine node for the block output
            print('component name=jesus{0}-block-affine type=BlockAffineComponent '
                  'learning-rate-factor={1} input-dim={2} output-dim={3} num-blocks={4} bias-mean=0.1 bias-stddev=0'.
                  format(l, args.block_learning_rate_factor,
                         args.jesus_block_output_dim, args.jesus_block_input_dim,
                         args.num_affine_blocks), file=f)
            print('component-node name=jesus{0}-block-affine component=jesus{0}-block-affine '
                  'input=jesus{0}-block-output'.format(l), file=f)
            cur_dims = [ args.jesus_full_input_dim, args.jesus_block_input_dim ]
            cur_outputs = [ 'jesus{0}-full-affine'.format(l), 'jesus{0}-block-affine'.format(l) ]
            # for producing the temporary final node for discriminative pretraining,
            # only use the full-affine part.  This will create warnings about orphan components.
            cur_dim_if_final =  args.jesus_full_input_dim
            cur_output_if_final = 'jesus{0}-full-affine'.format(l)
        else:
            # this is the last hidden layer -> we don't have dim-range nodes.
            # print an affine node for the full output.  We're hardcoding this
            # to have the same input and output dims for now.
            print('component name=jesus{0}-full-affine type=NaturalGradientAffineComponent '
                  'input-dim={1} output-dim={1} bias-mean=0.1 bias-stddev=0'.
                  format(l, args.jesus_final_output_dim), file=f)
            print('component-node name=jesus{0}-full-affine component=jesus{0}-full-affine '
                  'input=post-jesus{0}'.format(l), file=f)
            # only the final layer goes after this so no need for cur_dims and cur_outputs.
            cur_dim_if_final =  args.jesus_final_output_dim
            cur_output_if_final = 'jesus{0}-full-affine'.format(l)


    print('component name=final-relu type=RectifiedLinearComponent dim={0}'.format(
            cur_dim_if_final), file=f)
    print('component-node name=final-relu component=final-relu input={0}'.format(cur_output_if_final),
          file=f)
    print('component name=final-normalize type=NormalizeComponent dim={0} target-rms={1}'.format(
            cur_dim_if_final, args.final_layer_target_rms), file=f)
    print('component-node name=final-normalize component=final-normalize input=final-relu',
          file=f)
    print('component name=final-affine type=NaturalGradientAffineComponent '
          'input-dim={0} output-dim={1} param-stddev=0.0 bias-stddev=0'.format(
            cur_dim_if_final, args.num_targets), file=f)
    print('component-node name=final-affine component=final-affine input=final-normalize',
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
