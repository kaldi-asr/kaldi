#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings


parser = argparse.ArgumentParser(description="Writes config files and variables "
                                 "for TDNNs creation and training",
                                 epilog="See steps/nnet3/train_tdnn.sh for example.");
parser.add_argument("--splice-indexes", type=str,
                    help="Splice indexes at each hidden layer, e.g. '-3:-2:-1:0:1:2:3 0 -2:2 0 -4:4 0 -8:8'")
parser.add_argument("--x-expand", type=int,
                    help="If >1, number of times to expand the network internally using the x index.",
                    default=1);
parser.add_argument("--feat-dim", type=int,
                    help="Raw feature dimension, e.g. 13")
parser.add_argument("--ivector-dim", type=int,
                    help="iVector dimension, e.g. 100", default=0)
parser.add_argument("--include-log-softmax", type=str,
                    help="add the final softmax layer ", default="true", choices = ["false", "true"])
parser.add_argument("--final-layer-normalize-target", type=float,
                    help="RMS target for final layer (set to <1 if final layer learns too fast",
                    default=1.0)
parser.add_argument("--pnorm-input-dim", type=int,
                    help="input dimension to p-norm nonlinearities")
parser.add_argument("--pnorm-output-dim", type=int,
                    help="output dimension of p-norm nonlinearities")
parser.add_argument("--relu-dim", type=int,
                    help="dimension of ReLU nonlinearities")
parser.add_argument("--use-presoftmax-prior-scale", type=str,
                    help="if true, a presoftmax-prior-scale is added",
                    choices=['true', 'false'], default = "true")
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
if not args.relu_dim is None:
    if not args.pnorm_input_dim is None or not args.pnorm_output_dim is None:
        sys.exit("--relu-dim argument not compatible with "
                 "--pnorm-input-dim or --pnorm-output-dim options");
    nonlin_input_dim = args.relu_dim
    nonlin_output_dim = args.relu_dim
else:
    if not args.pnorm_input_dim > 0 or not args.pnorm_output_dim > 0:
        sys.exit("--relu-dim not set, so expected --pnorm-input-dim and "
                 "--pnorm-output-dim to be provided.");
    nonlin_input_dim = args.pnorm_input_dim
    nonlin_output_dim = args.pnorm_output_dim

if args.use_presoftmax_prior_scale == "true":
    use_presoftmax_prior_scale = True
else:
    use_presoftmax_prior_scale = False

## Work out splice_array e.g. for regular TDNNs,
#  splice_indexes == '-3,-2,-1,0,1,2,3 0 -2,2 0 -4,4 0 -8,8'
# we'd get the following (and it may seem to use arrays unnecessarily):
#   splice_array = [ [ [-3],[-2],...[3] ], [[0]], [[-2],[2]], [[0]], [[-4],[4]], [[0]], [ [-8],[8] ] ]
#  For 'generalized' TDNNs, we make use of the deeply nested list, where
#  each sub-list, where it replaces an integer, represents a list of
#  time-offsets that we're doing a weighted sum over, for instance, if:
#  splice_indexes == '-3,-2,-1,0,1,2,3 0 -6/-3/0/3,-3/0/3/6 0 ' #HERE
#  = [ [ [-3],[-2],...[3] ], [[0]], [[-6,-3,0,3],[-3,0,3,6]], [[0]] ]

splice_array = []
left_context = 0
right_context = 0
split1 = args.splice_indexes.split(" ");  # we already checked the string is nonempty.
if len(split1) < 1:
    sys.exit("invalid --splice-indexes argument, too short: "
             + args.splice_indexes)
try:
    for string in split1:
        split2 = string.split(",")
        if len(split2) < 1:
            sys.exit("invalid --splice-indexes argument, too-short element: "
                     + args.splice_indexes)
        # int_array_list will contain a list of sub-lists, where in
        # regular TDNNs, each sub-list has length one.
        int_array_list = []
        for int_array_str in split2:
            sub_list = int_array_str.split("/");
            int_array_list.append([ int(x) for x in sub_list])
            if len(splice_array) == 0 and len(sub_list) != 1:
                sys.exit("invalid --splice-indexes argument (first splicing is not simple): " +
                         args.splice_indexes)


        # update left_context and right_context
        sorted_list = []
        map(sorted_list.extend, int_array_list);  # flatten to sorted_list
        sorted_list = sorted(sorted_list);
        left_context += -sorted_list[0]
        right_context += sorted_list[-1]
        splice_array.append(int_array_list)
except ValueError as e:
    sys.exit("invalid --splice-indexes argument " + args.splice_indexes + e)
left_context = max(0, left_context)
right_context = max(0, right_context)
num_hidden_layers = len(splice_array)
input_dim = len(splice_array[0]) * args.feat_dim  +  args.ivector_dim

f = open(args.config_dir + "/vars", "w")
print('left_context=' + str(left_context), file=f)
print('right_context=' + str(right_context), file=f)
# the initial l/r contexts are actually not needed.
# print('initial_left_context=' + str(splice_array[0][0]), file=f)
# print('initial_right_context=' + str(splice_array[0][-1]), file=f)
print('num_hidden_layers=' + str(num_hidden_layers), file=f)
f.close()

## Write the 'init.config' which is a network that just does the initial frame-splicing.
## We use this to accumulate stats for the LDA transform, and then add to and modify this
## network using 'layer*.config'.
f = open(args.config_dir + "/init.config", "w")
print('# Config file for initializing neural network prior to', file=f)
print('# preconditioning matrix computation', file=f)
print('input-node name=input dim=' + str(args.feat_dim), file=f)



list=[ ('Offset(input, {0})'.format(n[0]) if n[0] != 0 else 'input' ) for n in splice_array[0] ]
if args.ivector_dim > 0:
    print('input-node name=ivector dim=' + str(args.ivector_dim), file=f)
    list.append('ReplaceIndex(ivector, t, 0)')
# example of next line:
# output-node name=output input="Append(Offset(input, -3), Offset(input, -2), Offset(input, -1), ... , Offset(input, 3), ReplaceIndex(ivector, t, 0))"
print('output-node name=output input=Append({0})'.format(", ".join(list)), file=f)
f.close()

## Next create the 'layer*.config'.  These partial configs are used to add
## components and nodes incrementally to the network, as we add more layers.

for l in range(1, num_hidden_layers + 1):
    f = open(args.config_dir + "/layer{0}.config".format(l), "w")
    print('# Config file for layer {0} of the network'.format(l), file=f)
    if l == 1:
        cur_spliced_dim = input_dim
        print('component name=lda type=FixedAffineComponent matrix={0}/lda.mat'.
              format(args.config_dir), file=f)
    else:
        cur_unspliced_dim = nonlin_output_dim
        cur_spliced_dim = len(splice_array[l-1]) * cur_unspliced_dim

    print('# Note: param-stddev in next component defaults to 1/sqrt(input-dim).', file=f)
    print('component name=affine{0} type=NaturalGradientAffineComponent '
          'input-dim={1} output-dim={2} bias-stddev=0'.
        format(l, cur_spliced_dim, nonlin_input_dim), file=f)
    if args.relu_dim is not None:
        print('component name=nonlin{0} type=RectifiedLinearComponent dim={1}'.
              format(l, args.relu_dim), file=f)
    else:
        print('# In nnet3 framework, p in P-norm is always 2.', file=f)
        print('component name=nonlin{0} type=PnormComponent input-dim={1} output-dim={2}'.
              format(l, args.pnorm_input_dim, args.pnorm_output_dim), file=f)
    print('component name=renorm{0} type=NormalizeComponent dim={1} target-rms={2}'.format(
        l, nonlin_output_dim,
        (1.0 if l < num_hidden_layers else args.final_layer_normalize_target)), file=f)
    print('component name=final-affine type=NaturalGradientAffineComponent '
          'input-dim={0} output-dim={1} param-stddev=0 bias-stddev=0'.format(
          nonlin_output_dim, args.num_targets), file=f)
    # printing out the next two, and their component-nodes, for l > 1 is not
    # really necessary as they will already exist, but it doesn't hurt and makes
    # the structure clearer.
    if args.include_log_softmax == "true":
        if use_presoftmax_prior_scale :
            print('component name=final-fixed-scale type=FixedScaleComponent '
                  'scales={0}/presoftmax_prior_scale.vec'.format(
                    args.config_dir), file=f)
        print('component name=final-log-softmax type=LogSoftmaxComponent dim={0}'.format(
                args.num_targets), file=f)

    # If we have any non-simple splices, add the relevant components- we'd need
    # a PerElementScaleComponent and a SumReduceComponent for each one.
    # This will only possibly be the case for l > 1.
    for n in range(0, len(splice_array[l-1])):
        sub_array = splice_array[l-1][n]
        sub_array_length = len(sub_array)
        if sub_array_length > 1:
            # this is a non-simple-splice.
            if l == 1:
                sys.exit("code error");  # we already checked this.
            print('component name=per-element-scale-{0}-{1} type=PerElementScaleComponent '
                  'dim={2} param-stddev=1.0 param-mean=0.0'.format(
                    l, n+1, cur_unspliced_dim * sub_array_length), file=f)
            print('component name=sum-reduce-{0}-{1} type=SumReduceComponent '
                  'input-dim={2} output-dim={3}'.format(
                    l, n+1, cur_unspliced_dim * sub_array_length, cur_unspliced_dim), file=f)

    print('# Now for the network structure', file=f)
    if l == 1:
        splices = [ ('Offset(input, {0})'.format(n[0]) if n[0] != 0 else 'input') for n in splice_array[l-1] ]
        if args.ivector_dim > 0: splices.append('ReplaceIndex(ivector, t, 0)')
        orig_input='Append({0})'.format(', '.join(splices))
        # e.g. orig_input = 'Append(Offset(input, -2), ... Offset(input, 2), ivector)'
        print('component-node name=lda component=lda input={0}'.format(orig_input),
              file=f)
        cur_input='lda'
    else:
        # e.g. cur_input = 'Append(Offset(renorm1, -2), renorm1, Offset(renorm1, 2))'
        prev_output='renorm{0}'.format(l-1)
        if l == 2 and args.x_expand > 1:  # Expand on x.
            for x in range(0, args.x_expand):
                print('component name=expand-x-{0} type=NaturalGradientAffineComponent '
                      'input-dim={1} output-dim={1} bias-stddev=0'.format(
                        x, cur_unspliced_dim), file=f)
                # the input to this component is the previous output, with the x dimension offset
                # by the negative of this x value.  Since the input will only be defined for x=0
                # this only ends up getting defined for this x value; and later we rely on this.
                print('component-node name=expand-x-{0} component=expand-x-{0} input=Offset({1}, 0, {2})'.format(
                        x, prev_output, -x), file=f)
                # the use of 'Failover' in the descriptor is a trick to get it,
                # for a particular x value, to use the input processed with the
                # appropriate version of the 'expand-x' component.
                if x == 0:
                    prev_output_modified = 'expand-x-0'
                else:
                    prev_output_modified = 'Failover(expand-x-{0}, {1})'.format(
                        x, prev_output_modified);
            prev_output = prev_output_modified
        if l == num_hidden_layers and args.x_expand > 1:
            # collapse back down over the x indexes, by summing.
            offset_list = [ 'Offset({0}, 0, {1})'.format(prev_output, x) for x in range(0, args.x_expand) ]
            prev_output = 'Sum({0})'.format(', '.join(offset_list))

        splices = []
        for n in range(0, len(splice_array[l-1])):
            sub_array = splice_array[l-1][n]
            sub_array_length = len(sub_array)
            if sub_array_length <= 1:
                offset = sub_array[0];
                if offset == 0:
                    splices.append(prev_output)
                else:
                    splices.append('Offset({0}, {1})'.format(prev_output, offset))
            else:
                # we need to create two component instances: one to do per-element scaling, one for
                # sum-reduce.
                sub_splices = [ ('Offset({0}, {1})'.format(prev_output, m) if m != 0 else prev_output)
                                for m in sub_array ]
                sub_splice_input = 'Append({0})'.format(', '.join(sub_splices))
                print('component-node name=per-element-scale-{0}-{1} component=per-element-scale-{0}-{1} '
                      'input={2} '.format(l, n+1, sub_splice_input), file=f)
                print ('component-node name=sum-reduce-{0}-{1} component=sum-reduce-{0}-{1} '
                       'input=per-element-scale-{0}-{1}'.format(l, n+1), file=f)
                # after creating the appropriate component nodes, update 'splices'.
                splices.append('sum-reduce-{0}-{1}'.format(l, n+1))
        cur_input='Append({0})'.format(', '.join(splices))
    print('component-node name=affine{0} component=affine{0} input={1} '.
          format(l, cur_input), file=f)
    print('component-node name=nonlin{0} component=nonlin{0} input=affine{0}'.
          format(l), file=f)
    print('component-node name=renorm{0} component=renorm{0} input=nonlin{0}'.
          format(l), file=f)

    print('component-node name=final-affine component=final-affine input=renorm{0}'.
          format(l), file=f)

    if args.include_log_softmax == "true":
        if use_presoftmax_prior_scale:
            print('component-node name=final-fixed-scale component=final-fixed-scale input=final-affine',
                  file=f)
            print('component-node name=final-log-softmax component=final-log-softmax '
                  'input=final-fixed-scale', file=f)
        else:
            print('component-node name=final-log-softmax component=final-log-softmax '
                  'input=final-affine', file=f)
        print('output-node name=output input=final-log-softmax', file=f)
    else:
        print('output-node name=output input=final-affine', file=f)
    f.close()

# component name=nonlin1 type=PnormComponent input-dim=$pnorm_input_dim output-dim=$pnorm_output_dim
# component name=renorm1 type=NormalizeComponent dim=$pnorm_output_dim
# component name=final-affine type=NaturalGradientAffineComponent input-dim=$pnorm_output_dim output-dim=$num_leaves param-stddev=0 bias-stddev=0
# component name=final-log-softmax type=LogSoftmaxComponent dim=$num_leaves


# ## Write file $config_dir/init.config to initialize the network, prior to computing the LDA matrix.
# ##will look like this, if we have iVectors:
# input-node name=input dim=13
# input-node name=ivector dim=100
# output-node name=output input="Append(Offset(input, -3), Offset(input, -2), Offset(input, -1), ... , Offset(input, 3), ReplaceIndex(ivector, t, 0))"

# ## Write file $config_dir/layer1.config that adds the LDA matrix, assumed to be in the config directory as
# ## lda.mat, the first hidden layer, and the output layer.
# component name=lda type=FixedAffineComponent matrix=$config_dir/lda.mat
# component name=affine1 type=NaturalGradientAffineComponent input-dim=$lda_input_dim output-dim=$pnorm_input_dim bias-stddev=0
# component name=nonlin1 type=PnormComponent input-dim=$pnorm_input_dim output-dim=$pnorm_output_dim
# component name=renorm1 type=NormalizeComponent dim=$pnorm_output_dim
# component name=final-affine type=NaturalGradientAffineComponent input-dim=$pnorm_output_dim output-dim=$num_leaves param-stddev=0 bias-stddev=0
# component name=final-log-softmax type=LogSoftmax dim=$num_leaves
# # InputOf(output) says use the same Descriptor of the current "output" node.
# component-node name=lda component=lda input=InputOf(output)
# component-node name=affine1 component=affine1 input=lda
# component-node name=nonlin1 component=nonlin1 input=affine1
# component-node name=renorm1 component=renorm1 input=nonlin1
# component-node name=final-affine component=final-affine input=renorm1
# component-node name=final-log-softmax component=final-log-softmax input=final-affine
# output-node name=output input=final-log-softmax


# ## Write file $config_dir/layer2.config that adds the second hidden layer.
# component name=affine2 type=NaturalGradientAffineComponent input-dim=$lda_input_dim output-dim=$pnorm_input_dim bias-stddev=0
# component name=nonlin2 type=PnormComponent input-dim=$pnorm_input_dim output-dim=$pnorm_output_dim
# component name=renorm2 type=NormalizeComponent dim=$pnorm_output_dim
# component name=final-affine type=NaturalGradientAffineComponent input-dim=$pnorm_output_dim output-dim=$num_leaves param-stddev=0 bias-stddev=0
# component-node name=affine2 component=affine2 input=Append(Offset(renorm1, -2), Offset(renorm1, 2))
# component-node name=nonlin2 component=nonlin2 input=affine2
# component-node name=renorm2 component=renorm2 input=nonlin2
# component-node name=final-affine component=final-affine input=renorm2
# component-node name=final-log-softmax component=final-log-softmax input=final-affine
# output-node name=output input=final-log-softmax


# ## ... etc.  In this example it would go up to $config_dir/layer5.config.

