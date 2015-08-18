#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
import sys
import warnings

# adds the input nodes and returns the descriptor
def AddInputLayer(config_lines, feat_dim, splice_indexes = [0], ivector_dim = 0):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    output_dim = 0
    components.append('input-node name=input dim=' + str(feat_dim))
    list=[ ('Offset(input, {0})'.format(n) if n != 0 else 'input' ) for n in splice_array[0] ]
    output_dim += len(splice_array[0]) * feat_dim
    if args.ivector_dim > 0:
        components.append('input-node name=ivector dim=' + str(ivector_dim))
        list.append('ReplaceIndex(ivector, t, 0)')
        output_dim += ivector_dim
    splice_descriptor = ", ".join(list)
    return {'descriptor' : splice_descriptor,
            'dimension' : output_dim}

def AddLstmLayer(config_lines, \
                 name, input, \
                 output_dim, cell_dim, \
                 recurrent_projection_dim = 0, \
                 non_recurrent_projection_dim = 0):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    input_descriptor = input['descriptor']
    input_dim = input['dimension']
    name = name.strip()

    if (recurrent_projection_dim <= 0):
        recurrent_projection_dim = cell_dim

    # Parameter Definitions W*(* replaced by - to have valid names)
    components.add("# Input gate control : W_i* matrices")
    components.add("component name={0}_W_i-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2}".format(name, input_dim + recurrent_projection_dim, cell_dim))
    components.add("# note : the cell outputs pass through a diagonal matrix")
    components.add("component name={0}_w_ic type=PerElementScaleComponent  dim={1}".format(name, cell_dim))

    components.add("# Forget gate control : W_f* matrices")
    components.add("component name={0}_W_f-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2}".format(name, input_dim + recurrent_projection_dim, cell_dim))
    components.add("# note : the cell outputs pass through a diagonal matrix")
    components.add("component name={0}_w_fc type=PerElementScaleComponent  dim={1}".format(name, cell_dim))

    components.add("#  Output gate control : W_o* matrices")
    components.add("component name={0}_W_o-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2}".format(name, input_dim + recurrent_projection_dim, cell_dim))
    components.add("# note : the cell outputs pass through a diagonal matrix")
    components.add("component name={0}_w_oc type=PerElementScaleComponent  dim={1}".format(name, cell_dim))

    components.add("# Cell input matrices : W_c* matrices")
    components.add("component name={0}_W_c-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2}".format(input_dim + projection_dim, cell_dim))

    components.add("# projection matrices : Wrm and Wpm")
    components.add("component name={0}_W-m type=NaturalGradientAffineComponent input-dim={1} output-dim={2}".format(name, cell_dim, recurrent_projection_dim + non_recurrent_projection_dim))

    components.add("# Output : Wyr and Wyp")
    components.add("component name={0}_Wy- type=NaturalGradientAffineComponent input-dim={1} output-dim={2}".format(name, recurrent_projection_dim + non_recurrent_projection_dim, cell_dim))

    components.add("# Defining the non-linearities")
    components.add("component name={0}_i type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.add("component name={0}_f type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.add("component name={0}_o type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.add("component name={0}_g type=TanhComponent dim={1}".format(name, cell_dim))
    components.add("component name={0}_h type=TanhComponent dim={1}".format(name, cell_dim))

    components.add("# Defining the cell computations")
    components.add("component name={0}_c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.add("component name={0}_c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.add("component name={0}_m type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))

    # c1_t and c2_t defined below
    c_tminus1_descriptor = "Sum(IfDefined(Offset({0}_c1_t, -1)), IfDefined(Offset( {0}_c2_t, -1)))".format(name)

    component_nodes.add("# i_t")
    component_nodes.add("component-node name={0}_i1 component={0}_W_i-xr input=Append({1}, IfDefined(Offset({0}_r_t, -1)))".format(name, input_descriptor))
    component_nodes.add("component-node name={0}_i2 component={0}_w_ic  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.add("component-node name={0}_i_t component={0}_i input=Sum({0}_i1, {0}_i2)".format(name))

    component_nodes.add("# f_t")
    component_nodes.add("component-node name={0}_f1 component={0}_W_f-xr input=Append({1}, IfDefined(Offset({0}_r_t, -1)))".format(name, input_descriptor))
    component_nodes.add("component-node name={0}_f2 component={0}_W_fc  input={1}".format(name, c_tminus1))
    component_nodes.add("component-node name={0}_f_t component={0}_f input=Sum({0}_f1,{0}_f2)".format(name))

    component_nodes.add("# o_t")
    component_nodes.add("component-node name={0}_o1 component={0}_W_o-xr input=Append({1}, IfDefined(Offset({0}_r_t, -1)))".format(name, input_descriptor))
    component_nodes.add("component-node name={0}_o2 component={0}_W_oc input=Sum({0}_c1_t, {0}_c2_t)".format(name))
    component_nodes.add("component-node name={0}_o_t component={0}_o input=Sum({0}_o1, {0}_o2)".format(name))

    component_nodes.add("# h_t")
    component_nodes.add("component-node name={0}_h_t component={0}_h input=Sum({0}_c1_t, {0}_c2_t)".format(name))

    component_nodes.add("# g_t")
    component_nodes.add("component-node name={0}_g1 component={0}_W_c-xr input=Append({1}, IfDefined(Offset(r_t, -1)))".format(name, input_descriptor))
    component_nodes.add("component-node name={0}_g_t component={0}_g input={0}_g1".format(name))

    component_nodes.add("# parts of c_t")
    component_nodes.add("component-node name={0}_c1_t component={0}_c1  input=Append({0}_f_t, {1})".format(name, c_tminus1))
    component_nodes.add("component-node name={0}_c2_t component=c2 input=Append(i_t, g_t)\n";
#
#  // m_t
#  os << "component-node name=m_t component=m input=Append(o_t, h_t)\n";
#
#  // r_t and p_t
#  os << "component-node name=rp_t component=W-m input=m_t\n";
#  // Splitting outputs of Wy- node
#  os << "dim-range-node name=r_t input-node=rp_t dim-offset=0 "
#     << "dim=" << projection_dim << std::endl;
#
#  // y_t
#  os << "component-node name=y_t component=Wy- input=rp_t\n";
#
#  // Final affine transform
#  os << "component-node name=final_affine component=final_affine input=y_t\n";
#  os << "component-node name=posteriors component=logsoftmax input=final_affine\n";
#  os << "output-node name=output input=posteriors\n";
#  configs->push_back(os.str());



def ParseSpliceString(splice_indexes):
    ## Work out splice_array e.g. splice_array = [ [ -3,-2,...3 ], [0], [-2,2], .. [ -8,8 ] ]
    split1 = splice_indexes.split(" ");  # we already checked the string is nonempty.
    if len(split1) < 1:
        splice_indexes = "0"
    elif len(split1) > 1:
        raise ValueError("invalid --splice-indexes argument, splicing is only allowed at first layer"
                + splice_indexes)

    try:
        indexes = map(lambda x: int(x), split1[0].split(","))
        if len(indexes) < 1:
            raise ValueError("invalid --splice-indexes argument, too-short element: "
                            + splice_indexes)
        if not indexes == sorted(int_list):
            raise ValueError("elements of --splice-indexes must be sorted: "
                            + splice_indexes)
        left_context += -indexes[0]
        right_context += indexes[-1]
    except ValueError as e:
        raise ValueError("invalid --splice-indexes argument " + splice_indexes + e)

    left_context = max(0, left_context)
    right_context = max(0, right_context)

    return [left_context, right_context, indexes]

if __name__ == "__main__":
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="Writes config files and variables "
                                                 "for LSTMs creation and training",
                                     epilog="See steps/nnet3/lstm/train.sh for example.")
    parser.add_argument("--splice-indexes", type=str,
                        help="Splice indexes at input layer, e.g. '-3,-2,-1,0,1,2,3' [compulsary argument]")
    parser.add_argument("--feat-dim", type=int,
                        help="Raw feature dimension, e.g. 13")
    parser.add_argument("--ivector-dim", type=int,
                        help="iVector dimension, e.g. 100", default=0)
    parser.add_argument("--cell-dim", type=int,
                        help="dimension of lstm-cell")
    parser.add_argument("--recurrent-projection-dim", type=int,
                        help="dimension of recurrent projection")
    parser.add_argument("--non-recurrent-projection-dim", type=int,
                        help="dimension of non-recurrent projection")
    parser.add_argument("--num-targets", type=int,
                        help="number of network targets (e.g. num-pdf-ids/num-leaves)")
    parser.add_argument("config_dir",
                        help="Directory to write config files and variables")

    print(' '.join(sys.argv))

    args = parser.parse_args()

    if not os.path.exists(args.config_dir):
        os.makedirs(args.config_dir)

    ## Check arguments.
    if args.splice_indexes is None:
        sys.exit("--splice-indexes argument is required")
    if args.feat_dim is None or not (args.feat_dim > 0):
        sys.exit("--feat-dim argument is required")
    if args.num_targets is None or not (args.num_targets > 0):
        sys.exit("--feat-dim argument is required")


    [left_context, right_context, splice_indexes] = ParseSpliceString(args.splice_indexes)
    input_dim = len(splice_indexes) * args.feat_dim  +  args.ivector_dim

    # write the files used by other scripts like steps/nnet3/get_egs.sh
    f = open(args.config_dir + "/vars", "w")
    print('left_context=' + str(left_context), file=f)
    print('right_context=' + str(right_context), file=f)
    # the initial l/r contexts are actually not needed.
    # print('initial_left_context=' + str(splice_array[0][0]), file=f)
    # print('initial_right_context=' + str(splice_array[0][-1]), file=f)
    f.close()

    f = open(args.config_dir + "/init.config", "w")
    print('# Config file for initializing neural network prior to', file=f)
    print('# preconditioning matrix computation', file=f)
    config_lines = []

    AddInputLayer(feat_dim, splice_indexes = splice_indexes,
                  ivector_dim = args.ivector_dim)

    # example of next line:
    # output-node name=output input="Append(Offset(input, -3), Offset(input, -2), Offset(input, -1), ... , Offset(input, 3), ReplaceIndex(ivector, t, 0))"
    print('output-node name=output input=Append({0})'.format(", ".join(list)), file=f)
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

