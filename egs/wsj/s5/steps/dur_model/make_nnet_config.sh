#!/bin/bash

# This simple script, writes a simple nnet3 config file based on some
# options for the nnet-duration model

learning_rate=0.001
natural_gradient=true        # If true, use natural gradient for the affine components
hidden_dim1=                 # The dimension of the first hidden layer
hidden_dim2=                 # The dimension of the second hidden layer
hidden_dim3=
bottleneck=true              # If true, hidden_dim2 will be small
lognormal_objective=false
threelayers=false

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "Usage: $0 [options] <input-dim> <output-dim>"
   echo "e.g.: $0 320 50"
   exit 1;
fi

feat_dim=$1
output_dim=$2

if [ -z $hidden_dim1 ]; then
  hidden_dim1=$[$feat_dim*3]
fi

if [ -z $hidden_dim2 ]; then
  if $bottleneck; then
    hidden_dim2=10
  else
    hidden_dim2=$[$output_dim*2]
  fi
fi

if $natural_gradient; then
  affine_comp="NaturalGradientAffineComponent"
else
  affine_comp="AffineComponent"
fi


if $lognormal_objective; then
  softmax_comp_line=
  softmax_node_line=
  output_line="output-node name=output input=final_affine_node objective=lognormal"
else
  softmax_comp_line="component name=softmax type=LogSoftmaxComponent dim=$output_dim"
  softmax_node_line="component-node name=softmax_node component=softmax input=final_affine_node"
  output_line="output-node name=output input=softmax_node"
fi

if $threelayers; then
  input_of_final_node=norm3_node
  if [ -z $hidden_dim3 ]; then
    hidden_dim3=100
  fi
else
  hidden_dim3=$hidden_dim2
  input_of_final_node=norm2_node
fi

printf "\
component name=affine1 type=$affine_comp learning-rate=$learning_rate \
          param-stddev=0.01 bias-stddev=0 input-dim=$feat_dim output-dim=$hidden_dim1\n\
component name=relu1 type=RectifiedLinearComponent dim=$hidden_dim1\n\
component name=norm1 type=NormalizeComponent dim=$hidden_dim1\n\
component name=affine2 type=$affine_comp learning-rate=$learning_rate \
          param-stddev=0.01 bias-stddev=0 input-dim=$hidden_dim1 output-dim=$hidden_dim2\n\
component name=relu2 type=RectifiedLinearComponent dim=$hidden_dim2\n\
component name=norm2 type=NormalizeComponent dim=$hidden_dim2\n\

component name=affine3 type=$affine_comp learning-rate=$learning_rate \
          param-stddev=0.01 bias-stddev=0 input-dim=$hidden_dim2 output-dim=$hidden_dim3\n\
component name=relu3 type=RectifiedLinearComponent dim=$hidden_dim3\n\
component name=norm3 type=NormalizeComponent dim=$hidden_dim3\n\

component name=final_affine type=$affine_comp learning-rate=$learning_rate \
          param-stddev=0.01 bias-stddev=0 input-dim=$hidden_dim3 output-dim=$output_dim\n\
$softmax_comp_line\n\
input-node name=input dim=$feat_dim\n\
component-node name=affine1_node component=affine1 input=input\n\
component-node name=relu1_node component=relu1 input=affine1_node\n\
component-node name=norm1_node component=norm1 input=relu1_node\n\
component-node name=affine2_node component=affine2 input=norm1_node\n\
component-node name=relu2_node component=relu2 input=affine2_node\n\
component-node name=norm2_node component=norm2 input=relu2_node\n\

component-node name=affine3_node component=affine3 input=norm2_node\n\
component-node name=relu3_node component=relu3 input=affine3_node\n\
component-node name=norm3_node component=norm3 input=relu3_node\n\

component-node name=final_affine_node component=final_affine input=$input_of_final_node\n\
$softmax_node_line\n\
$output_line\n"
