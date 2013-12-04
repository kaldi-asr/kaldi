#!/bin/bash

# Copyright 2012  Karel Vesely (Brno University of Technology)
# Apache 2.0

# Initialize neural network

# Begin configuration.
model_size=8000000 # nr. of parameteres in MLP
hid_layers=4       # nr. of hidden layers (prior to sotfmax or bottleneck)
bn_dim=            # set value to get a bottleneck network
hid_dim=           # set value to override the $model_size
seed=777           # seed for the initialization 
init_opts="--gauss --negbias"
# End configuration.

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh; 

. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 <in-dim> <out-dim> <nnet-init>"
   echo " e.g.: $0 400 3000 nnet.init"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   echo "  --model-size <N>        # number of weights in the nnet"
   echo "  --hid-layers <N>        # number of hidden layers"
   echo "  --bn-dim <N>            # dim of linear bottleneck"
   echo "  --hid-dim <N>           # dim of hidden layers (overrides --model-size)"
   exit 1;
fi

in_dim=$1
out_dim=$2
nnet_out=$3
dir=$(dirname $nnet_out)
[ ! -d $dir ] && mkdir -p $dir

###
### What is the topology? Straight or bottleneck?
###
if [ -z "$bn_dim" ]; then #MLP w/o bottleneck
  case "$hid_layers" in
    0) #just logistic regresion
      mlp_init=$dir/nnet_${in_dim}_${out_dim}.init
      echo "Initializing MLP : $mlp_init"
      utils/nnet/gen_mlp_init.py --dim=${in_dim}:${out_dim} \
        ${init_opts} --seed=$seed > $mlp_init || exit 1;
      ;;
    1) #MLP with one hidden layer
      if [ -z "$hid_dim" ]; then
        hid_dim=$((model_size/(in_dim+out_dim)))
      fi
      mlp_init=$dir/nnet_${in_dim}_${hid_dim}_${out_dim}.init
      echo "Initializing MLP : $mlp_init"
      utils/nnet/gen_mlp_init.py --dim=${in_dim}:${hid_dim}:${out_dim} \
        ${init_opts} --seed=$seed > $mlp_init || exit 1;
      ;;
    2|3|4|5|6|7|8|9|10) #MLP with more than 1 hidden layer
      if [ -z "$hid_dim" ]; then
        a=$((hid_layers-1))
        b=$((in_dim+out_dim))
        c=$((-model_size))
        hid_dim=$(awk "BEGIN{ hid_dim= -$b/(2*$a) + sqrt($b^2 -4*$a*$c)/(2*$a); print int(hid_dim) }") 
      fi
      #build the mlp name mlp_init and dim argument dim_arg
      mlp_init=
      dim_arg=
      { 
        mlp_init=$dir/nnet_${in_dim}
        dim_arg=${in_dim}
        for i in $(seq $hid_layers); do
          mlp_init=${mlp_init}_$hid_dim
          dim_arg=${dim_arg}:${hid_dim}
        done
        mlp_init=${mlp_init}_${out_dim}.init
        dim_arg=${dim_arg}:${out_dim}
      }
      echo "Initializing MLP : $mlp_init"
      utils/nnet/gen_mlp_init.py --dim=${dim_arg} ${init_opts} \
        --seed=$seed > $mlp_init || exit 1;
      ;;
    *)
      echo "Unsupported number of hidden layers $hid_layers"
      exit 1;
  esac
else #MLP with bottleneck
  bn_dim=$bn_dim
  case "$hid_layers" in # ie. number of layers in front of bottleneck
    1) #1-hidden layer in front of the bottleneck
      if [ -z "$hid_dim" ]; then
        hid_dim=$((model_size/(in_dim+out_dim+(2*bn_dim))))
      fi
      mlp_init=$dir/nnet_${in_dim}_${hid_dim}_${bn_dim}_${hid_dim}_${out_dim}.init
      echo "Initializing MLP : $mlp_init"
      utils/nnet/gen_mlp_init.py --dim=${in_dim}:${hid_dim}:${bn_dim}:${hid_dim}:${out_dim} \
        ${init_opts} --seed=$seed --linBNdim=$bn_dim > $mlp_init || exit 1;
      ;;
    2|3|4|5|6|7|8|9|10) #more than 1 hidden layer in front of bottleneck
      if [ -z "$hid_dim" ]; then
        a=$((hid_layers-1))
        b=$((in_dim+2*bn_dim+out_dim))
        c=$((-model_size))
        hid_dim=$(awk "BEGIN{ hid_dim= -$b/(2*$a) + sqrt($b^2 -4*$a*$c)/(2*$a); print int(hid_dim) }") 
      fi
      #build the nnet name mlp_init and dim agument dim_arg
      mlp_init=
      dim_arg=
      { 
        mlp_init=$dir/nnet_${in_dim}
        dim_arg=${in_dim}
        for i in $(seq $hid_layers); do
          mlp_init=${mlp_init}_$hid_dim
          dim_arg=${dim_arg}:${hid_dim}
        done
        mlp_init=${mlp_init}_${bn_dim}lin_${hid_dim}_${out_dim}.init
        dim_arg=${dim_arg}:${bn_dim}:${hid_dim}:${out_dim}
      }
      echo "Initializing MLP : $mlp_init"
      utils/nnet/gen_mlp_init.py --dim=${dim_arg} ${init_opts} \
        --seed=$seed --linBNdim=$bn_dim > $mlp_init || exit 1;
      ;;
    *)
      echo "Unsupported number of hidden layers $hid_layers"
      exit 1;
  esac
fi

#The output name same as the mlp name, we are done..
[ $nnet_out == $mlp_init ] && "Successfuly created '$nnet_out'" && exit 0;

#Or we need to link the destination file
#(we want to keep the name showing the topology)
([ -f $nnet_out ] && unlink $nnet_out; cd $dir; ln -s $(basename $mlp_init) $(basename $nnet_out))

echo "Successfuly created linked '$nnet_out'"
