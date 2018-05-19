#!/bin/bash

# 1f10g is as 1f10e but futher reducing the 192's to 128's.
#  fixing that inconsequential bug in the input of tdnn5l.

# 1f10e is as 1f10d but changing all the 256's to 192's in the parts that are served
# by bypass connections.
#  note: there's an inconsequential mistake in the input of tdnn5l.


# 1f10d is as 1f10c but changing how the bypass connections work, making them
# go all the way through but with a scale of 0.5 on the previous one.  Also
# adding a previously missing sum thing for layer tdnn4.

# 1f10c is as 1f10b but adding bypass connections to the TDNN-F blocks,
# in the big dims (1024).

# 1f10b is as 1f10 but changing all 1280's to 1024.  It's a baseline for a future
# experiment.

# run_tdnn_1f10.sh is modified from 1f.sh but using a configuation from a more
# recent switchboard setup with TDNN-F, taken from 7o there (quite a lot of the
# configuration was different). I'm making it slightly smaller though, by
# changing 1280->1024 and 1536->1280.  (I will separately test the effect of
# --constrained false; I left that out here for now).

# It's better but also bigger.
# local/chain/compare_wer_general.sh exp/chain_cleaned/tdnn1f_sp_bi exp/chain_cleaned/tdnn1f10_sp_bi
# System                tdnn1f_sp_bi tdnn1f10_sp_bi
# WER on dev(orig)            8.9       8.4
# WER on dev(rescored)        8.1       7.6
# WER on test(orig)           9.1       8.4
# WER on test(rescored)       8.6       7.8
# Final train prob        -0.1026   -0.0844
# Final valid prob        -0.1031   -0.0965
# Final train prob (xent)   -1.4370   -1.0073
# Final valid prob (xent)   -1.4670   -1.0901
# Num-params                 6994800  18081056


# run_tdnn_1f.sh is like run_tdnn_1e.sh but it use 2 to 6 jobs and add proportional-shrink 20.

#exp/chain_cleaned/tdnn1e_sp_bi/: num-iters=253 nj=2..12 num-params=7.0M dim=40+100->3597 combine=-0.095->-0.095 xent:train/valid[167,252,final]=(-1.37,-1.31,-1.31/-1.47,-1.44,-1.44) logprob:train/valid[167,252,final]=(-0.087,-0.078,-0.078/-0.102,-0.099,-0.099)
#exp/chain_cleaned/tdnn1f_sp_bi/: num-iters=444 nj=2..6 num-params=7.0M dim=40+100->3603 combine=-0.114->-0.113 xent:train/valid[295,443,final]=(-1.59,-1.51,-1.49/-1.58,-1.52,-1.50) logprob:train/valid[295,443,final]=(-0.112,-0.102,-0.098/-0.122,-0.113,-0.110)

# local/chain/compare_wer_general.sh exp/chain_cleaned/tdnn1d_sp_bi exp/chain_cleaned/tdnn1e_sp_bi
# System                 tdnn1e_sp_bi   tdnn1f_sp_bi
# WER on dev(orig)            9.2           9.0
# WER on dev(rescored)        8.6           8.2
# WER on test(orig)           9.4           9.1
# WER on test(rescored)       8.9           8.7
# Final train prob         -0.0776       -0.0983
# Final valid prob         -0.0992       -0.1103
# Final train prob (xent)  -1.3110       -1.4893
# Final valid prob (xent)  -1.4353       -1.4951

## how you run this (note: this assumes that the run_tdnn.sh soft link points here;
## otherwise call it directly in its location).
# by default, with cleanup:
# local/chain/run_tdnn.sh

# without cleanup:
# local/chain/run_tdnn.sh  --train-set train --gmm tri3 --nnet3-affix "" &

# note, if you have already run the corresponding non-chain nnet3 system
# (local/nnet3/run_tdnn.sh), you may want to run with --stage 14.

# This script is like run_tdnn_1a.sh except it uses an xconfig-based mechanism
# to get the configuration.

set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
nj=30
decode_nj=30
min_seg_len=1.55
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

train_set=train_cleaned
gmm=tri3_cleaned  # the gmm for the target data
num_threads_ubm=32
nnet3_affix=_cleaned  # cleanup affix for nnet3 and chain dirs, e.g. _cleaned

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
train_stage=-10
tree_affix=  # affix for tree directory, e.g. "a" or "b", in case we change the configuration.
tdnn_affix=1f10g  #affix for TDNN directory, e.g. "a" or "b", in case we change the configuration.
common_egs_dir=  # you can set this to use previously dumped egs.
remove_egs=true

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

local/nnet3/run_ivector_common.sh --stage $stage \
                                  --nj $nj \
                                  --min-seg-len $min_seg_len \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --num-threads-ubm $num_threads_ubm \
                                  --nnet3-affix "$nnet3_affix"


gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp_comb
tree_dir=exp/chain${nnet3_affix}/tree_bi${tree_affix}
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_comb_lats
dir=exp/chain${nnet3_affix}/tdnn${tdnn_affix}_sp_bi
train_data_dir=data/${train_set}_sp_hires_comb
lores_train_data_dir=data/${train_set}_sp_comb
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb


for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 14 ]; then
  echo "$0: creating lang directory with one state per phone."
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d data/lang_chain ]; then
    if [ data/lang_chain/L.fst -nt data/lang/L.fst ]; then
      echo "$0: data/lang_chain already exists, not overwriting it; continuing"
    else
      echo "$0: data/lang_chain already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang data/lang_chain
    silphonelist=$(cat data/lang_chain/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat data/lang_chain/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >data/lang_chain/topo
  fi
fi

if [ $stage -le 15 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 100 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 16 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.
  if [ -f $tree_dir/final.mdl ]; then
    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
    exit 1;
  fi
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 4000 ${lores_train_data_dir} data/lang_chain $ali_dir $tree_dir
fi

if [ $stage -le 17 ]; then
  mkdir -p $dir

  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  opts="l2-regularize=0.004 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  linear_opts="orthonormal-constraint=-1.0 l2-regularize=0.004"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $opts dim=1024
  linear-component name=tdnn2l0 dim=128 $linear_opts input=Append(-1,0)
  linear-component name=tdnn2l dim=128 $linear_opts input=Append(-1,0)
  relu-batchnorm-dropout-layer name=tdnn2 $opts input=Append(0,1) dim=1024
  no-op-component name=tdnn2sum input=Sum(Scale(0.5,tdnn1),tdnn2)
  linear-component name=tdnn3l dim=128 $linear_opts input=Append(-1,0)
  relu-batchnorm-dropout-layer name=tdnn3 $opts dim=1024 input=Append(0,1)
  no-op-component name=tdnn3sum input=Sum(Scale(0.5,tdnn2sum), tdnn3)
  linear-component name=tdnn4l0 dim=128 $linear_opts input=Append(-1,0)
  linear-component name=tdnn4l dim=128 $linear_opts input=Append(0,1)
  relu-batchnorm-dropout-layer name=tdnn4 $opts input=Append(0,1) dim=1024
  no-op-component name=tdnn4sum input=Sum(Scale(0.5,tdnn3sum), tdnn4)
  linear-component name=tdnn5l dim=128 $linear_opts
  relu-batchnorm-dropout-layer name=tdnn5 $opts dim=1024 input=Append(0, tdnn3l)
  no-op-component name=tdnn5sum input=Sum(Scale(0.5,tdnn4sum), tdnn5)
  linear-component name=tdnn6l0 dim=128 $linear_opts input=Append(-3,0)
  linear-component name=tdnn6l dim=128 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn6 $opts input=Append(0,3) dim=1024
  no-op-component name=tdnn6sum input=Sum(Scale(0.5,tdnn5sum), tdnn6)
  linear-component name=tdnn7l0 dim=128 $linear_opts input=Append(-3,0)
  linear-component name=tdnn7l dim=128 $linear_opts input=Append(0,3)
  relu-batchnorm-dropout-layer name=tdnn7 $opts input=Append(0,3,tdnn6l,tdnn4l,tdnn2l) dim=1024
  no-op-component name=tdnn7sum input=Sum(Scale(0.5,tdnn6sum), tdnn7)
  linear-component name=tdnn8l0 dim=128 $linear_opts input=Append(-3,0)
  linear-component name=tdnn8l dim=128 $linear_opts input=Append(0,3)
  relu-batchnorm-dropout-layer name=tdnn8 $opts input=Append(0,3) dim=1024
  no-op-component name=tdnn8sum input=Sum(Scale(0.5,tdnn7sum), tdnn8)
  linear-component name=tdnn9l0 dim=128 $linear_opts input=Append(-3,0)
  linear-component name=tdnn9l dim=128 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn9 $opts input=Append(0,3,tdnn8l,tdnn6l,tdnn5l) dim=1024
  no-op-component name=tdnn9sum input=Sum(Scale(0.5,tdnn8sum), tdnn9)
  linear-component name=tdnn10l0 dim=128 $linear_opts input=Append(-3,0)
  linear-component name=tdnn10l dim=128 $linear_opts input=Append(0,3)
  relu-batchnorm-dropout-layer name=tdnn10 $opts input=Append(0,3) dim=1024
  no-op-component name=tdnn10sum input=Sum(Scale(0.5,tdnn9sum), tdnn10)
  linear-component name=tdnn11l0 dim=128 $linear_opts input=Append(-3,0)
  linear-component name=tdnn11l dim=128 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn11 $opts input=Append(0,3,tdnn10l,tdnn9l,tdnn7l) dim=1024
  no-op-component name=tdnn11sum input=Sum(Scale(0.5,tdnn10sum), tdnn11)
  linear-component name=prefinal-l dim=256 $linear_opts

  relu-batchnorm-layer name=prefinal-chain input=prefinal-l $opts dim=1024
  linear-component name=prefinal-chain-l dim=256 $linear_opts
  batchnorm-component name=prefinal-chain-batchnorm
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  relu-batchnorm-layer name=prefinal-xent input=prefinal-l $opts dim=1024
  linear-component name=prefinal-xent-l dim=256 $linear_opts
  batchnorm-component name=prefinal-xent-batchnorm
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/

fi

if [ $stage -le 18 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width 150,110,100 \
    --trainer.num-chunk-per-minibatch 128 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 6 \
    --trainer.optimization.proportional-shrink 20 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.0005 \
    --trainer.optimization.final-effective-lrate 0.00005 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir
fi



if [ $stage -le 19 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang $dir $dir/graph
fi

if [ $stage -le 20 ]; then
  rm $dir/.error 2>/dev/null || true
  for dset in dev test; do
      (
      steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $dir/graph data/${dset}_hires $dir/decode_${dset} || exit 1;
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
        data/${dset}_hires ${dir}/decode_${dset} ${dir}/decode_${dset}_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi
exit 0
