#!/usr/bin/env bash

# Copyright 2017-2018  Johns Hopkins University (author: Daniel Povey)
#           2017-2018  Yiming Wang
#           2019       Fei Wu

# Based on material recipe for low-resource languages
# Factored TDNN with skip connectiong and splicing (two bottle neck layers)

#   WER results on dev
#   Model       LM          Corpus      WER(%)
#   tdnn_1a     tg_large    Combined    11.72
#   tdnn_1a     tg_small    Combined    13.61
#   tdnn_1a     tg_large    CMU_Kids    17.26
#   tdnn_1a     tg_small    CMU_Kids    26.43
#   tdnn_1a     tg_large    CSLU_Kids   10.80
#   tdnn_1a     tg_small    CSLU_Kids   12.50

# steps/info/chain_dir_info.pl exp/chain/tdnn1a_sp
# exp/chain/tdnn1a_sp/: num-iters=342 nj=2..5 num-params=17.9M dim=40+100->3192 combine=-0.042->-0.041 (over 8) xent:train/valid[227,341,final]=(-0.451,-0.363,-0.346/-0.524,-0.466,-0.434) logprob:train/valid[227,341,final]=(-0.047,-0.043,-0.042/-0.058,-0.056,-0.054) 

set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
nj=10
train_set=train
test_sets="test"
gmm=tri3       
nnet3_affix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1a  
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# training chunk-options
chunk_width=140,100,160
dropout_schedule='0,0@0.20,0.3@0.50,0'
common_egs_dir=
xent_regularize=0.1

# training options
srand=0
remove_egs=true
reporting_email=


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

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 7" if you have already
# run those things.
local/nnet3/run_ivector_common.sh\
    --stage $stage \
    --train-set $train_set \
    --test-sets $test_sets \
    --gmm $gmm \
    --nnet3_affix "$nnet3_affix" || exit 1;

gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
    [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 7 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 8 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 75 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi


if [ $stage -le 10 ]; then
    # Build a tree using our new topology.  We know we have alignments for the
    # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
    # those.  The num-leaves is always somewhat less than the num-leaves from
    # the GMM baseline.
     if [ -f $tree_dir/final.mdl ]; then
       echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
       exit 1;
    fi
    steps/nnet3/chain/build_tree.sh \
      --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 3500 ${lores_train_data_dir} \
      $lang $ali_dir $tree_dir
fi

if [ $stage -le 11 ]; then
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
    linear-component name=tdnn2l0 dim=256 $linear_opts input=Append(-1,0)
    linear-component name=tdnn2l dim=256 $linear_opts input=Append(-1,0)
    relu-batchnorm-dropout-layer name=tdnn2 $opts input=Append(0,1) dim=1024
    linear-component name=tdnn3l dim=256 $linear_opts input=Append(-1,0)
    relu-batchnorm-dropout-layer name=tdnn3 $opts dim=1024 input=Append(0,1)
    linear-component name=tdnn4l0 dim=256 $linear_opts input=Append(-1,0)
    linear-component name=tdnn4l dim=256 $linear_opts input=Append(0,1)
    relu-batchnorm-dropout-layer name=tdnn4 $opts input=Append(0,1) dim=1024
    linear-component name=tdnn5l dim=256 $linear_opts
    relu-batchnorm-dropout-layer name=tdnn5 $opts dim=1024 input=Append(0, tdnn3l)
    linear-component name=tdnn6l0 dim=256 $linear_opts input=Append(-3,0)
    linear-component name=tdnn6l dim=256 $linear_opts input=Append(-3,0)
    relu-batchnorm-dropout-layer name=tdnn6 $opts input=Append(0,3) dim=1280
    linear-component name=tdnn7l0 dim=256 $linear_opts input=Append(-3,0)
    linear-component name=tdnn7l dim=256 $linear_opts input=Append(0,3)
    relu-batchnorm-dropout-layer name=tdnn7 $opts input=Append(0,3,tdnn6l,tdnn4l,tdnn2l) dim=1024
    linear-component name=tdnn8l0 dim=256 $linear_opts input=Append(-3,0)
    linear-component name=tdnn8l dim=256 $linear_opts input=Append(0,3)
    relu-batchnorm-dropout-layer name=tdnn8 $opts input=Append(0,3) dim=1280
    linear-component name=tdnn9l0 dim=256 $linear_opts input=Append(-3,0)
    linear-component name=tdnn9l dim=256 $linear_opts input=Append(-3,0)
    relu-batchnorm-dropout-layer name=tdnn9 $opts input=Append(0,3,tdnn8l,tdnn6l,tdnn5l) dim=1024
    linear-component name=tdnn10l0 dim=256 $linear_opts input=Append(-3,0)
    linear-component name=tdnn10l dim=256 $linear_opts input=Append(0,3)
    relu-batchnorm-dropout-layer name=tdnn10 $opts input=Append(0,3) dim=1280
    linear-component name=tdnn11l0 dim=256 $linear_opts input=Append(-3,0)
    linear-component name=tdnn11l dim=256 $linear_opts input=Append(-3,0)
    relu-batchnorm-dropout-layer name=tdnn11 $opts input=Append(0,3,tdnn10l,tdnn9l,tdnn7l) dim=1024
    linear-component name=prefinal-l dim=256 $linear_opts
    
    relu-batchnorm-layer name=prefinal-chain input=prefinal-l $opts dim=1280
    linear-component name=prefinal-chain-l dim=256 $linear_opts
    batchnorm-component name=prefinal-chain-batchnorm
    output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
    
    relu-batchnorm-layer name=prefinal-xent input=prefinal-l $opts dim=1280
    linear-component name=prefinal-xent-l dim=256 $linear_opts
    batchnorm-component name=prefinal-xent-batchnorm
    output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
    
EOF

    steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
    
fi


if [ $stage -le 12 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{3,4,5,6}/$USER/kaldi-data/egs/mini_librispeech-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
    fi
    
    steps/nnet3/chain/train.py --stage=$train_stage \
      --cmd="$decode_cmd" \
      --feat.online-ivector-dir=$train_ivector_dir \
      --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
      --chain.xent-regularize $xent_regularize \
      --chain.leaky-hmm-coefficient=0.1 \
      --chain.l2-regularize=0.0 \
      --chain.apply-deriv-weights=false \
      --chain.lm-opts="--num-extra-lm-states=2000" \
      --trainer.dropout-schedule $dropout_schedule \
      --trainer.add-option="--optimization.memory-compression-level=2" \
      --trainer.srand=$srand \
      --trainer.max-param-change=2.0 \
      --trainer.num-epochs=20 \
      --trainer.frames-per-iter=3000000 \
      --trainer.optimization.num-jobs-initial=2 \
      --trainer.optimization.num-jobs-final=5 \
      --trainer.optimization.initial-effective-lrate=0.002 \
      --trainer.optimization.final-effective-lrate=0.0002 \
      --trainer.num-chunk-per-minibatch=128,64 \
      --egs.chunk-width=$chunk_width \
      --egs.dir="$common_egs_dir" \
      --egs.opts="--frames-overlap-per-eg 0" \
      --cleanup.remove-egs=$remove_egs \
      --use-gpu=true \
      --reporting.email="$reporting_email" \
      --feat-dir=$train_data_dir \
      --tree-dir=$tree_dir \
      --lat-dir=$lat_dir \
      --dir=$dir  || exit 1;
fi

if [ $stage -le 13 ]; then
    # Note: it's not important to give mkgraph.sh the lang directory with the
    # matched topology (since it gets the topology file from the model).
    utils/mkgraph.sh \
      --self-loop-scale 1.0 data/lang_test_tgsmall \
      $tree_dir $tree_dir/graph_tgsmall || exit 1;
fi

if [ $stage -le 14 ]; then
    frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
    rm $dir/.error 2>/dev/null || true

    for data in $test_sets; do
        (
        nspk=$(wc -l <data/${data}_hires/spk2utt)
        steps/nnet3/decode.sh \
            --acwt 1.0 --post-decode-acwt 10.0 \
            --frames-per-chunk $frames_per_chunk \
            --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
            --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
            $tree_dir/graph_tgsmall data/${data}_hires ${dir}/decode_tgsmall_${data} || exit 1
        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_{tgsmall,tglarge} \
        data/${data}_hires ${dir}/decode_{tgsmall,tglarge}_${data} || exit 1
        ) || touch $dir/.error &
    done
    wait
    [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

exit 0
