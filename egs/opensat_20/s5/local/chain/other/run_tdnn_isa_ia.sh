#!/usr/bin/env bash

set -e -o pipefail
# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
mic=ihm
nj=30
use_ihm_ali=false
train_set=train_ia
gmm=tri3  # the gmm for the target data
ihm_gmm=tri3  # the gmm for the IHM system (if --use-ihm-ali true).
nnet3_affix=_1a  # cleanup affix for nnet3 and chain dirs, e.g. _cleaned
num_epochs=10

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
train_stage=-10
tree_affix=_ia  # affix for tree directory, e.g. "a" or "b", in case we change the configuration.
tdnn_affix=_ia  #affix for TDNN directory, e.g. "a" or "b", in case we change the configuration.
common_egs_dir=  # you can set this to use previously dumped egs.
frames_per_eg=150,110,100
dropout_schedule='0,0@0.20,0.5@0.50,0'
remove_egs=true
aug_list="noise_low noise_high clean"
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

local/nnet3/multicondition/run_ivector_common.sh --stage $stage \
                                  --mic $mic \
                                  --nj $nj \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --nnet3-affix "$nnet3_affix"

gmm_dir=exp/${mic}/$gmm
ali_dir=exp/$mic/tri3_train_ali_aug_ia
lores_train_data_dir=data/$mic/${train_set}
tree_dir=exp/$mic/chain${nnet3_affix}/tree_bi${tree_affix}
lat_dir=exp/$mic/tri3_train_lats_aug_ia
dir=exp/$mic/chain${nnet3_affix}/tdnn${tdnn_affix}_aug

train_data_dir=data/$mic/${train_set}_hires
train_ivector_dir=exp/$mic/nnet3${nnet3_affix}/ivectors_${train_set}_hires

if [ $stage -le 11 ]; then
  echo "Starting SAT+FMLLR training."
  steps/align_si.sh --nj 30 --cmd "$train_cmd" \
       data/$mic/$train_set data/lang exp/$mic/tri3 $ali_dir
  nj=$(cat exp/$mic/$ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$mic/$train_set \
    data/lang exp/$mic/tri3 $lat_dir
fi

if [ $stage -le 12 ]; then
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
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >data/lang_chain/topo
  fi
fi

if [ $stage -le 13 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.
  if [ -f $tree_dir/final.mdl ]; then
    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
    exit 1;
  fi
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --leftmost-questions-truncate -1 \
      --cmd "$train_cmd" 5500 ${lores_train_data_dir} data/lang_chain $ali_dir $tree_dir
fi

xent_regularize=0.1

if [ $stage -le 14 ]; then
  mkdir -p $dir

  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  affine_opts="l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.008 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.008"
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
  relu-batchnorm-dropout-layer name=tdnn1 $affine_opts dim=1024
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=0

  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  linear-component name=prefinal-l dim=256 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1024 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1024 small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/

fi

if [ $stage -le 15 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$gpu_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width 140,100,160 \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 3000000 \
    --trainer.num-epochs 10 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 5 \
    --trainer.optimization.initial-effective-lrate 0.0005 \
    --trainer.optimization.final-effective-lrate 0.00005 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir
fi


if [ $stage -le 16 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang $dir $dir/graph
fi

if [ $stage -le 17 ]; then
    steps/nnet3/decode.sh --num-threads 4 --nj 20 --cmd "$decode_cmd" \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --online-ivector-dir exp/ihm/nnet3${nnet3_affix}/ivectors_safe_t_dev1_hires \
       $dir/graph data/safe_t_dev1_hires $dir/decode_safe_t_dev1 || exit 1;
fi
exit 0

