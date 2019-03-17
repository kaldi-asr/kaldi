#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

set -e
set -o pipefail

# This is fisher chain recipe for training a model on a subset of around
# 100-300 hours of supervised data.
# This system uses phone LM to model UNK.
# local/semisup/run_50k.sh and local/semisup/run_100k.sh show how to call this.

# train_set                 train_sup15k           train_sup50k           train_sup
# ivector_train_set         semisup15k_100k_250k   semisup50k_100k_250k   train_sup
# WER on dev                27.75                  21.41                  19.23
# WER on test               27.24                  21.03                  19.01
# Final train prob          -0.0959                -0.1035                -0.1224
# Final valid prob          -0.1823                -0.1667                -0.1503
# Final train prob (xent)   -1.9246                -1.5926                -1.6454
# Final valid prob (xent)   -2.1873                -1.7990                -1.7107

# train_set                           semisup15k_100k_250k    semisup50k_100k_250k    semisup100k_250k
# ivector_train_set                   semisup15k_100k_250k    semisup50k_100k_250k    train_sup
# WER on dev                          17.92                   17.55                   16.97
# WER on test                         17.95                   17.72                   17.03
# Final output train prob             -0.1145                 -0.1155                 -0.1196
# Final output valid prob             -0.1370                 -0.1510                 -0.1469
# Final output train prob (xent)      -1.7449                 -1.7458                 -1.5487
# Final output valid prob (xent)      -1.7785                 -1.9045                 -1.6360

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
exp_root=exp/semisup_100k

nj=30
tdnn_affix=_1a
train_set=train_sup
ivector_train_set=   # dataset for training i-vector extractor

nnet3_affix=  # affix for nnet3 dir -- relates to i-vector used
chain_affix=  # affix for chain dir
tree_affix=bi_a
gmm=tri4a  # Expect GMM model in $exp/$gmm for alignment

# Neural network opts
xent_regularize=0.1
hidden_dim=725

# training options
num_epochs=4

remove_egs=false
common_egs_dir=   # if provided, will skip egs generation
common_treedir=   # if provided, will skip the tree building stage

decode_iter=

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

gmm_dir=$exp_root/$gmm   # used to get training lattices (for chain supervision)
treedir=$exp_root/chain${chain_affix}/tree_${tree_affix}
lat_dir=$exp_root/chain${chain_affix}/${gmm}_${train_set}_sp_unk_lats  # training lattices directory
dir=$exp_root/chain${chain_affix}/tdnn${tdnn_affix}_sp
train_data_dir=data/${train_set}_sp_hires
train_ivector_dir=$exp_root/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
lang=data/lang_chain_unk

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.
local/nnet3/run_ivector_common.sh --stage $stage --exp-root $exp_root \
                                  --speed-perturb true \
                                  --train-set $train_set \
                                  --ivector-train-set "$ivector_train_set" \
                                  --nnet3-affix "$nnet3_affix" || exit 1

if [ "$train_set" != "$ivector_train_set" ]; then
  if [ $stage -le 9 ]; then
    # We extract iVectors on all the ${train_set} data, which will be what we
    # train the system on.
    # having a larger number of speakers is helpful for generalization, and to
    # handle per-utterance decoding well (iVector starts at zero).
    utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
      data/${train_set}_sp_hires data/${train_set}_sp_max2_hires

    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
      data/${train_set}_sp_max2_hires $exp_root/nnet3${nnet3_affix}/extractor \
      $exp_root/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires || exit 1;
  fi
fi

if [ $stage -le 10 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" \
    --generate-ali-from-lats true data/${train_set}_sp \
    data/lang_unk $gmm_dir $lat_dir || exit 1;
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 11 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang_unk $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ -z "$common_treedir" ]; then
  if [ $stage -le 12 ]; then
    # Build a tree using our new topology.
    steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
        --context-opts "--context-width=2 --central-position=1" \
        --cmd "$train_cmd" 7000 data/${train_set}_sp $lang $lat_dir $treedir || exit 1
  fi
else
  treedir=$common_treedir
fi

if [ $stage -le 13 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=$hidden_dim
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1,2) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn3 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn6 input=Append(-6,-3,0) dim=$hidden_dim

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain input=tdnn6 dim=$hidden_dim target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn6 dim=$hidden_dim target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 14 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/fisher_english-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  mkdir -p $dir/egs
  touch $dir/egs/.nodelete # keep egs around when that run dies.

  steps/nnet3/chain/train.py --stage $train_stage \
    --egs.dir "$common_egs_dir" \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize 0.1 \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --generate-egs-scp true" \
    --egs.chunk-width 160,140,110,80 \
    --trainer.num-chunk-per-minibatch 128 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $treedir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;
fi

graph_dir=$dir/graph_poco_unk
if [ $stage -le 15 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test_poco_unk $dir $graph_dir
fi

decode_suff=
if [ $stage -le 16 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in dev test; do
      (
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $num_jobs --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir $exp_root/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
          $graph_dir data/${decode_set}_hires $dir/decode_poco_unk_${decode_set}${decode_iter:+_$decode_iter}${decode_suff} || exit 1;
      ) &
  done
fi
wait;
exit 0;
