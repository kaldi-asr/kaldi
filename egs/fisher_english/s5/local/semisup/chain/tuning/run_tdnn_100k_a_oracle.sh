#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

set -e

# This is an oracle experiment using oracle transcription of 250 hours of 
# unsupervised data, along with 100 hours of supervised data.
# We train the i-vector extractor only on the supervised data, for 
# fair comparison with unsupervised data experiments.

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
tdnn_affix=7a_oracle
exp=exp/semisup_100k

# Datasets -- Expects data/$supervised_set and data/$unsupervised_set to be 
# present
supervised_set=train_sup
unsupervised_set=train_unsup100k_250k_n10k
combined_train_set=train_oracle100k_250k_n10k

# Seed model options
nnet3_affix=
chain_affix=
tree_affix=bi_a
train_supervised_opts="--stage -10 --train-stage -10"
gmm=tri4a

# Neural network opts
xent_regularize=0.1
hidden_dim=725

# training options
num_epochs=4
remove_egs=false
common_egs_dir=

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

gmm_dir=$exp/$gmm   # used to get training lattices (for chain supervision)
treedir=$exp/chain${chain_affix}/tree_${tree_affix}
lat_dir=$exp/chain${chain_affix}/${gmm}_${combined_train_set}_sp_lats  # training lattices directory
dir=$exp/chain${chain_affix}/tdnn${tdnn_affix}_sp
train_data_dir=data/${combined_train_set}_sp_hires
train_ivector_dir=$exp/nnet3${nnet3_affix}/ivectors_${supervised_set}_sp_hires
lang=data/lang_chain

if [ $stage -le -1 ]; then
  echo "$0: chain training on the supervised subset data/${supervised_set}"
  local/semisup/chain/tuning/run_tdnn_100k_a.sh $train_supervised_opts \
                          --train-set $supervised_set \
                          --nnet3-affix "$nnet3_affix" --tdnn-affix "$tdnn_affix" \
                          --tree-affix "$tree_affix" --gmm $gmm --exp $exp || exit 1
fi

for f in data/${supervised_set}_sp_hires/feats.scp \
  data/${unsupervised_set}_sp_hires/feats.scp \
  data/${supervised_set}/feats.scp data/${unsupervised_set}/feats.scp \
  $exp/nnet3${nnet3_affix}/extractor \
  $gmm_dir/final.mdl $treedir/final.mdl $treedir/tree; do
  if [ ! -f $f ]; then
    echo "$0: Could not find $f"
    exit 1
  fi
done

if [ $stage -le 7 ]; then
  utils/combine_data.sh data/${combined_train_set}_sp_hires \
    data/${supervised_set}_sp_hires data/${unsupervised_set}_sp_hires

  utils/combine_data.sh data/${combined_train_set}_sp \
    data/${supervised_set}_sp data/${unsupervised_set}_sp
fi

if [ $stage -le 8 ]; then
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${combined_train_set}_sp_hires data/${combined_train_set}_sp_max2_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/${combined_train_set}_max2_hires $exp/nnet3${nnet3_affix}/extractor $exp/nnet3${nnet3_affix}/ivectors_${combined_train_set}_hires || exit 1;
fi

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 30 --cmd "$train_cmd" data/${combined_train_set}_sp \
    data/lang $gmm_dir $lat_dir || exit 1;
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 12 ]; then
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

if [ $stage -le 13 ]; then
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

graph_dir=$dir/graph_poco
if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_poco_test $dir $graph_dir
fi

decode_suff=
if [ $stage -le 15 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in dev test; do
      (
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $num_jobs --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir $exp/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
          $graph_dir data/${decode_set}_hires $dir/decode_poco_${decode_set}${decode_iter:+_$decode_iter}${decode_suff} || exit 1;
      ) &
  done
fi
wait;
exit 0;
