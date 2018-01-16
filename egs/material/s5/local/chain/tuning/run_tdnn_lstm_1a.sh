#!/bin/bash

# 1e is as 1d but instead of the --proportional-shrink option, using
#  the newly added xconfig-layer-specific 'l2-regularize' options.

# local/chain/compare_wer.sh exp/chain/tdnn1d_sp exp/chain/tdnn1e_sp
# System                tdnn1d_sp tdnn1e_sp
#WER dev_clean_2 (tgsmall)      14.21     13.43
#WER dev_clean_2 (tglarge)      10.41      9.76
# Final train prob        -0.0473   -0.0510
# Final valid prob        -0.0893   -0.0889
# Final train prob (xent)   -1.0757   -1.4148
# Final valid prob (xent)   -1.4222   -1.6640


# steps/info/chain_dir_info.pl exp/chain/tdnn1{d,e}_sp
# exp/chain/tdnn1d_sp: num-iters=17 nj=2..5 num-params=7.5M dim=40+100->2309 combine=-0.063->-0.052 xent:train/valid[10,16,final]=(-1.65,-1.23,-1.08/-1.91,-1.55,-1.42) logprob:train/valid[10,16,final]=(-0.084,-0.057,-0.047/-0.125,-0.100,-0.089)
# exp/chain/tdnn1e_sp: num-iters=17 nj=2..5 num-params=7.5M dim=40+100->2309 combine=-0.061->-0.056 xent:train/valid[10,16,final]=(-1.69,-1.41,-1.41/-1.91,-1.67,-1.66) logprob:train/valid[10,16,final]=(-0.065,-0.055,-0.051/-0.104,-0.095,-0.089)

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
decode_nj=30
train_set=train
test_sets=dev
gmm=tri3
language=swahili
nnet3_affix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
tlstm_affix=1a   # affix for the TDNN-LSTM directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
# training chunk-options
chunk_width=140,100,160
chunk_left_context=40
chunk_right_context=0
label_delay=5
common_egs_dir=
xent_regularize=0.1

# training options
srand=0
remove_egs=true
reporting_email=

#decode options
test_online_decoding=true  # if true, it will run the last decoding stage.


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
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.
local/nnet3/run_ivector_common.sh --stage $stage \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --language $language \
                                  --nnet3-affix "$nnet3_affix" || exit 1;

# Problem: We have removed the "train_" prefix of our training set in
# the alignment directory names! Bad!
gmm_dir=exp/$language/$gmm
ali_dir=exp/$language/${gmm}_ali_${train_set}_sp
tree_dir=exp/$language/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/$language/lang_chain
lat_dir=exp/$language/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/$language/chain${nnet3_affix}/tdnn_lstm${tlstm_affix}_sp
if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi
train_data_dir=data/$language/${train_set}_sp_hires
lores_train_data_dir=data/$language/${train_set}_sp
train_ivector_dir=exp/$language/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 10 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/$language/lang_test/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/$language/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/$language/lang_test $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 11 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 75 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/$language/lang_test $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 12 ]; then
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
    --cmd "$train_cmd" 6000 ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi

if [ $stage -le 13 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  tdnn_opts="l2-regularize=0.02"
  lstm_opts="l2-regularize=0.005"
  output_opts="l2-regularize=0.004"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 $tdnn_opts dim=512
  relu-batchnorm-layer name=tdnn2 $tdnn_opts input=Append(-1,0,1) dim=512
  relu-batchnorm-layer name=tdnn3 $tdnn_opts input=Append(-1,0,1) dim=512

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmp-layer name=lstm1 cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-3 $lstm_opts
  relu-batchnorm-layer name=tdnn4 $tdnn_opts input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn5 $tdnn_opts input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn6 $tdnn_opts input=Append(-3,0,3) dim=512
  fast-lstmp-layer name=lstm2 cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-3 $lstm_opts
  relu-batchnorm-layer name=tdnn7 $tdnn_opts input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn8 $tdnn_opts input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn9 $tdnn_opts input=Append(-3,0,3) dim=512
  fast-lstmp-layer name=lstm3 cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-3 $lstm_opts

  ## adding the layers for chain branch
  output-layer name=output input=lstm3 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5 $output_opts

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=lstm3 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5 $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 14 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/material-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.00005 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=5 \
    --trainer.frames-per-iter=1500000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=12 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.shrink-value=1.0 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.0 \
    --trainer.deriv-truncate-margin=8 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
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

if [ $stage -le 15 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/$language/lang_test \
    $tree_dir $tree_dir/graph || exit 1;
fi

if [ $stage -le 16 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context $chunk_left_context \
          --extra-right-context $chunk_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/$language/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph data/$language/${data}_hires ${dir}/decode_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

exit 0;
