#!/bin/bash

set -e

# based on run_tdnn_7b.sh in the swbd recipe

# System                      exp/chain/tdnn_lstm_1a
# WER on dev_aspire (fg)                    22.9
# Final train prob                        -0.118
# Final valid prob                        -0.123
# Final train prob (xent)                 -1.243
# Final valid prob (xent)                -1.2350
# Num-parameters                        49945168

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
test_stage=1
nj=70

tdnn_affix=_1a

hidden_dim=1024
cell_dim=1024
projection_dim=256

# training options
num_epochs=2
minibatch_size=64,32
chunk_left_context=40
chunk_right_context=0
dropout_schedule='0,0@0.20,0.3@0.50,0'
xent_regularize=0.025
label_delay=5

# decode options
extra_left_context=50
extra_right_context=0

# training options
num_epochs=4
remove_egs=false
common_egs_dir=

num_data_reps=3
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

train_set=train_rvb

gmm_dir=exp/tri5a   # used to get training lattices (for chain supervision)
treedir=exp/chain/tree_bi_a
lat_dir=exp/chain/tri5a_${train_set}_lats  # training lattices directory
dir=exp/chain/tdnn_lstm${tdnn_affix}
train_data_dir=data/${train_set}_hires
train_ivector_dir=exp/nnet3/ivectors_${train_set}
lang=data/lang_chain


# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.
local/nnet3/run_ivector_common.sh --stage $stage --num-data-reps 3 || exit 1

mkdir -p $dir

norvb_lat_dir=exp/chain/tri5a_train_lats

if [ $stage -le 7 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 30 --cmd "$train_cmd" \
    --generate-ali-from-lats true data/train \
    data/lang $gmm_dir $norvb_lat_dir || exit 1;
  rm $norvb_lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 8 ]; then
  mkdir -p $lat_dir

  utils/split_data.sh data/${train_set} $nj

  for n in `seq $nj`; do
    awk '{print $1}' data/${train_set}/split$nj/$n/utt2spk | \
      perl -ane 's/rev[1-3]_//g' > $lat_dir/uttlist.$n.$nj
  done

  rm -f $lat_dir/lat_tmp.*.{ark,scp} 2>/dev/null

  norvb_nj=$(cat $norvb_lat_dir/num_jobs)
  $train_cmd JOB=1:$norvb_nj $lat_dir/log/copy_lattices.JOB.log \
    lattice-copy "ark:gunzip -c $norvb_lat_dir/lat.JOB.gz |" \
    ark,scp:$lat_dir/lat_tmp.JOB.ark,$lat_dir/lat_tmp.JOB.scp || exit 1

  for n in `seq 3`; do
    cat $lat_dir/lat_tmp.*.scp | awk -v n=$n '{print "rev"n"_"$1" "$2}'
  done > $lat_dir/lat_rvb.scp

  $train_cmd JOB=1:$nj $lat_dir/log/copy_rvb_lattices.JOB.log \
    lattice-copy \
    "scp:utils/filter_scp.pl data/${train_set}/split$nj/JOB/utt2spk $lat_dir/lat_rvb.scp |" \
    "ark:| gzip -c > $lat_dir/lat.JOB.gz" || exit 1

  rm $lat_dir/lat_tmp.* $lat_dir/lat_rvb.scp

  echo $nj > $lat_dir/num_jobs

  for f in cmvn_opts final.mdl splice_opts tree; do
    cp $norvb_lat_dir/$f $lat_dir/$f
  done
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

if [ $stage -le 11 ]; then
  # Build a tree using our new topology.
  # we build the tree using clean features (data/train) rather than
  # the augmented features (data/train_rvb) to get better alignments

  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate -1 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 data/train $lang exp/tri5a $treedir || exit 1
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  lstm_opts="decay-time=40"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=$hidden_dim
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=$hidden_dim

  fast-lstmp-layer name=lstm1 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=$hidden_dim
  fast-lstmp-layer name=lstm2 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=$hidden_dim
  fast-lstmp-layer name=lstm3 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=tdnn8 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn9 input=Append(-3,0,3) dim=$hidden_dim
  fast-lstmp-layer name=lstm4 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts

  ## adding the layers for chain branch
  output-layer name=output input=lstm4 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=lstm4 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  mkdir -p $dir/egs
  touch $dir/egs/.nodelete # keep egs around when that run dies.

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.num-chunk-per-minibatch 64,32 \
    --trainer.frames-per-iter 1500000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.optimization.momentum 0.0 \
    --trainer.deriv-truncate-margin 8 \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --generate-egs-scp true" \
    --egs.chunk-width 160,140,110,80 \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $treedir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;
fi

graph_dir=$dir/graph_pp
if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_pp_test $dir $graph_dir
fi

if [ $stage -le 15 ]; then
  rm $dir/.error 2>/dev/null || true

  for d in dev_rvb test_rvb; do
    (
      if [ ! -f exp/nnet3/ivectors_${d}/ivector_online.scp ]; then
        steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
          data/${d}_hires exp/nnet3/extractor \
          exp/nnet3/ivectors_${d} || { echo "Failed i-vector extraction for data/${d}_hires"; touch $dir/.error; }
      fi

      decode_dir=$dir/decode_${d}_pp
      steps/nnet3/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --extra-left-context $extra_left_context \
        --extra-right-context $extra_right_context \
        --extra-left-context-initial 0 --extra-right-context-final 0 \
        --frames-per-chunk 160 \
        --online-ivector-dir exp/nnet3/ivectors_${d} \
        $graph_dir data/${d}_hires $decode_dir || { echo "Failed decoding in $decode_dir"; touch $dir/.error; }
    ) &
  done
  wait

  if [ -f $dir/.error ]; then
    echo "Failed decoding."
    exit 1
  fi
fi

if [ $stage -le 16 ]; then
  # %WER 22.9 | 2083 25834 | 81.6 12.0 6.4 4.5 22.9 70.7 | -0.546 | exp/chain/tdnn_lstm_1a/decode_dev_aspire_uniformsegmented_v9_pp_fg/score_8/penalty_0.0/ctm.filt.filt.sys
  local/nnet3/decode.sh --stage $test_stage --decode-num-jobs 30 --affix "v9" \
   --acwt 1.0 --post-decode-acwt 10.0 \
   --window 10 --overlap 5 --frames-per-chunk 160 \
   --extra-left-context $extra_left_context \
   --extra-right-context $extra_right_context \
   --extra-left-context-initial 0 --extra-right-context-final 0 \
   --sub-speaker-frames 6000 --max-count 75 --ivector-scale 0.75 \
   --pass2-decode-opts "--min-active 1000" \
   dev_aspire data/lang $dir/graph_pp $dir
fi

if [ $stage -le 17 ]; then
  # %WER 24.0 | 2083 25820 | 79.9 12.0 8.1 4.0 24.0 71.8 | -0.444 | exp/chain/tdnn_lstm_1a_online/decode_dev_aspire_uniformsegmented_v9_pp_fg/score_10/penalty_0.0/ctm.filt.filt.sys
  local/nnet3/decode_online.sh --stage $test_stage --decode-num-jobs 30 --affix "v9" \
   --acwt 1.0 --post-decode-acwt 10.0 \
   --window 10 --overlap 5 --frames-per-chunk 160 \
   --extra-left-context $extra_left_context \
   --extra-right-context $extra_right_context \
   --extra-left-context-initial 0 \
   --max-count 75 \
   --pass2-decode-opts "--min-active 1000" \
   dev_aspire data/lang $dir/graph_pp $dir
fi

exit 0;

