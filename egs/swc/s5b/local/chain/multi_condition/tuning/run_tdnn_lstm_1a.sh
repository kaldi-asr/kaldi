#!/bin/bash

# This is a chain-training script with TDNN+LSTM neural networks.
# This script is based on local/chain/tuning/run_tdnn_lstm_1i.sh, but adding
# the reverberated IHM data into the train set.
# This script obtains better results on IHM, SDM and MDM tasks.

# Please see RESULTS_* for examples of command lines invoking this script.

# local/chain/multi_condition/run_tdnn_lstm.sh --mic ihm --train-set train_cleaned --gmm tri3_cleaned &
# local/chain/multi_condition/run_tdnn_lstm.sh --mic sdm1 --use-ihm-ali true --train-set train_cleaned --gmm tri3_cleaned &
# local/chain/multi_condition/run_tdnn_lstm.sh --mic mdm8 --use-ihm-ali true --train-set train_cleaned --gmm tri3_cleaned &


set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
mic=ihm
nj=30
min_seg_len=1.55
use_ihm_ali=false
train_set=train_cleaned
gmm=tri3_cleaned  # the gmm for the target data
ihm_gmm=tri3_cleaned  # the gmm for the IHM system (if --use-ihm-ali true).
num_threads_ubm=32
num_data_reps=1

chunk_width=150
chunk_left_context=40
chunk_right_context=0
label_delay=5
# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
train_stage=-10
tree_affix=  # affix for tree directory, e.g. "a" or "b", in case we change the configuration.
tlstm_affix=1i  #affix for TDNN-LSTM directory, e.g. "a" or "b", in case we change the configuration.
common_egs_dir=  # you can set this to use previously dumped egs.


# decode options
extra_left_context=50
frames_per_chunk=


# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! $use_ihm_ali; then
  [ "$mic" != "ihm" ] && \
    echo "$0: you cannot specify --use-ihm-ali false if the microphone is not ihm." && \
    exit 1;
else
  [ "$mic" == "ihm" ] && \
    echo "$0: you must specify --use-ihm-ali false if the microphone is ihm." && \
    exit 1;
fi

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

nnet3_affix=_cleaned
rvb_affix=_rvb


if $use_ihm_ali; then
  gmm_dir=exp/ihm/${ihm_gmm}
  ali_dir=exp/${mic}/${ihm_gmm}_ali_${train_set}_sp_comb_ihmdata
  lores_train_data_dir=data/$mic/${train_set}_ihmdata_sp_comb
  tree_dir=exp/$mic/chain${nnet3_affix}/tree_bi${tree_affix}_ihmdata
  original_lat_dir=exp/$mic/chain${nnet3_affix}/${ihm_gmm}_${train_set}_sp_comb_lats_ihmdata
  lat_dir=exp/$mic/chain${nnet3_affix}${rvb_affix}/${ihm_gmm}_${train_set}_sp${rvb_affix}_comb_lats_ihmdata
  dir=exp/$mic/chain${nnet3_affix}${rvb_affix}/tdnn_lstm${tlstm_affix}_sp${rvb_affix}_bi_ihmali
  # note: the distinction between when we use the 'ihmdata' suffix versus
  # 'ihmali' is pretty arbitrary.
else
  gmm_dir=exp/${mic}/$gmm
  ali_dir=exp/${mic}/${gmm}_ali_${train_set}_sp_comb
  lores_train_data_dir=data/$mic/${train_set}_sp_comb
  tree_dir=exp/$mic/chain${nnet3_affix}/tree_bi${tree_affix}
  original_lat_dir=exp/$mic/chain${nnet3_affix}/${gmm}_${train_set}_sp_comb_lats
  lat_dir=exp/$mic/chain${nnet3_affix}${rvb_affix}/${gmm}_${train_set}_sp${rvb_affix}_comb_lats
  dir=exp/$mic/chain${nnet3_affix}${rvb_affix}/tdnn_lstm${tlstm_affix}_sp${rvb_affix}_bi
fi


local/nnet3/multi_condition/run_ivector_common.sh --stage $stage \
                                  --mic $mic \
                                  --nj $nj \
                                  --min-seg-len $min_seg_len \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --num-threads-ubm $num_threads_ubm \
                                  --num-data-reps $num_data_reps \
                                  --nnet3-affix "$nnet3_affix"


# Note: the first stage of the following script is stage 8.
local/nnet3/prepare_lores_feats.sh --stage $stage \
                                   --mic $mic \
                                   --nj $nj \
                                   --min-seg-len $min_seg_len \
                                   --use-ihm-ali $use_ihm_ali \
                                   --train-set $train_set


train_data_dir=data/$mic/${train_set}_sp${rvb_affix}_hires_comb
train_ivector_dir=exp/$mic/nnet3${nnet3_affix}${rvb_affix}/ivectors_${train_set}_sp${rvb_affix}_hires_comb
final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7


for f in $gmm_dir/final.mdl $lores_train_data_dir/feats.scp \
   $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le 11 ]; then
  if [ -f $ali_dir/ali.1.gz ]; then
    echo "$0: alignments in $ali_dir appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning perturbed, short-segment-combined ${maybe_ihm}data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
     ${lores_train_data_dir} data/lang $gmm_dir $ali_dir
fi

[ ! -f $ali_dir/ali.1.gz ] && echo  "$0: expected $ali_dir/ali.1.gz to exist" && exit 1

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
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >data/lang_chain/topo
  fi
fi

if [ $stage -le 13 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 100 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $original_lat_dir
  rm $original_lat_dir/fsts.*.gz # save space

  lat_dir_ihmdata=exp/ihm/chain${nnet3_affix}/${gmm}_${train_set}_sp_comb_lats

  mkdir -p $lat_dir/temp/
  mkdir -p $lat_dir/temp2/
  lattice-copy "ark:gunzip -c $original_lat_dir/lat.*.gz |" ark,scp:$lat_dir/temp/lats.ark,$lat_dir/temp/lats.scp
  lattice-copy "ark:gunzip -c $lat_dir_ihmdata/lat.*.gz |" ark,scp:$lat_dir/temp2/lats.ark,$lat_dir/temp2/lats.scp

  # copy the lattices for the reverberated data
  rm -f $lat_dir/temp/combined_lats.scp
  touch $lat_dir/temp/combined_lats.scp
  cat $lat_dir/temp/lats.scp >> $lat_dir/temp/combined_lats.scp
  for i in `seq 1 $num_data_reps`; do
    cat $lat_dir/temp2/lats.scp | sed -e "s/^/rev${i}_/" >> $lat_dir/temp/combined_lats.scp
  done
  sort -u $lat_dir/temp/combined_lats.scp > $lat_dir/temp/combined_lats_sorted.scp

  lattice-copy scp:$lat_dir/temp/combined_lats_sorted.scp "ark:|gzip -c >$lat_dir/lat.1.gz" || exit 1;
  echo "1" > $lat_dir/num_jobs

  # copy other files from original lattice dir
  for f in cmvn_opts final.mdl splice_opts tree; do
    cp $original_lat_dir/$f $lat_dir/$f
  done
fi


if [ $stage -le 14 ]; then
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
      --cmd "$train_cmd" 4200 ${lores_train_data_dir} data/lang_chain $ali_dir $tree_dir
fi

xent_regularize=0.1

if [ $stage -le 15 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
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
  relu-renorm-layer name=tdnn1 dim=1024
  relu-renorm-layer name=tdnn2 input=Append(-1,0,1) dim=1024
  relu-renorm-layer name=tdnn3 input=Append(-1,0,1) dim=1024

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  lstmp-layer name=lstm1 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3
  relu-renorm-layer name=tdnn4 input=Append(-3,0,3) dim=1024
  relu-renorm-layer name=tdnn5 input=Append(-3,0,3) dim=1024
  relu-renorm-layer name=tdnn6 input=Append(-3,0,3) dim=1024
  lstmp-layer name=lstm2 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3
  relu-renorm-layer name=tdnn7 input=Append(-3,0,3) dim=1024
  relu-renorm-layer name=tdnn8 input=Append(-3,0,3) dim=1024
  relu-renorm-layer name=tdnn9 input=Append(-3,0,3) dim=1024
  lstmp-layer name=lstm3 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3

  ## adding the layers for chain branch
  output-layer name=output input=lstm3 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=lstm3 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 16 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5b/$dir/egs/storage $dir/egs/storage
  fi

 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $chunk_width \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 4 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
    --trainer.deriv-truncate-margin 8 \
    --cleanup.remove-egs true \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir
fi


graph_dir=$dir/graph_${LM}
if [ $stage -le 17 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_${LM} $dir $graph_dir
fi

if [ $stage -le 18 ]; then
  rm $dir/.error 2>/dev/null || true

  [ -z $extra_left_context ] && extra_left_context=$chunk_left_context;
  [ -z $frames_per_chunk ] && frames_per_chunk=$chunk_width;

  for decode_set in dev eval; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nj --cmd "$decode_cmd" \
          --extra-left-context $extra_left_context  \
          --frames-per-chunk "$frames_per_chunk" \
          --online-ivector-dir exp/$mic/nnet3${nnet3_affix}${rvb_affix}/ivectors_${decode_set}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $graph_dir data/$mic/${decode_set}_hires $dir/decode_${decode_set} || exit 1;
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi
exit 0
