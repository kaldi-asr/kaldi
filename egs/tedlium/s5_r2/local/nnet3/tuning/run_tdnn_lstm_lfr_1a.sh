#!/usr/bin/env bash


# run_tdnn_lstm_lfr_1a.sh is like run_tdnn_lstm_1a.sh, but
# it's a low-frame-rate system. (however, using num-jobs-final=10,
# not 15, which was very high).


# Generally the WER is the same or slightly better than before.

# local/nnet3/compare_wer.sh --looped --online exp/nnet3_cleaned/tdnn_lstm1c_sp exp/nnet3_cleaned/tdnn_lstm_lfr1a_sp  2>/dev/null
# local/nnet3/compare_wer.sh --looped --online exp/nnet3_cleaned/tdnn_lstm1c_sp exp/nnet3_cleaned/tdnn_lstm_lfr1a_sp
# System                tdnn_lstm1c_sp tdnn_lstm_lfr1a_sp
# WER on dev(orig)           11.0      10.9
#         [looped:]          10.9      10.9
#         [online:]                    10.8
# WER on dev(rescored)       10.4      10.3
#         [looped:]          10.3      10.3
#         [online:]                    10.3
# WER on test(orig)          10.8      10.7
#         [looped:]          10.7      10.7
#         [online:]                    10.7
# WER on test(rescored)      10.1      10.2
#         [looped:]          10.1      10.1
#         [online:]                    10.2
# Final train prob        -0.5998   -0.5437
# Final valid prob        -0.8542   -0.7286
# Final train acc          0.7988    0.8343
# Final valid acc          0.7521    0.7888


# by default, with cleanup:
# local/nnet3/run_tdnn_lstm_lfr.sh

# without cleanup:
# local/nnet3/run_tdnn_lstm_lfr.sh  --train-set train --gmm tri3 --nnet3-affix "" &


set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
nj=30
decode_nj=30
min_seg_len=1.55
train_set=train_cleaned
gmm=tri3_cleaned  # this is the source gmm-dir for the data-type of interest; it
                  # should have alignments for the specified training data.
num_threads_ubm=32
nnet3_affix=_cleaned  # cleanup affix for exp dirs, e.g. _cleaned

# Options which are not passed through to run_ivector_common.sh
affix=1a
common_egs_dir=
reporting_email=

# LSTM options
train_stage=-10
label_delay=5

# training chunk-options
chunk_width=40,30,20
chunk_left_context=40
chunk_right_context=0
# decode chunk-size options (for non-looped decoding)
extra_left_context=50
extra_right_context=0

# training options
srand=0
remove_egs=true

#decode options
test_online_decoding=false  # if true, it will run the last decoding stage.

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



gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp_comb
dir=exp/nnet3${nnet3_affix}/tdnn_lstm_lfr${affix}_sp
train_data_dir=data/${train_set}_sp_hires_comb
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb
# note: you don't necessarily have to change the treedir name
# each time you do a new experiment-- only if you change the
# configuration in a way that affects the tree.
treedir=exp/nnet3${nnet3_affix}/tree_lfr_a_sp
# the 'lang' directory is created by this script; it's one
# suitable for a low-frame-rate system such as this one.
lang=data/lang_lfr_a

for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le 12 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 13 ]; then
  # Build a tree using our new topology and a reduced sampling rate.
  # We use 4000 leaves, which is a little less than the number used
  # in the baseline GMM system (5k) in this setup, since generally
  # LFR systems do best with somewhat fewer leaves.
  #
  # To get the stats to build the tree this script only uses every third frame,
  # but it dumps converted alignments that essentially have 3 different
  # frame-shifted versions of the alignment interpolated together; these can be
  # used without modification in getting labels for training.
  steps/nnet3/chain/build_tree.sh \
    --repeat-frames true --frame-subsampling-factor 3 \
    --cmd "$train_cmd" 4000 data/${train_set}_sp_comb \
    $lang $ali_dir $treedir
fi


if [ $stage -le 14 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=768
  relu-renorm-layer name=tdnn2 dim=768 input=Append(-1,0,1)
  fast-lstmp-layer name=lstm1 cell-dim=768 recurrent-projection-dim=192 non-recurrent-projection-dim=192 decay-time=20 delay=-3
  relu-renorm-layer name=tdnn3 dim=768 input=Append(-3,0,3)
  relu-renorm-layer name=tdnn4 dim=768 input=Append(-3,0,3)
  fast-lstmp-layer name=lstm2 cell-dim=768 recurrent-projection-dim=192 non-recurrent-projection-dim=192 decay-time=20 delay=-3
  relu-renorm-layer name=tdnn5 dim=768 input=Append(-3,0,3)
  relu-renorm-layer name=tdnn6 dim=768 input=Append(-3,0,3)
  fast-lstmp-layer name=lstm3 cell-dim=768 recurrent-projection-dim=192 non-recurrent-projection-dim=192 decay-time=20 delay=-3

  output-layer name=output input=lstm3 output-delay=$label_delay dim=$num_targets max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 15 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5_r2/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_rnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=6 \
    --trainer.deriv-truncate-margin=10 \
    --trainer.samples-per-iter=20000 \
    --trainer.optimization.num-jobs-initial=3 \
    --trainer.optimization.num-jobs-final=10 \
    --trainer.optimization.initial-effective-lrate=0.0003 \
    --trainer.optimization.final-effective-lrate=0.00003 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.rnn.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.5 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
    --egs.dir="$common_egs_dir" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --feat-dir=$train_data_dir \
    --ali-dir=$treedir \
    --lang=$lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
  echo 3 >$dir/frame_subsampling_factor
fi

if [ $stage -le 16 ]; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/lang/check_phones_compatible.sh data/lang/phones.txt $lang/phones.txt
  utils/mkgraph.sh --self-loop-scale 0.333 data/lang $dir $dir/graph
fi

if [ $stage -le 17 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true
  for dset in dev test; do
   (
     steps/nnet3/decode.sh \
       --acwt 0.333 --post-decode-acwt 3.0 --nj $decode_nj \
       --cmd "$decode_cmd"  --num-threads 4 \
       --extra-left-context $chunk_left_context \
       --extra-right-context $chunk_right_context \
       --frames-per-chunk $frames_per_chunk \
       --extra-left-context-initial 0 --extra-right-context-final 0 \
       --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
      $dir/graph data/${dset}_hires ${dir}/decode_${dset} || exit 1
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
       data/${dset}_hires ${dir}/decode_${dset} ${dir}/decode_${dset}_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi


if [ $stage -le 18 ]; then
  # 'looped' decoding.
  # note: you should NOT do this decoding step for setups that have bidirectional
  # recurrence, like BLSTMs-- it doesn't make sense and will give bad results.
  # we didn't write a -parallel version of this program yet,
  # so it will take a bit longer as the --num-threads option is not supported.
  # we just hardcode the --frames-per-chunk option as it doesn't have to
  # match any value used in training, and it won't affect the results (unlike
  # regular decoding).
  rm $dir/.error 2>/dev/null || true
  for dset in dev test; do
      (
        steps/nnet3/decode_looped.sh \
          --acwt 0.333 --post-decode-acwt 3.0 \
          --nj $decode_nj --cmd "$decode_cmd" \
          --frames-per-chunk 30 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
         $dir/graph data/${dset}_hires $dir/decode_looped_${dset} || exit 1;
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
        data/${dset}_hires ${dir}/decode_looped_${dset} ${dir}/decode_looped_${dset}_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

if $test_online_decoding && [ $stage -le 19 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3${nnet3_affix}/extractor ${dir} ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for dset in dev test; do
    (
      # note: we just give it "$dset" as it only uses the wav.scp, the
      # feature type does not matter.

      steps/online/nnet3/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 0.333 --post-decode-acwt 3.0 \
          --scoring-opts "--min-lmwt 5 " \
         $dir/graph data/${dset} ${dir}_online/decode_${dset} || exit 1;
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
        data/${dset}_hires ${dir}_online/decode_${dset} ${dir}_online/decode_${dset}_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi



exit 0;
