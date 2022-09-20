#!/usr/bin/env bash


# run_tdnn_lstm_lfr_1a.sh is modified from the same named script
# in egs/tedlium/s5_r2/local/nnet3/tuning/.
# Of course reducing the hidden-dims).
# This is a low-frame-rate TDNN+LSTM system.


# steps/info/nnet3_dir_info.pl exp/nnet3/tdnn_lstm_lfr1a_sp
# exp/nnet3/tdnn_lstm_lfr1a_sp: num-iters=136 nj=3..10 num-params=8.7M dim=40+100->3205 combine=-0.43->-0.42 loglike:train/valid[89,135,combined]=(-0.51,-0.39,-0.38/-0.59,-0.51,-0.51) accuracy:train/valid[89,135,combined]=(0.85,0.88,0.88/0.82,0.84,0.84)


# It seems to be a little worse the regular-frame-rate system.

# local/nnet3/compare_wer.sh --looped exp/nnet3/tdnn_lstm1a_sp exp/nnet3/tdnn_lstm_lfr1a_sp
# System                tdnn_lstm1a_sp tdnn_lstm_lfr1a_sp
#WER dev93 (tgpr)                8.54      9.02
#             [looped:]          8.54      8.99
#WER dev93 (tg)                  8.25      8.60
#             [looped:]          8.21      8.54
#WER dev93 (big-dict,tgpr)       6.24      6.85
#             [looped:]          6.28      6.81
#WER dev93 (big-dict,fg)         5.70      6.33
#             [looped:]          5.70      6.33
#WER eval92 (tgpr)               6.52      6.52
#             [looped:]          6.45      6.42
#WER eval92 (tg)                 6.13      6.01
#             [looped:]          6.08      5.92
#WER eval92 (big-dict,tgpr)      3.88      4.22
#             [looped:]          3.93      4.20
#WER eval92 (big-dict,fg)        3.38      3.76
#             [looped:]          3.47      3.79
# Final train prob        -0.5492   -0.3100
# Final valid prob        -0.6343   -0.4646
# Final train acc          0.8154    0.9051
# Final valid acc          0.7849    0.8615


set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
nj=30
train_set=train_si284
test_sets="test_dev93 test_eval92"
gmm=tri4b        # this is the source gmm-dir that we'll use for alignments; it
                 # should have alignments for the specified training data.
num_threads_ubm=32
nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.

# Options which are not passed through to run_ivector_common.sh
affix=1a  #affix for TDNN+LSTM directory e.g. "1a" or "1b", in case we change the configuration.
common_egs_dir=
reporting_email=

# LSTM options
train_stage=-10
label_delay=5

# training chunk-options
chunk_width=40,30,20
chunk_left_context=40
chunk_right_context=0

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

local/nnet3/run_ivector_common.sh \
  --stage $stage --nj $nj \
  --train-set $train_set --gmm $gmm \
  --num-threads-ubm $num_threads_ubm \
  --nnet3-affix "$nnet3_affix"



gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp
dir=exp/nnet3${nnet3_affix}/tdnn_lstm_lfr${affix}_sp
train_data_dir=data/${train_set}_sp_hires
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
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
    --cmd "$train_cmd" 4000 data/${train_set}_sp \
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
  relu-renorm-layer name=tdnn1 dim=520
  relu-renorm-layer name=tdnn2 dim=520 input=Append(-1,0,1)
  fast-lstmp-layer name=lstm1 cell-dim=520 recurrent-projection-dim=130 non-recurrent-projection-dim=130 decay-time=20 delay=-3
  relu-renorm-layer name=tdnn3 dim=520 input=Append(-3,0,3)
  relu-renorm-layer name=tdnn4 dim=520 input=Append(-3,0,3)
  fast-lstmp-layer name=lstm2 cell-dim=520 recurrent-projection-dim=130 non-recurrent-projection-dim=130 decay-time=20 delay=-3
  relu-renorm-layer name=tdnn5 dim=520 input=Append(-3,0,3)
  relu-renorm-layer name=tdnn6 dim=520 input=Append(-3,0,3)
  fast-lstmp-layer name=lstm3 cell-dim=520 recurrent-projection-dim=130 non-recurrent-projection-dim=130 decay-time=20 delay=-3

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
    --trainer.samples-per-iter=10000 \
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

  utils/lang/check_phones_compatible.sh \
    data/lang_test_tgpr/phones.txt $lang/phones.txt
  utils/mkgraph.sh --self-loop-scale 0.333 data/lang_test_tgpr \
                   $dir $dir/graph_tgpr || exit 1;

  utils/lang/check_phones_compatible.sh \
    data/lang_test_bd_tgpr/phones.txt $lang/phones.txt
  utils/mkgraph.sh --self-loop-scale 0.333 data/lang_test_bd_tgpr \
      $dir $dir/graph_bd_tgpr || exit 1;
fi

if [ $stage -le 17 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      for lmtype in tgpr bd_tgpr; do
        steps/nnet3/decode.sh \
          --acwt 0.333 --post-decode-acwt 3.0 \
          --extra-left-context $chunk_left_context \
          --extra-right-context $chunk_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $dir/graph_${lmtype} data/${data}_hires ${dir}/decode_${lmtype}_${data_affix} || exit 1
      done
      steps/lmrescore.sh \
        --self-loop-scale 0.333 \
        --cmd "$decode_cmd" data/lang_test_{tgpr,tg} \
        data/${data}_hires ${dir}/decode_{tgpr,tg}_${data_affix} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_bd_{tgpr,fgconst} \
       data/${data}_hires ${dir}/decode_${lmtype}_${data_affix}{,_fg} || exit 1
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

  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      for lmtype in tgpr bd_tgpr; do
        steps/nnet3/decode_looped.sh \
          --acwt 0.333 --post-decode-acwt 3.0 \
          --frames-per-chunk 30 \
          --nj $nspk --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $dir/graph_${lmtype} data/${data}_hires ${dir}/decode_looped_${lmtype}_${data_affix} || exit 1
      done
      steps/lmrescore.sh \
        --self-loop-scale 0.333 \
        --cmd "$decode_cmd" data/lang_test_{tgpr,tg} \
        data/${data}_hires ${dir}/decode_looped_{tgpr,tg}_${data_affix} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_bd_{tgpr,fgconst} \
       data/${data}_hires ${dir}/decode_looped_${lmtype}_${data_affix}{,_fg} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

if $test_online_decoding && [ $stage -le 19 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3${nnet3_affix}/extractor ${dir} ${dir}_online

  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      # note: we just give it "data/${data}" as it only uses the wav.scp, the
      # feature type does not matter.
      for lmtype in tgpr bd_tgpr; do
        steps/online/nnet3/decode.sh \
          --acwt 0.333 --post-decode-acwt 3.0 \
          --nj $nspk --cmd "$decode_cmd" \
          $dir/graph_${lmtype} data/${data} ${dir}_online/decode_${lmtype}_${data_affix} || exit 1
      done
      steps/lmrescore.sh \
        --self-loop-scale 0.333 \
        --cmd "$decode_cmd" data/lang_test_{tgpr,tg} \
        data/${data}_hires ${dir}_online/decode_{tgpr,tg}_${data_affix} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_bd_{tgpr,fgconst} \
       data/${data}_hires ${dir}_online/decode_${lmtype}_${data_affix}{,_fg} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi



exit 0;
