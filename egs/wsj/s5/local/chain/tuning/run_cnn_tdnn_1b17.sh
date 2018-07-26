#!/bin/bash

# 1b17 is as 1b16 but taking the first layers from the 1a54 setup in mini_librispeech.
# A little better than the baseline.  Overfits more.
#
# local/chain/compare_wer.sh exp/chain/tdnn1g_sp exp/chain/cnn_tdnn1b17_sp
# System                tdnn1g_sp cnn_tdnn1b17_sp
#WER dev93 (tgpr)                6.68      6.55
#WER dev93 (tg)                  6.57      6.49
#WER dev93 (big-dict,tgpr)       4.60      4.52
#WER dev93 (big-dict,fg)         4.26      4.13
#WER eval92 (tgpr)               4.54      4.47
#WER eval92 (tg)                 4.32      4.15
#WER eval92 (big-dict,tgpr)      2.62      2.57
#WER eval92 (big-dict,fg)        2.32      2.02
# Final train prob        -0.0417   -0.0409
# Final valid prob        -0.0487   -0.0486
# Final train prob (xent)   -0.6461   -0.6203
# Final valid prob (xent)   -0.6882   -0.6591
# Num-params                 8354636   6935084

# 1b16 is like 1b12 but taking the cnn-layer part from the 1a30 setup in mini-librispeech,
# and adding another TDNN-F layer with splicing 3.
# Doesn't seem helpful.  This setup seems very vulnerable to overfitting.
#
# local/chain/compare_wer.sh exp/chain/tdnn1g_sp exp/chain/cnn_tdnn1b_sp exp/chain/cnn_tdnn1b10_sp exp/chain/cnn_tdnn1b11_sp exp/chain/cnn_tdnn1b12_sp exp/chain/cnn_tdnn1b13_sp
# System                tdnn1g_sp cnn_tdnn1b_sp cnn_tdnn1b10_sp cnn_tdnn1b11_sp cnn_tdnn1b12_sp cnn_tdnn1b13_sp
#WER dev93 (tgpr)                6.68      8.19      7.85      6.95      6.58      6.73
#WER dev93 (tg)                  6.57      7.76      7.49      6.98      6.59      6.59
#WER dev93 (big-dict,tgpr)       4.60      6.06      5.93      4.87      4.69      4.86
#WER dev93 (big-dict,fg)         4.26      5.49      5.15      4.62      4.30      4.35
#WER eval92 (tgpr)               4.54      5.69      5.58      4.59      4.59      4.73
#WER eval92 (tg)                 4.32      5.12      5.30      4.29      4.36      4.36
#WER eval92 (big-dict,tgpr)      2.62      3.47      3.56      2.60      2.59      2.68
#WER eval92 (big-dict,fg)        2.32      3.10      3.01      2.13      2.22      2.27
# Final train prob        -0.0417   -0.0469   -0.0463   -0.0401   -0.0414   -0.0401
# Final valid prob        -0.0487   -0.0663   -0.0668   -0.0504   -0.0483   -0.0486
# Final train prob (xent)   -0.6461   -0.8759   -0.8764   -0.6210   -0.6353   -0.6173
# Final valid prob (xent)   -0.6882   -1.0003   -0.9906   -0.6880   -0.6857   -0.6623
# Num-params                 8354636   5470268   5470268   6337852   6385980   6571836

# 1b12 is like 1b11 but making various changes that were helpful in the mini-librispeech
#  setup: using the same l2 values for the early layers; not doing splicing in the first
#  TDNN-F layer; and adding an extra TDNN-F layer.
# It's now about the same as tdnn1g_sp.
#
# local/chain/compare_wer.sh exp/chain/tdnn1g_sp exp/chain/cnn_tdnn1b_sp exp/chain/cnn_tdnn1b10_sp exp/chain/cnn_tdnn1b11_sp exp/chain/cnn_tdnn1b12_sp
# System                tdnn1g_sp cnn_tdnn1b_sp cnn_tdnn1b10_sp cnn_tdnn1b11_sp cnn_tdnn1b12_sp
#WER dev93 (tgpr)                6.68      8.19      7.85      6.95      6.58
#WER dev93 (tg)                  6.57      7.76      7.49      6.98      6.59
#WER dev93 (big-dict,tgpr)       4.60      6.06      5.93      4.87      4.69
#WER dev93 (big-dict,fg)         4.26      5.49      5.15      4.62      4.30
#WER eval92 (tgpr)               4.54      5.69      5.58      4.59      4.59
#WER eval92 (tg)                 4.32      5.12      5.30      4.29      4.36
#WER eval92 (big-dict,tgpr)      2.62      3.47      3.56      2.60      2.59
#WER eval92 (big-dict,fg)        2.32      3.10      3.01      2.13      2.22
# Final train prob        -0.0417   -0.0469   -0.0463   -0.0401   -0.0414
# Final valid prob        -0.0487   -0.0663   -0.0668   -0.0504   -0.0483
# Final train prob (xent)   -0.6461   -0.8759   -0.8764   -0.6210   -0.6353
# Final valid prob (xent)   -0.6882   -1.0003   -0.9906   -0.6880   -0.6857
# Num-params                 8354636   5470268   5470268   6337852   6385980
#
# 1b11 is like 1b10 but taking options and resnet-style TDNN-F configuration from tdnn_1g.sh.
#  (using slightly fewer epochs than 1g since the frames-per-minibatch is smaller here).
#   (re-dumped egs into 1b11b due to disk crash of b03).
# It's better than the previous cnn_tdnn experiments but not yet better than tdnn1g.
# local/chain/compare_wer.sh exp/chain/tdnn1g_sp exp/chain/cnn_tdnn1b_sp exp/chain/cnn_tdnn1b10_sp exp/chain/cnn_tdnn1b11_sp
# System                tdnn1g_sp cnn_tdnn1b_sp cnn_tdnn1b10_sp cnn_tdnn1b11_sp
#WER dev93 (tgpr)                6.68      8.19      7.85      6.95
#WER dev93 (tg)                  6.57      7.76      7.49      6.98
#WER dev93 (big-dict,tgpr)       4.60      6.06      5.93      4.87
#WER dev93 (big-dict,fg)         4.26      5.49      5.15      4.62
#WER eval92 (tgpr)               4.54      5.69      5.58      4.59
#WER eval92 (tg)                 4.32      5.12      5.30      4.29
#WER eval92 (big-dict,tgpr)      2.62      3.47      3.56      2.60
#WER eval92 (big-dict,fg)        2.32      3.10      3.01      2.13
# Final train prob        -0.0417   -0.0469   -0.0463   -0.0401
# Final valid prob        -0.0487   -0.0663   -0.0668   -0.0504
# Final train prob (xent)   -0.6461   -0.8759   -0.8764   -0.6210
# Final valid prob (xent)   -0.6882   -1.0003   -0.9906   -0.6880
# Num-params                 8354636   5470268   5470268   6337852

#
# 1b10 is like 1b but adding a batchnorm-component before the first CNN layer.

# 1b is like 1a, but converting the batch-norm layers in all but the CNN
# components back into renorm layers.
# Note: I'm not confident that the differences from 1a are entirely due
# to this change, as there have also been code changes, about how the
# combination works.

# exp/chain/tdnn1g_sp: num-iters=108 nj=2..8 num-params=8.4M dim=40+100->2854 combine=-0.042->-0.042 (over 2) xent:train/valid[71,107,final]=(-0.975,-0.640,-0.646/-0.980,-0.678,-0.688) logprob:train/valid[71,107,final]=(-0.067,-0.043,-0.042/-0.069,-0.050,-0.049)
# exp/chain/cnn_tdnn1b17_sp: num-iters=144 nj=2..8 num-params=6.9M dim=40+100->2854 combine=-0.041->-0.041 (over 3) xent:train/valid[95,143,final]=(-0.866,-0.617,-0.620/-0.881,-0.657,-0.659) logprob:train/valid[95,143,final]=(-0.061,-0.042,-0.041/-0.062,-0.050,-0.049)

# The following table compares chain (TDNN+LSTM, TDNN, CNN+TDNN).
# The CNN+TDNN doesn't seem to have any advantages versus the TDNN (and it's
# about 5 times slower per iteration).  But it's not well tuned.
# And the num-params is fewer (5.5M vs 7.6M for TDNN).

# local/chain/compare_wer.sh exp/chain/tdnn_lstm1a_sp exp/chain/tdnn1a_sp exp/chain/cnn_tdnn1a_sp
# System                tdnn_lstm1a_sp tdnn1a_sp cnn_tdnn1a_sp
#WER dev93 (tgpr)                7.48      7.87      9.02
#WER dev93 (tg)                  7.41      7.61      8.60
#WER dev93 (big-dict,tgpr)       5.64      5.71      6.97
#WER dev93 (big-dict,fg)         5.40      5.10      6.12
#WER eval92 (tgpr)               5.67      5.23      5.56
#WER eval92 (tg)                 5.46      4.87      5.05
#WER eval92 (big-dict,tgpr)      3.69      3.24      3.40
#WER eval92 (big-dict,fg)        3.28      2.71      2.73
# Final train prob        -0.0341   -0.0414   -0.0532
# Final valid prob        -0.0506   -0.0634   -0.0752
# Final train prob (xent)   -0.5643   -0.8216   -1.0857
# Final valid prob (xent)   -0.6648   -0.9208   -1.1505



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
affix=1b17  #affix for TDNN+LSTM directory e.g. "1a" or "1b", in case we change the configuration.
common_egs_dir=
reporting_email=

# LSTM/chain options
train_stage=-10
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

# training chunk-options
chunk_width=140,100,160

# training options
srand=0
remove_egs=true

#decode options
test_online_decoding=false  # if true, it will run the last decoding stage.

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

local/nnet3/run_ivector_common.sh \
  --stage $stage --nj $nj \
  --train-set $train_set --gmm $gmm \
  --num-threads-ubm $num_threads_ubm \
  --nnet3-affix "$nnet3_affix"



gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/cnn_tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp

# note: you don't necessarily have to change the treedir name
# each time you do a new experiment-- only if you change the
# configuration in a way that affects the tree.
tree_dir=exp/chain${nnet3_affix}/tree_a_sp
# the 'lang' directory is created by this script.
# If you create such a directory with a non-standard topology
# you should probably name it differently.
lang=data/lang_chain

for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $gmm_dir/final.mdl \
    $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le 12 ]; then
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

if [ $stage -le 13 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 100 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 14 ]; then
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


if [ $stage -le 15 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  cnn_opts="l2-regularize=0.01"
  ivector_affine_opts="l2-regularize=0.01"
  tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_first_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.0"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.005"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # this takes the MFCCs and generates filterbank coefficients.  The MFCCs
  # are more compressible so we prefer to dump the MFCCs to disk rather
  # than filterbanks.
  idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat

  linear-component name=ivector-linear $ivector_affine_opts dim=200 input=ReplaceIndex(ivector, t, 0)
  batchnorm-component name=ivector-batchnorm target-rms=0.025

  batchnorm-component name=idct-batchnorm input=idct
  combine-feature-maps-layer name=combine_inputs input=Append(idct-batchnorm, ivector-batchnorm) num-filters1=1 num-filters2=5 height=40

  conv-relu-batchnorm-layer name=cnn1 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=48 learning-rate-factor=0.333 max-change=0.25
  conv-relu-batchnorm-layer name=cnn2 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=48
  conv-relu-batchnorm-layer name=cnn3 $cnn_opts height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn4 $cnn_opts height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn5 $cnn_opts height-in=20 height-out=10 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn6 $cnn_opts height-in=10 height-out=5 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128

  # the first TDNN-F layer has no bypass (since dims don't match), and a larger bottleneck so the
  # information bottleneck doesn't become a problem.
  tdnnf-layer name=tdnnf7 $tdnnf_first_opts dim=1024 bottleneck-dim=256 time-stride=0
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  linear-component name=prefinal-l dim=192 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 16 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{4,5,6,7}/$USER/kaldi-data/egs/wsj-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
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
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=8 \
    --trainer.frames-per-iter=3000000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=8 \
    --trainer.optimization.initial-effective-lrate=0.0005 \
    --trainer.optimization.final-effective-lrate=0.00005 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.0 \
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

if [ $stage -le 17 ]; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/lang/check_phones_compatible.sh \
    data/lang_test_tgpr/phones.txt $lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_tgpr \
    $tree_dir $tree_dir/graph_tgpr || exit 1;

  utils/lang/check_phones_compatible.sh \
    data/lang_test_bd_tgpr/phones.txt $lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_bd_tgpr \
    $tree_dir $tree_dir/graph_bd_tgpr || exit 1;
fi

if [ $stage -le 18 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      for lmtype in tgpr bd_tgpr; do
        steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph_${lmtype} data/${data}_hires ${dir}/decode_${lmtype}_${data_affix} || exit 1
      done
      steps/lmrescore.sh \
        --self-loop-scale 1.0 \
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

# Not testing the 'looped' decoding separately, because for
# TDNN systems it would give exactly the same results as the
# normal decoding.

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
          --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nspk --cmd "$decode_cmd" \
          $tree_dir/graph_${lmtype} data/${data} ${dir}_online/decode_${lmtype}_${data_affix} || exit 1
      done
      steps/lmrescore.sh \
        --self-loop-scale 1.0 \
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
