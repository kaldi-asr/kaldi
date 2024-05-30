#!/usr/bin/env bash


# 1b is like 1a but instead of having 3 fast-lstm-layers, having one
#  lstmb-layer.  Caution: although it's better than run_tdnn_lstm_1a.sh, it's
#  still not better than run_tdnn_1f.sh, and my experience with this LSTMB layer
#  on larger-scale setups like Switchboard has not been good.  So I *don't
#  particularly recommend* this setup.


# local/chain/compare_wer.sh exp/chain/tdnn_lstm1a_sp exp/chain/tdnn_lstm1b_sp
# System                tdnn_lstm1a_sp  tdnn_lstm1b_sp
#WER dev93 (tgpr)                7.64       7.24
#WER dev93 (tg)                  7.29       7.03
#WER dev93 (big-dict,tgpr)       5.53       5.04
#WER dev93 (big-dict,fg)         5.14       4.92
#WER eval92 (tgpr)               5.62       5.23
#WER eval92 (tg)                 5.30       4.78
#WER eval92 (big-dict,tgpr)      3.62       3.17
#WER eval92 (big-dict,fg)        3.31       2.73
# Final train prob        -0.0344    -0.0403
# Final valid prob        -0.0518    -0.0526
# Final train prob (xent)   -0.5589    -0.7406
# Final valid prob (xent)   -0.6620    -0.7766
# Num-params                 9106252    4216524

# 1b22 is as 1b21 but setting chain.l2-regularize to zero.

# 1b21 is as 1b20 but half the learning rate..

# 1b20 is as 1b19b but reducing dimensions of TDNN layers from 512 to 448.
# 1b19b is as 1b19 but with more epochs (4->6)
# 1b19 is a rerun of 1b18d3 (a fairly small LSTM+TDNN setup).
#
#
# 1b18d3 is as 1b18d2 but reducing lstm bottleneck dim from 304 to 256.
# [1b18d2 is just a rerun of 1b18d as I merged various code changes and
#  I want to make sure nothing bad happened.]
#
# Results below show it's probably slightly better than the average of 18d and 18d2
#   (which are supposed to be the same experiment)...
#
# local/chain/compare_wer.sh exp/chain/tdnn_lstm1b18d_sp exp/chain/tdnn_lstm1b18d2_sp exp/chain/tdnn_lstm1b18d3_sp
# System                tdnn_lstm1b18d_sp tdnn_lstm1b18d2_sp tdnn_lstm1b18d3_sp
#WER dev93 (tgpr)                7.78      7.46      7.46
#WER dev93 (tg)                  7.29      7.30      7.04
#WER dev93 (big-dict,tgpr)       5.56      5.51      5.55
#WER dev93 (big-dict,fg)         5.32      5.08      5.05
#WER eval92 (tgpr)               5.33      5.40      5.39
#WER eval92 (tg)                 5.05      5.03      4.96
#WER eval92 (big-dict,tgpr)      3.42      3.26      3.35
#WER eval92 (big-dict,fg)        2.91      2.64      2.82
# Final train prob        -0.0529   -0.0536   -0.0543
# Final valid prob        -0.0633   -0.0630   -0.0636
# Final train prob (xent)   -0.8327   -0.8330   -0.8415
# Final valid prob (xent)   -0.8693   -0.8672   -0.8695
# Num-params                 4922060   4922060   4805324

#
# 1b18d is as 1b18c, but adding 'self-scale=2.0' to scale up the m_trunc when it is given
# as input to the affine projections (I found previously this was helpful).
# .. Interesting: objf improves but WER is not better.
#
# local/chain/compare_wer.sh exp/chain/tdnn_lstm1b18c_sp exp/chain/tdnn_lstm1b18d_sp
# System                tdnn_lstm1b18c_sp tdnn_lstm1b18d_sp
#WER dev93 (tgpr)                7.77      7.78
#WER dev93 (tg)                  7.40      7.29
#WER dev93 (big-dict,tgpr)       5.39      5.56
#WER dev93 (big-dict,fg)         5.25      5.32
#WER eval92 (tgpr)               5.48      5.33
#WER eval92 (tg)                 4.98      5.05
#WER eval92 (big-dict,tgpr)      3.07      3.42
#WER eval92 (big-dict,fg)        2.69      2.91
# Final train prob        -0.0546   -0.0529
# Final valid prob        -0.0641   -0.0633
# Final train prob (xent)   -0.8679   -0.8327
# Final valid prob (xent)   -0.8954   -0.8693
# Num-params                 4922060   4922060

# 1b18c is as 1b18b, but fixing a bug in the script whereby c instead of m had been used
# as input to the affine projections.

# 1b18b is as 1b18, but doubling l2 regularization on the output
#  and lstm layers, parts of them were training too slowly.
#
# 1b18 is as 1b17, but via script change, not using memory-norm (actually
#   this is the same as 1b17d).
#  I don't see any WER change, but objf is worse.

# local/chain/compare_wer.sh exp/chain/tdnn_lstm1b17_sp exp/chain/tdnn_lstm1b17d_sp exp/chain/tdnn_lstm1b18_sp
# System                tdnn_lstm1b17_sp tdnn_lstm1b17d_sp tdnn_lstm1b18_sp
#WER dev93 (tgpr)                7.49      7.44      7.48
#WER dev93 (tg)                  7.18      7.13      7.19
#WER dev93 (big-dict,tgpr)       5.50      5.34      5.48
#WER dev93 (big-dict,fg)         5.11      5.15      5.04
#WER eval92 (tgpr)               5.26      5.32      5.32
#WER eval92 (tg)                 5.00      4.94      5.03
#WER eval92 (big-dict,tgpr)      3.24      3.28      3.26
#WER eval92 (big-dict,fg)        2.82      2.80      2.84
# Final train prob        -0.0489   -0.0486   -0.0496
# Final valid prob        -0.0583   -0.0599   -0.0612
# Final train prob (xent)   -0.7550   -0.7809   -0.7749
# Final valid prob (xent)   -0.7988   -0.8121   -0.8131
# Num-params                 4922060   4922060   4922060

# 1b17 is as 1b13m, it's just a rerun after some code changes (adding
# diagonal natural gradient stuff) which should make no difference.
# Still seems to be working.

# local/chain/compare_wer.sh exp/chain/tdnn_lstm1b13d_sp exp/chain/tdnn_lstm1b13m_sp exp/chain/tdnn_lstm1b17_sp
# System                tdnn_lstm1b13d_sp tdnn_lstm1b13m_sp tdnn_lstm1b17_sp
#WER dev93 (tgpr)                7.86      7.43      7.49
#WER dev93 (tg)                  7.40      7.00      7.18
#WER dev93 (big-dict,tgpr)       5.65      5.21      5.50
#WER dev93 (big-dict,fg)         5.11      4.76      5.11
#WER eval92 (tgpr)               5.64      5.39      5.26
#WER eval92 (tg)                 5.17      5.00      5.00
#WER eval92 (big-dict,tgpr)      3.21      3.30      3.24
#WER eval92 (big-dict,fg)        2.84      2.62      2.82
# Final train prob        -0.0469   -0.0516   -0.0489
# Final valid prob        -0.0601   -0.0607   -0.0583
# Final train prob (xent)   -0.7424   -0.7593   -0.7550
# Final valid prob (xent)   -0.7920   -0.7982   -0.7988
# Num-params                 5456076   4922060   4922060

# 1b13m is as 1b13l, but reverting the LSTM script "fix" (which actually
#  made things worse), so the baseline is 1b13{c,d} (and the change versus
# c,d is to add bottleneck-dim=256).
#
# It's helpful:
# local/chain/compare_wer.sh exp/chain/tdnn_lstm1b13c_sp exp/chain/tdnn_lstm1b13d_sp exp/chain/tdnn_lstm1b13m_sp
# System                tdnn_lstm1b13c_sp tdnn_lstm1b13d_sp tdnn_lstm1b13m_sp
#WER dev93 (tgpr)                7.68      7.86      7.43
#WER dev93 (tg)                  7.34      7.40      7.00
#WER dev93 (big-dict,tgpr)       5.42      5.65      5.21
#WER dev93 (big-dict,fg)         5.05      5.11      4.76
#WER eval92 (tgpr)               5.48      5.64      5.39
#WER eval92 (tg)                 5.26      5.17      5.00
#WER eval92 (big-dict,tgpr)      3.23      3.21      3.30
#WER eval92 (big-dict,fg)        2.82      2.84      2.62
# Final train prob        -0.0490   -0.0469   -0.0516
# Final valid prob        -0.0597   -0.0601   -0.0607
# Final train prob (xent)   -0.7549   -0.7424   -0.7593
# Final valid prob (xent)   -0.7910   -0.7920   -0.7982
# Num-params                 5456076   5456076   4922060
#
#
# 1b13l is as 1b13k, but adding bottleneck-dim=256 to the output layers.
#  Definitely helpful:

# local/chain/compare_wer.sh exp/chain/tdnn_lstm1b13k_sp exp/chain/tdnn_lstm1b13l_sp
# System                tdnn_lstm1b13k_sp tdnn_lstm1b13l_sp
#WER dev93 (tgpr)                7.94      7.46
#WER dev93 (tg)                  7.68      7.09
#WER dev93 (big-dict,tgpr)       5.91      5.39
#WER dev93 (big-dict,fg)         5.56      4.94
#WER eval92 (tgpr)               5.65      5.44
#WER eval92 (tg)                 5.32      5.09
#WER eval92 (big-dict,tgpr)      3.49      3.15
#WER eval92 (big-dict,fg)        3.07      2.94
# Final train prob        -0.0491   -0.0513
# Final valid prob        -0.0600   -0.0599
# Final train prob (xent)   -0.7395   -0.7490
# Final valid prob (xent)   -0.7762   -0.7860
# Num-params                 5456076   4922060

# 1b13k is as 1b13d, but after a script fix: previously we were using the 'c'
# for the full-matrix part of the recurrence instead of the 'm'.

# 1b13d is as 1b13c, but a rerun after fixing a code bug whereby the natural gradient
# for the LinearComponent was turned off by default when initializing from config.
#   **Update: turns out there was no difference here, the code had been ignoring
#     that config variable.**
#
# It seems to optimize better, although the WER change is unclear.  However, it's
# interesting that the average objf in the individual training jobs (train.*.log) is not better-
# but in compute_prob_train.*.log it is.  It seems that the natural gradient interacts
# well with model averaging, which is what we found previously in the NG paper.


# local/chain/compare_wer.sh exp/chain/tdnn_lstm1b13c_sp exp/chain/tdnn_lstm1b13d_sp
# System                tdnn_lstm1b13c_sp tdnn_lstm1b13d_sp
#WER dev93 (tgpr)                7.68      7.86
#WER dev93 (tg)                  7.34      7.40
#WER dev93 (big-dict,tgpr)       5.42      5.65
#WER dev93 (big-dict,fg)         5.05      5.11
#WER eval92 (tgpr)               5.48      5.64
#WER eval92 (tg)                 5.26      5.17
#WER eval92 (big-dict,tgpr)      3.23      3.21
#WER eval92 (big-dict,fg)        2.82      2.84
# Final train prob        -0.0490   -0.0469
# Final valid prob        -0.0597   -0.0601
# Final train prob (xent)   -0.7549   -0.7424
# Final valid prob (xent)   -0.7910   -0.7920
# Num-params                 5456076   5456076
#
#
# 1b13c is as 1b13b, but after script change in which the lstmb layer was
# rewritten, adding memnorm and removing the scale of 4.0, along with some
#  more minor changes and streamlining/removing options.
#
# 1b13b is as 1b13, but a rerun after merging with the memnorm-and-combine
#   branch.  Slight difference in num-params is because of 300 vs 304.

# 1b13 is as 1b10 but reducing the bottleneck dim to 304
# (because I want to get in the habit of using multiples of 8).
#  WER seems improved.
#
#

# local/chain/compare_wer.sh exp/chain/tdnn_lstm1b10_sp exp/chain/tdnn_lstm1b13_sp
# System                tdnn_lstm1b10_sp tdnn_lstm1b13_sp
#WER dev93 (tgpr)                7.87      7.63
#WER dev93 (tg)                  7.48      7.46
#WER dev93 (big-dict,tgpr)       5.55      5.56
#WER dev93 (big-dict,fg)         5.25      5.09
#WER eval92 (tgpr)               5.44      5.48
#WER eval92 (tg)                 5.05      5.12
#WER eval92 (big-dict,tgpr)      3.24      3.17
#WER eval92 (big-dict,fg)        2.73      2.60
# Final train prob        -0.0463   -0.0470
# Final valid prob        -0.0561   -0.0565
# Final train prob (xent)   -0.7362   -0.7588
# Final valid prob (xent)   -0.7730   -0.7831
# Num-params                 5650636   5446348

# 1b10 is as 1b9 but reducing the cell and bottleneck dimension of LSTM layer from 512 to 384.
# Seems helpful on average-- nice!

# local/chain/compare_wer.sh exp/chain/tdnn_lstm1b9_sp exp/chain/tdnn_lstm1b10_sp
# System                tdnn_lstm1b9_sp tdnn_lstm1b10_sp
#WER dev93 (tgpr)                7.74      7.87
#WER dev93 (tg)                  7.46      7.48
#WER dev93 (big-dict,tgpr)       5.67      5.55
#WER dev93 (big-dict,fg)         5.31      5.25
#WER eval92 (tgpr)               5.60      5.44
#WER eval92 (tg)                 5.42      5.05
#WER eval92 (big-dict,tgpr)      3.47      3.24
#WER eval92 (big-dict,fg)        3.07      2.73
# Final train prob        -0.0413   -0.0463
# Final valid prob        -0.0543   -0.0561
# Final train prob (xent)   -0.6786   -0.7362
# Final valid prob (xent)   -0.7249   -0.7730
# Num-params                 7021644   5650636

# 1b9 is as 1b8 but adding batchnorm after the LSTM layer.. this is
#  to correct an oversight.
# 1b8 is as 1b7 but with quite a few layers removed.  WER effect is unclear.

# local/chain/compare_wer.sh exp/chain/tdnn_lstm1b7_sp exp/chain/tdnn_lstm1b8_sp
# System                tdnn_lstm1b7_sp tdnn_lstm1b8_sp
#WER dev93 (tgpr)                7.31      7.60
#WER dev93 (tg)                  7.10      7.25
#WER dev93 (big-dict,tgpr)       5.26      5.26
#WER dev93 (big-dict,fg)         4.64      4.93
#WER eval92 (tgpr)               5.48      5.32
#WER eval92 (tg)                 5.00      5.07
#WER eval92 (big-dict,tgpr)      3.35      3.31
#WER eval92 (big-dict,fg)        2.99      2.84
# Final train prob        -0.0483   -0.0533
# Final valid prob        -0.0573   -0.0627
# Final train prob (xent)   -0.7207   -0.8234
# Final valid prob (xent)   -0.7467   -0.8466
# Num-params                11752524   7021644

# 1b7 is as 1b6 but adding self-stabilize=true and normalize-type=none;
# and after a script-level change that scale 'c' by 4 before giving it
# to the W_all_a matrix (to see where all this came from, look at run_tdnn_lstm_1b16.sh
# in the mini_librispeech setup, although by the time you see this, that may no longer exist).
#
# 1b6 is as 1b3 but replacing renorm with batchnorm for the TDNN layers,
# and adding batchnorm to the LSTMB layers.  Effect on WER unclear but generally
# it's better.


# local/chain/compare_wer.sh exp/chain/tdnn_lstm1{a2,a3,b3,b6}_sp
# local/chain/compare_wer.sh exp/chain/tdnn_lstm1a2_sp exp/chain/tdnn_lstm1a3_sp exp/chain/tdnn_lstm1b3_sp exp/chain/tdnn_lstm1b6_sp
# System                tdnn_lstm1a2_sp tdnn_lstm1a3_sp tdnn_lstm1b3_sp tdnn_lstm1b6_sp
#WER dev93 (tgpr)                7.47      7.65      7.26      7.32
#WER dev93 (tg)                  7.29      7.24      6.96      6.98
#WER dev93 (big-dict,tgpr)       5.44      5.60      5.43      5.22
#WER dev93 (big-dict,fg)         4.98      5.04      4.97      4.86
#WER eval92 (tgpr)               5.78      5.21      5.30      5.14
#WER eval92 (tg)                 5.44      5.00      4.87      4.82
#WER eval92 (big-dict,tgpr)      3.35      3.23      3.42      3.24
#WER eval92 (big-dict,fg)        2.99      2.96      3.03      2.82
# Final train prob        -0.0447   -0.0410   -0.0484   -0.0503
# Final valid prob        -0.0566   -0.0518   -0.0594   -0.0599
# Final train prob (xent)   -0.6859   -0.6676   -0.7528   -0.7415
# Final valid prob (xent)   -0.7378   -0.7230   -0.8078   -0.7804
# Num-params                 9106252   9106252  11747916  11746380

# 1b3 is as 1a2 but with the same change as in a->b, replacing lstmp with lstmb
# 1a2 is as 1a but adding l2 regularization.

# this is a TDNN+LSTM chain system.
# It was modified from local/nnet3/tuning/run_tdnn_lstm_lfr_1a.sh with
# reference to ../../tedlium/s5_r2/local/chain/run_tdnn_lstm_1e.sh.
# Note: we're using the same hidden-layer sizes as
# ../../tedlium/s5_r2/local/chain/run_tdnn_lstm_1e.sh despite the
# fact that we'd normally choose a smaller model for a setup with
# less data, because the Tedlium model was probably on the small side.
# Note: we normally use more parameters for LSTM-containing than TDNN-only
# systems.

# steps/info/chain_dir_info.pl exp/chain/tdnn_lstm1a_sp
# exp/chain/tdnn_lstm1a_sp: num-iters=120 nj=2..10 num-params=9.1M dim=40+100->2889 combine=-0.047->-0.045 xent:train/valid[79,119,final]=(-0.684,-0.569,-0.564/-0.742,-0.668,-0.665) logprob:train/valid[79,119,final]=(-0.045,-0.035,-0.034/-0.058,-0.051,-0.051)

# The following compares:
# (nnet3 TDNN+LSTM, chain TDNN, this experiment == chain TDNN+LSTM)
# system.
# This is consistently better than the nnet3 TDNN+LSTM, but the
# difference with the chain TDNN is inconsistent.

# local/chain/compare_wer.sh --online exp/nnet3/tdnn_lstm1a_sp exp/chain/tdnn1a_sp exp/chain/tdnn_lstm1a_sp
# System                tdnn_lstm1a_sp tdnn1a_sp tdnn_lstm1a_sp
#WER dev93 (tgpr)                8.54      7.87      7.48
#             [online:]          8.57      8.02      7.49
#WER dev93 (tg)                  8.25      7.61      7.41
#             [online:]          8.34      7.70      7.40
#WER dev93 (big-dict,tgpr)       6.24      5.71      5.64
#             [online:]          6.40      5.60      5.70
#WER dev93 (big-dict,fg)         5.70      5.10      5.40
#             [online:]          5.77      5.21      5.19
#WER eval92 (tgpr)               6.52      5.23      5.67
#             [online:]          6.56      5.44      5.60
#WER eval92 (tg)                 6.13      4.87      5.46
#             [online:]          6.24      4.87      5.53
#WER eval92 (big-dict,tgpr)      3.88      3.24      3.69
#             [online:]          3.88      3.31      3.63
#WER eval92 (big-dict,fg)        3.38      2.71      3.28
#             [online:]          3.53      2.92      3.31
# Final train prob                  -0.0414   -0.0341
# Final valid prob                  -0.0634   -0.0506
# Final train prob (xent)             -0.8216   -0.5643
# Final valid prob (xent)             -0.9208   -0.6648


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
affix=1b  #affix for TDNN+LSTM directory e.g. "1a" or "1b", in case we change the configuration.
common_egs_dir=
reporting_email=

# LSTM/chain options
train_stage=-10
label_delay=8
xent_regularize=0.1

# training chunk-options
chunk_width=140,100,160
chunk_left_context=40
chunk_right_context=0

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
dir=exp/chain${nnet3_affix}/tdnn_lstm${affix}_sp
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
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  tdnn_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.005 bottleneck-dim=256"
  lstm_opts="l2-regularize=0.005 self-scale=2.0"


  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda delay=5 input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 $tdnn_opts dim=448
  relu-batchnorm-layer name=tdnn2 $tdnn_opts dim=448 input=Append(-1,0,1)
  relu-batchnorm-layer name=tdnn3 $tdnn_opts dim=448 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn4 $tdnn_opts dim=448 input=Append(-3,0,3)
  lstmb-layer name=lstm3 $lstm_opts cell-dim=384 bottleneck-dim=256 decay-time=20 delay=-3

  ## adding the layers for chain branch
  output-layer name=output input=lstm3 $output_opts output-delay=$label_delay include-log-softmax=false dim=$num_targets

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=lstm3 $output_opts output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 16 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5_r2/$dir/egs/storage $dir/egs/storage
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
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=6 \
    --trainer.deriv-truncate-margin=10 \
    --trainer.frames-per-iter=1500000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=10 \
    --trainer.optimization.initial-effective-lrate=0.0005 \
    --trainer.optimization.final-effective-lrate=0.00005 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.0 \
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
          --extra-left-context $chunk_left_context \
          --extra-right-context $chunk_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
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

if [ $stage -le 19 ]; then
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
          --acwt 1.0 --post-decode-acwt 10.0 \
          --frames-per-chunk 30 \
          --nj $nspk --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph_${lmtype} data/${data}_hires ${dir}/decode_looped_${lmtype}_${data_affix} || exit 1
      done
      steps/lmrescore.sh \
        --self-loop-scale 1.0 \
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

if $test_online_decoding && [ $stage -le 20 ]; then
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
