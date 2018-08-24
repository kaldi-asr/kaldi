#!/bin/bash

# 1a54 is as 1a53 but halving the target-rms of the ivector stuff to 0.025.
# No clear difference.

# local/chain/compare_wer.sh --online exp/chain/tdnn1h_sp exp/chain/cnn_tdnn1a47_sp exp/chain/cnn_tdnn1a47b_sp exp/chain/cnn_tdnn1a53_sp exp/chain/cnn_tdnn1a53b_sp exp/chain/cnn_tdnn1a54_sp exp/chain/cnn_tdnn1a54b_sp
# System                tdnn1h_sp cnn_tdnn1a47_sp cnn_tdnn1a47b_sp cnn_tdnn1a53_sp cnn_tdnn1a53b_sp cnn_tdnn1a54_sp cnn_tdnn1a54b_sp
#WER dev_clean_2 (tgsmall)      12.09     11.42     11.58     11.39     11.14     11.15     11.32
#             [online:]         12.11     11.40     11.51     11.30     11.17     11.17     11.34
#WER dev_clean_2 (tglarge)       8.59      7.93      7.97      7.81      7.87      7.79      7.73
#             [online:]          8.76      8.02      8.12      7.72      7.89      7.80      7.74
# Final train prob        -0.0493   -0.0455   -0.0460   -0.0459   -0.0462   -0.0467   -0.0464
# Final valid prob        -0.0805   -0.0780   -0.0779   -0.0782   -0.0793   -0.0789   -0.0785
# Final train prob (xent)   -1.1730   -1.0483   -1.0442   -1.0630   -1.0690   -1.0767   -1.0774
# Final valid prob (xent)   -1.3872   -1.2909   -1.2944   -1.3032   -1.3110   -1.3070   -1.3113
# Num-params                 5207856   4591616   4591616   4492816   4492816   4492816   4492816


# 1a53 is as 1a47 but removing the previous way we had included the ivector.
# Better!
# local/chain/compare_wer.sh --online exp/chain/cnn_tdnn1a47_sp exp/chain/cnn_tdnn1a47b_sp exp/chain/cnn_tdnn1a53_sp exp/chain/cnn_tdnn1a53b_sp
# System                cnn_tdnn1a47_sp cnn_tdnn1a47b_sp cnn_tdnn1a53_sp cnn_tdnn1a53b_sp
#WER dev_clean_2 (tgsmall)      11.42     11.58     11.39     11.14
#             [online:]         11.40     11.51     11.30     11.17
#WER dev_clean_2 (tglarge)       7.93      7.97      7.81      7.87
#             [online:]          8.02      8.12      7.72      7.89
# Final train prob        -0.0455   -0.0460   -0.0459   -0.0462
# Final valid prob        -0.0780   -0.0779   -0.0782   -0.0793
# Final train prob (xent)   -1.0483   -1.0442   -1.0630   -1.0690
# Final valid prob (xent)   -1.2909   -1.2944   -1.3032   -1.3110
# Num-params                 4591616   4591616   4492816   4492816


# 1a47 is as 1a45 but reducing target-rms in the ivector-batchnorm component from
#  0.1 to 0.05.  Helpful!
# local/chain/compare_wer.sh --online exp/chain/cnn_tdnn1a30_sp exp/chain/cnn_tdnn1a30b_sp exp/chain/cnn_tdnn1a41_sp exp/chain/cnn_tdnn1a42_sp exp/chain/cnn_tdnn1a45_sp exp/chain/cnn_tdnn1a45b_sp exp/chain/cnn_tdnn1a47_sp exp/chain/cnn_tdnn1a47b_sp
# System                cnn_tdnn1a30_sp cnn_tdnn1a30b_sp cnn_tdnn1a41_sp cnn_tdnn1a42_sp cnn_tdnn1a45_sp cnn_tdnn1a45b_sp cnn_tdnn1a47_sp cnn_tdnn1a47b_sp
#WER dev_clean_2 (tgsmall)      12.04     12.01     11.69     11.29     11.73     11.47     11.42     11.58
#             [online:]         12.00     12.01     11.76     11.36     11.71     11.44     11.40     11.51
#WER dev_clean_2 (tglarge)       8.36      8.32      8.17      7.94      8.21      8.03      7.93      7.97
#             [online:]          8.46      8.45      8.27      8.05      8.30      8.07      8.02      8.12
# Final train prob        -0.0459   -0.0452   -0.0455   -0.0452   -0.0457   -0.0454   -0.0455   -0.0460
# Final valid prob        -0.0809   -0.0800   -0.0788   -0.0779   -0.0793   -0.0795   -0.0780   -0.0779
# Final train prob (xent)   -1.0709   -1.0708   -1.0446   -1.0330   -1.0429   -1.0393   -1.0483   -1.0442
# Final valid prob (xent)   -1.3090   -1.3158   -1.2952   -1.2786   -1.2892   -1.2938   -1.2909   -1.2944
# Num-params                 4569456   4569456   4690384   4691664   4591616   4591616   4591616   4591616


# local/chain/compare_wer.sh --online exp/chain/cnn_tdnn1a30_sp exp/chain/cnn_tdnn1a30b_sp exp/chain/cnn_tdnn1a41_sp exp/chain/cnn_tdnn1a42_sp exp/chain/cnn_tdnn1a45_sp exp/chain/cnn_tdnn1a45b_sp exp/chain/cnn_tdnn1a47_sp
# System                cnn_tdnn1a30_sp cnn_tdnn1a30b_sp cnn_tdnn1a41_sp cnn_tdnn1a42_sp cnn_tdnn1a45_sp cnn_tdnn1a45b_sp cnn_tdnn1a47_sp
#WER dev_clean_2 (tgsmall)      12.04     12.01     11.69     11.29     11.73     11.47     11.42
#             [online:]         12.00     12.01     11.76     11.36     11.71     11.44     11.40
#WER dev_clean_2 (tglarge)       8.36      8.32      8.17      7.94      8.21      8.03      7.93
#             [online:]          8.46      8.45      8.27      8.05      8.30      8.07      8.02
# Final train prob        -0.0459   -0.0452   -0.0455   -0.0452   -0.0457   -0.0454   -0.0455
# Final valid prob        -0.0809   -0.0800   -0.0788   -0.0779   -0.0793   -0.0795   -0.0780
# Final train prob (xent)   -1.0709   -1.0708   -1.0446   -1.0330   -1.0429   -1.0393   -1.0483
# Final valid prob (xent)   -1.3090   -1.3158   -1.2952   -1.2786   -1.2892   -1.2938   -1.2909
# Num-params                 4569456   4569456   4690384   4691664   4591616   4591616   4591616


# local/chain/compare_wer.sh --online exp/chain/cnn_tdnn1a30_sp exp/chain/cnn_tdnn1a30b_sp exp/chain/cnn_tdnn1a41_sp exp/chain/cnn_tdnn1a42_sp exp/chain/cnn_tdnn1a45_sp exp/chain/cnn_tdnn1a45b_sp exp/chain/cnn_tdnn1a47_sp
# System                cnn_tdnn1a30_sp cnn_tdnn1a30b_sp cnn_tdnn1a41_sp cnn_tdnn1a42_sp cnn_tdnn1a45_sp cnn_tdnn1a45b_sp cnn_tdnn1a47_sp
#WER dev_clean_2 (tgsmall)      12.04     12.01     11.69     11.29     11.73     11.47     11.42
#             [online:]         12.00     12.01     11.76     11.36     11.71               11.40
#WER dev_clean_2 (tglarge)       8.36      8.32      8.17      7.94      8.21      8.03      7.93
#             [online:]          8.46      8.45      8.27      8.05      8.30                8.02
# Final train prob        -0.0459   -0.0452   -0.0455   -0.0452   -0.0457   -0.0454   -0.0455
# Final valid prob        -0.0809   -0.0800   -0.0788   -0.0779   -0.0793   -0.0795   -0.0780
# Final train prob (xent)   -1.0709   -1.0708   -1.0446   -1.0330   -1.0429   -1.0393   -1.0483
# Final valid prob (xent)   -1.3090   -1.3158   -1.2952   -1.2786   -1.2892   -1.2938   -1.2909
# Num-params                 4569456   4569456   4690384   4691664   4591616   4591616   4591616



# 1a45 is 1a41 but making the ivector adaptation be done before, not after,
#  the first cnn layer, as additional filters; reverting the
#  num-filters of the 1st cnn layer from 32 to 48.
# 1a41 is 1a30 but changing how the ivector adaptation is done (adding another branch);
#   reducing the 1st num-filters-out from 48 to 32 to save parameters.
# 1a30 is as 1a29 but adding another cnn layer with subsampling.
# Promising.
# local/chain/compare_wer.sh --online exp/chain/tdnn1h_sp exp/chain/tdnn1h2_sp exp/chain/cnn_tdnn1a24_sp exp/chain/cnn_tdnn1a24b_sp exp/chain/cnn_tdnn1a29_sp exp/chain/cnn_tdnn1a29b_sp exp/chain/cnn_tdnn1a30_sp exp/chain/cnn_tdnn1a30b_sp
# System                tdnn1h_sp tdnn1h2_sp cnn_tdnn1a24_sp cnn_tdnn1a24b_sp cnn_tdnn1a29_sp cnn_tdnn1a29b_sp cnn_tdnn1a30_sp cnn_tdnn1a30b_sp
#WER dev_clean_2 (tgsmall)      13.18     13.04     11.95     11.86     12.19     11.95     12.04     12.01
#             [online:]         13.03     12.97     11.99     11.96     12.19     11.94     12.00     12.01
#WER dev_clean_2 (tglarge)       9.18      9.16      8.57      8.54      8.62      8.45      8.36      8.32
#             [online:]          9.29      9.24      8.63      8.57      8.68      8.53      8.46      8.45
# Final train prob        -0.0531   -0.0590   -0.0461   -0.0455   -0.0456   -0.0461   -0.0459   -0.0452
# Final valid prob        -0.0844   -0.0865   -0.0800   -0.0798   -0.0800   -0.0792   -0.0809   -0.0800
# Final train prob (xent)   -1.5244   -1.7771   -1.0776   -1.0781   -1.0778   -1.0792   -1.0709   -1.0708
# Final valid prob (xent)   -1.7447   -1.9611   -1.3131   -1.3190   -1.3153   -1.3210   -1.3090   -1.3158
# Num-params                 3512112   3512112   4474688   4474688   4495600   4495600   4569456   4569456

# 1a29 is as 1a24 but increasing the num-filters-out for the first two
#  layers from 32 to 48.
# 1a24 is as 1a23 but changing offsets for the last cnn layer to be -1,0,1,
#  as in 14->22.  Better, on average.
# local/chain/compare_wer.sh --online exp/chain/tdnn1h_sp exp/chain/tdnn1h2_sp exp/chain/cnn_tdnn1a23_sp exp/chain/cnn_tdnn1a23b_sp exp/chain/cnn_tdnn1a24_sp exp/chain/cnn_tdnn1a24b_sp
# System                tdnn1h_sp tdnn1h2_sp cnn_tdnn1a23_sp cnn_tdnn1a23b_sp cnn_tdnn1a24_sp cnn_tdnn1a24b_sp
#WER dev_clean_2 (tgsmall)      13.18     13.04     12.15     12.11     11.95     11.86
#             [online:]         13.03     12.97     12.18     12.07     11.99     11.96
#WER dev_clean_2 (tglarge)       9.18      9.16      8.57      8.47      8.57      8.54
#             [online:]          9.29      9.24      8.64      8.50      8.63      8.57
# Final train prob        -0.0531   -0.0590   -0.0456   -0.0462   -0.0461   -0.0455
# Final valid prob        -0.0844   -0.0865   -0.0800   -0.0802   -0.0800   -0.0798
# Final train prob (xent)   -1.5244   -1.7771   -1.0691   -1.0683   -1.0776   -1.0781
# Final valid prob (xent)   -1.7447   -1.9611   -1.3190   -1.3108   -1.3131   -1.3190
# Num-params                 3512112   3512112   4474688   4474688   4474688   4474688

# 1a23 is as 1a14 but for the last cnn layer (cnn5), using twice the num-filters
#  plus subsampling on the output.
# A bit better, on average!
# local/chain/compare_wer.sh --online exp/chain/tdnn1h_sp exp/chain/tdnn1h2_sp exp/chain/cnn_tdnn1a14_sp exp/chain/cnn_tdnn1a14b_sp exp/chain/cnn_tdnn1a23_sp exp/chain/cnn_tdnn1a23b_sp
# System                tdnn1h_sp tdnn1h2_sp cnn_tdnn1a14_sp cnn_tdnn1a14b_sp cnn_tdnn1a23_sp cnn_tdnn1a23b_sp
#WER dev_clean_2 (tgsmall)      13.18     13.04     12.14     12.39     12.15     12.11
#             [online:]         13.03     12.97     12.10     12.38     12.18     12.07
#WER dev_clean_2 (tglarge)       9.18      9.16      8.44      8.69      8.57      8.47
#             [online:]          9.29      9.24      8.58      8.81      8.64      8.50
# Final train prob        -0.0531   -0.0590   -0.0455   -0.0460   -0.0456   -0.0462
# Final valid prob        -0.0844   -0.0865   -0.0806   -0.0802   -0.0800   -0.0802
# Final train prob (xent)   -1.5244   -1.7771   -1.0792   -1.0763   -1.0691   -1.0683
# Final valid prob (xent)   -1.7447   -1.9611   -1.3221   -1.3173   -1.3190   -1.3108
# Num-params                 3512112   3512112   4456224   4456224   4474688   4474688

# 1a14 is as 1a13 but with an extra tdnn-f layer.  Better!
# local/chain/compare_wer.sh --online exp/chain/tdnn1h_sp exp/chain/tdnn1h2_sp exp/chain/cnn_tdnn1a13_sp exp/chain/cnn_tdnn1a14_sp
# System                tdnn1h_sp tdnn1h2_sp cnn_tdnn1a13_sp cnn_tdnn1a14_sp
#WER dev_clean_2 (tgsmall)      13.18     13.04     12.21     12.14
#             [online:]         13.03     12.97     12.26     12.10
#WER dev_clean_2 (tglarge)       9.18      9.16      8.65      8.44
#             [online:]          9.29      9.24      8.67      8.58
# Final train prob        -0.0531   -0.0590   -0.0459   -0.0455
# Final valid prob        -0.0844   -0.0865   -0.0810   -0.0806
# Final train prob (xent)   -1.5244   -1.7771   -1.0901   -1.0792
# Final valid prob (xent)   -1.7447   -1.9611   -1.3328   -1.3221
# Num-params                 3512112   3512112   4160544   4456224

# 1a13 is as 1a12 but using the same l2 values for the first layers as for the
#   later ones (more l2).
# 1a12 is as 1a11 but making the first TDNN-F layer non-splicing and restoring
#  the 640's to 768's.
# 1a11 is as 1a10 but adding some l2 to the CNN layers and to the TDNN layers
#  for the ivector training.
# run_cnn_tdnn_1a10.sh is as run_cnn_tdnn_1a.sh but reducing the 768's to 640
#  to make the num-params similar to the tdnn1h experiment (run_cnn_tdnn_1a.sh was overfitting
#  a bit).
#
# run_cnn_tdnn_1a.sh is modified from run_tdnn_1h.sh, but adding CNN layers
#  near the beginning.

# 1h is as 1g but a re-tuned model based on resnet-style TDNN-F layers with
# bypass connections.  Below, 1h2 is just a rerun of 1h with a different --affix
# option, to give some idea of the run-to-run variation.

# local/chain/compare_wer.sh --online exp/chain/tdnn1g_sp exp/chain/tdnn1h_sp exp/chain/tdnn1h2_sp
# System                tdnn1g_sp tdnn1h_sp tdnn1h2_sp
#WER dev_clean_2 (tgsmall)      13.50     13.18     13.04
#             [online:]         13.52     13.03     12.97
#WER dev_clean_2 (tglarge)       9.79      9.18      9.16
#             [online:]          9.79      9.29      9.24
# Final train prob        -0.0460   -0.0531   -0.0590
# Final valid prob        -0.0892   -0.0844   -0.0865
# Final train prob (xent)   -1.1739   -1.5244   -1.7771
# Final valid prob (xent)   -1.4487   -1.7447   -1.9611
# Num-params                 6234672   3512112   3512112

# steps/info/chain_dir_info.pl  exp/chain/tdnn1{g,h,h2}_sp
# exp/chain/tdnn1g_sp: num-iters=25 nj=2..5 num-params=6.2M dim=40+100->2328 combine=-0.056->-0.055 (over 3) xent:train/valid[15,24,final]=(-1.50,-1.23,-1.17/-1.73,-1.52,-1.45) logprob:train/valid[15,24,final]=(-0.063,-0.051,-0.046/-0.101,-0.094,-0.089)
# exp/chain/tdnn1h_sp: num-iters=34 nj=2..5 num-params=3.5M dim=40+100->2328 combine=-0.055->-0.050 (over 4) xent:train/valid[21,33,final]=(-1.97,-1.57,-1.52/-2.11,-1.78,-1.74) logprob:train/valid[21,33,final]=(-0.080,-0.061,-0.053/-0.106,-0.096,-0.084)
# exp/chain/tdnn1h2_sp: num-iters=34 nj=2..5 num-params=3.5M dim=40+100->2328 combine=-0.062->-0.056 (over 4) xent:train/valid[21,33,final]=(-2.21,-1.78,-1.78/-2.34,-1.96,-1.96) logprob:train/valid[21,33,final]=(-0.086,-0.066,-0.059/-0.110,-0.098,-0.087)

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
decode_nj=10
train_set=train_clean_5
test_sets=dev_clean_2
gmm=tri3b
nnet3_affix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1a54   # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
# training chunk-options
chunk_width=140,100,160
dropout_schedule='0,0@0.20,0.3@0.50,0'
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
                                  --nnet3-affix "$nnet3_affix" || exit 1;

# Problem: We have removed the "train_" prefix of our training set in
# the alignment directory names! Bad!
gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/cnn_tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

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

if [ $stage -le 11 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 75 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
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
    --cmd "$train_cmd" 3500 ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi


if [ $stage -le 13 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  cnn_opts="l2-regularize=0.03"
  ivector_affine_opts="l2-regularize=0.03"
  tdnn_opts="l2-regularize=0.03 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_first_opts="l2-regularize=0.03 dropout-proportion=0.0 bypass-scale=0.0"
  tdnnf_opts="l2-regularize=0.03 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.03 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.03"
  output_opts="l2-regularize=0.015"

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
  # information bottleneck doesn't become a problem.  (we use time-stride=0 so no splicing, to
  # limit the num-parameters).
  tdnnf-layer name=tdnnf7 $tdnnf_first_opts dim=768 bottleneck-dim=192 time-stride=0
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  linear-component name=prefinal-l dim=192 $linear_opts

  ## adding the layers for chain branch
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=192 big-dim=768
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  # adding the layers for xent branch
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=192 big-dim=768
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 14 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/fs0{1,2}/$USER/kaldi-data/egs/mini_librispeech-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
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
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=20 \
    --trainer.frames-per-iter=3000000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=5 \
    --trainer.optimization.initial-effective-lrate=0.002 \
    --trainer.optimization.final-effective-lrate=0.0002 \
    --trainer.num-chunk-per-minibatch=128,64 \
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

if [ $stage -le 15 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_tgsmall \
    $tree_dir $tree_dir/graph_tgsmall || exit 1;
fi

if [ $stage -le 16 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph_tgsmall data/${data}_hires ${dir}/decode_tgsmall_${data} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_{tgsmall,tglarge} \
       data/${data}_hires ${dir}/decode_{tgsmall,tglarge}_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

# Not testing the 'looped' decoding separately, because for
# TDNN systems it would give exactly the same results as the
# normal decoding.

if $test_online_decoding && [ $stage -le 17 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3${nnet3_affix}/extractor ${dir} ${dir}_online

  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      # note: we just give it "data/${data}" as it only uses the wav.scp, the
      # feature type does not matter.
      steps/online/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --nj $nspk --cmd "$decode_cmd" \
        $tree_dir/graph_tgsmall data/${data} ${dir}_online/decode_tgsmall_${data} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_{tgsmall,tglarge} \
       data/${data}_hires ${dir}_online/decode_{tgsmall,tglarge}_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi


exit 0;
