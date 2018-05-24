#!/bin/bash
#
# Copyright  2014  Nickolay V. Shmyrev
#            2014  Brno University of Technology (Author: Karel Vesely)
#            2016  Vincent Nguyen
#            2016  Johns Hopkins University (Author: Daniel Povey)
#            2018  Ewald Enzinger
#
# Apache 2.0
#
# Modified version of egs/tedlium/s5_r2/run.sh (commit edb1aae9457f6441a224dbc451bb8c5220dfefc7).

. ./cmd.sh
. ./path.sh


set -e -o pipefail -u

nj=40
stage=0

. utils/parse_options.sh # accept options

# Data preparation
if [ $stage -le 0 ]; then
  local/nnet3/bnf/tedlium_download_data.sh

  local/nnet3/bnf/tedlium_prepare_data.sh

  local/nnet3/bnf/tedlium_prepare_dict.sh

  utils/prepare_lang.sh data/local/dict_nosp \
    "<unk>" data/local/lang_nosp data/lang_nosp

  arpa_lm=db/cantab-TEDLIUM/cantab-TEDLIUM-pruned.lm3.gz
  utils/format_lm.sh data/lang_nosp $arpa_lm data/local/dict_nosp/lexicon.txt \
    data/lang_nosp_test
fi

# Feature extraction
if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --mfcc-config conf/mfcc_asr.conf --nj $nj --cmd "$train_cmd" data/tedlium
  steps/compute_cmvn_stats.sh data/tedlium
fi

# Well create a subset with 10k short segments to make flat-start training easier:
if [ $stage -le 2 ]; then
  utils/subset_data_dir.sh --shortest data/tedlium 10000 data/tedlium_10kshort
  utils/data/remove_dup_utts.sh 10 data/tedlium_10kshort data/tedlium_10kshort_nodup
fi

# Train
if [ $stage -le 3 ]; then
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
    data/tedlium_10kshort_nodup data/lang_nosp exp/mono
fi

if [ $stage -le 4 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/tedlium data/lang_nosp exp/mono exp/mono_ali
  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 30000 data/tedlium data/lang_nosp exp/mono_ali exp/tri1
fi

if [ $stage -le 5 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/tedlium data/lang_nosp exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    4000 50000 data/tedlium data/lang_nosp exp/tri1_ali exp/tri2
fi

if [ $stage -le 6 ]; then
  steps/get_prons.sh --cmd "$train_cmd" data/tedlium data/lang_nosp exp/tri2
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp exp/tri2/pron_counts_nowb.txt \
    exp/tri2/sil_counts_nowb.txt \
    exp/tri2/pron_bigram_counts_nowb.txt data/local/dict

  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
  cp data/lang_nosp_test/G.fst data/lang/
fi

if [ $stage -le 7 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/tedlium data/lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 data/tedlium data/lang exp/tri2_ali exp/tri3
fi

if [ $stage -le 8 ]; then
  # Although the nnet will be trained by high resolution data, we still have to
  # perturb the normal data to get the alignment _sp stands for speed-perturbed
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/tedlium data/tedlium_sp
  echo "$0: making MFCC features for low-resolution speed-perturbed data"
  steps/make_mfcc.sh --mfcc-config conf/mfcc_asr.conf --cmd "$train_cmd" --nj 10 data/tedlium_sp || exit 1;
  steps/compute_cmvn_stats.sh data/tedlium_sp || exit 1;
  utils/fix_data_dir.sh data/tedlium_sp
fi

if [ $stage -le 9 ]; then
  echo "$0: creating high-resolution MFCC features"
  mfccdir=data/tedlium_sp_hires/data
  utils/copy_data_dir.sh data/tedlium_sp data/tedlium_sp_hires
  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/tedlium_sp_hires || exit 1;

  steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
    --cmd "$train_cmd" data/tedlium_sp_hires || exit 1;
  steps/compute_cmvn_stats.sh data/tedlium_sp_hires || exit 1;
  utils/fix_data_dir.sh data/tedlium_sp_hires || exit 1;

  echo "$0: combining short segments of speed-perturbed high-resolution MFCC training data"
  # we have to combine short segments or we won't be able to train chain models
  # on those segments.
  utils/data/combine_short_segments.sh \
     data/tedlium_sp_hires 1.55 data/tedlium_sp_hires_comb
  cp data/tedlium_sp_hires/cmvn.scp data/tedlium_sp_hires_comb/
  utils/fix_data_dir.sh data/tedlium_sp_hires_comb/
fi

if [ $stage -le 10 ]; then
  echo "$0: combining short segments of speed-perturbed MFCC training data"
  # we have to combine short segments or we won't be able to train chain models
  # on those segments.
  utils/data/combine_short_segments.sh \
     data/tedlium_sp 1.55 data/tedlium_sp_comb

  # just copy over the CMVN to avoid having to recompute it.
  cp data/tedlium_sp/cmvn.scp data/tedlium_sp_comb/
  utils/fix_data_dir.sh data/tedlium_sp_comb/
  echo "$0: aligning with the perturbed, short-segment-combined low-resolution data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/tedlium_sp_comb data/lang exp/tri3 exp/tri3_ali
fi

if [ $stage -le 11 ]; then
  mkdir -p exp/nnet3
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info exp/tri3/tree |grep num-pdfs|awk '{print $2}')
  opts="l2-regularize=0.00005"
  output_opts="l2-regularize=0.005"

  mkdir -p exp/nnet3/configs
  cat <<EOF > exp/nnet3/configs/network.xconfig
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2) affine-transform-file=exp/nnet3/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 $opts dim=850
  relu-batchnorm-layer name=tdnn2 $opts dim=850 input=Append(-1,2)
  relu-batchnorm-layer name=tdnn3 $opts dim=850 input=Append(-3,3)
  relu-batchnorm-layer name=tdnn4 $opts dim=850 input=Append(-7,2)
  relu-batchnorm-layer name=tdnn5 $opts dim=850 input=Append(-3,3)
  relu-batchnorm-layer name=tdnn_bn dim=60
  relu-batchnorm-layer name=tdnn6 $opts dim=850
  output-layer name=output $output_opts dim=$num_targets max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file exp/nnet3/configs/network.xconfig --config-dir exp/nnet3/configs/
  cat <<EOF >> exp/nnet3/configs/vars
add_lda=false
include_log_softmax=false
EOF
fi

if [ $stage -le 12 ]; then
  # Train BNF nnet3 model
  steps/nnet3/train_dnn.py \
    --stage=-3 \
    --cmd="$train_cmd" \
    --feat.cmvn-opts="--norm-means=true --norm-vars=false" \
    --trainer.srand=123 \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=3 \
    --trainer.samples-per-iter=400000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=12 \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --trainer.optimization.minibatch-size=256,128 \
    --cleanup.remove-egs=false \
    --use-gpu=true \
    --feat-dir=data/tedlium_sp_hires \
    --ali-dir=exp/tri3_ali \
    --lang=data/lang \
    --dir=exp/nnet3  || exit 1;

  # Copy raw nnet3 model for bottleneck feature extraction
  nnet3-am-copy --raw=true exp/nnet3/final.mdl exp/nnet3/final.raw
fi

echo "$0: successfully trained BNF nnet3 model."
exit 0;
