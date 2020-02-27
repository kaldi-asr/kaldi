#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh


# Train systems,
nj=30 # number of parallel jobs,
stage=0
. utils/parse_options.sh

set -euo pipefail

# Path where MALACH gets downloaded (or where locally available):
MALACH_DIR=/speech7/picheny5_nb/new_malach/malach_eng_speech_recognition/data

[ ! -r data/local/lm/final_lm ] && echo "Please, run 'run_prepare_shared.sh' first!" && exit 1
final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-9

PROCESSED_MALACH_DIR=$MALACH_DIR

# Prepare original data directories data/ihm/train_orig, etc.
if [ $stage -le 2 ]; then
  local/malach_data_prep.sh $PROCESSED_MALACH_DIR
  local/malach_scoring_data_prep.sh $PROCESSED_MALACH_DIR dev
fi

if [ $stage -le 3 ]; then
  for dset in train dev; do
    # this splits up the speakers 
    # into 30-second chunks.  It's like a very brain-dead form
    # of diarization; we can later replace it with 'real' diarization.
    seconds_per_spk_max=120

    # Note: the 30 on the next line should have been $seconds_per_spk_max
    # (thanks: Pavel Denisov.  This is a bug but before fixing it we'd have to
    # test the WER impact.  I suspect it will be quite small and maybe hard to
    # measure consistently.
    utils/data/modify_speaker_info.sh --seconds-per-spk-max $seconds_per_spk_max \
      data/${dset}_orig data/$dset
  done
fi

# Feature extraction,
if [ $stage -le 4 ]; then
  for dset in train dev; do
    steps/make_mfcc.sh --nj 15 --cmd "$train_cmd" data/$dset
    steps/compute_cmvn_stats.sh data/$dset
    utils/fix_data_dir.sh data/$dset
  done
fi

# monophone training
if [ $stage -le 5 ]; then
  # Full set 77h, reduced set 10.8h,
  utils/subset_data_dir.sh data/train 15000 data/train_15k

  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
    data/train_15k data/lang exp/mono
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/mono exp/mono_ali
fi

# context-dep. training with delta features.
if [ $stage -le 6 ]; then
  steps/train_deltas.sh --cmd "$train_cmd" \
    5000 80000 data/train data/lang exp/mono_ali exp/tri1
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri1 exp/tri1_ali
fi

if [ $stage -le 7 ]; then
  # LDA_MLLT
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 80000 data/train data/lang exp/tri1_ali exp/tri2
    steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri2 exp/tri2_ali
# Decode
   graph_dir=exp/tri2/graph_${LM}
  $decode_cmd --mem 4G $graph_dir/mkgraph.log \
     utils/mkgraph.sh data/lang_${LM} exp/tri2 $graph_dir
  steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/dev exp/tri2/decode_dev_${LM}
fi

if [ $stage -le 8 ]; then
  # LDA+MLLT+SAT
  steps/train_sat.sh --cmd "$train_cmd" \
    5000 80000 data/train data/lang exp/tri2_ali exp/tri3
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri3 exp/tri3_ali
fi

if [ $stage -le 9 ]; then
  # Decode the fMLLR system.
  graph_dir=exp/tri3/graph_${LM}
  $decode_cmd --mem 4G $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_${LM} exp/tri3 $graph_dir
  steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/dev exp/tri3/decode_dev_${LM}
fi

if [ $stage -le 10 ]; then
  # The following script cleans the data and produces cleaned data
  # in data/train_cleaned, and a corresponding system
  # in exp/tri3_cleaned.  It also decodes.
  #
  # Note: local/run_cleanup_segmentation.sh defaults to using 50 jobs,
  # you can reduce it using the --nj option if you want.
  local/run_cleanup_segmentation.sh 
fi

if [ $stage -le 11 ]; then
  ali_opt=
  local/chain/run_tdnn.sh $ali_opt 
fi

exit 0
