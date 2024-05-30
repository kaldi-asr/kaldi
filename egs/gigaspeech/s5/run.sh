#!/usr/bin/env bash
# Copyright 2021  Xiaomi Corporation (Author: Yongqing Wang)
#                 Seasalt AI, Inc (Author: Guoguo Chen)
# Apache 2.0

set -e -o pipefail

stage=0

# GigaSpeech configurations.
gigaspeech_root=/data/GigaSpeech_Data/
gigaspeech_train_subset=XL
gigaspeech_test_sets="gigaspeech_dev gigaspeech_test"
gigaspeech_train_sets="gigaspeech_train_${gigaspeech_train_subset,,}"

# G2P models.
g2p_model=$gigaspeech_root/dict/g2p/g2p.model.4

# Experiment configurations.
train_nj=100
decode_nj=100
lm_order=4
lm_dir=data/local/lm
dict_dir=data/local/dict

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. ./utils/parse_options.sh || exit 1;

# Train/Dev/Test sets.
test_sets="$gigaspeech_test_sets"
train_sets="$gigaspeech_train_sets"
train_combined="$gigaspeech_train_sets"

if [ $stage -le 0 ]; then
  echo "======Prepare GigaSpeech START | current time : `date +%Y-%m-%d-%T`===="
  local/gigaspeech_data_prep.sh \
    --stage 0 \
    --train-subset $gigaspeech_train_subset \
    --test-sets "$gigaspeech_test_sets" \
    $gigaspeech_root data/ || exit 1;
  echo "======Prepare GigaSpeech END | current time : `date +%Y-%m-%d-%T`======"
fi

if [ $stage -le 1 ]; then
  echo "======Prepare Dictionary START | current time : `date +%Y-%m-%d-%T`===="
  [ ! -f $g2p_model ] && echo "$0: Cannot find G2P model $g2p_model" && exit 1
  local/prepare_dict.sh \
    --cmd "$train_cmd" --nj $train_nj \
    $g2p_model data/$train_combined $dict_dir || exit 1;
  echo "======Prepare Dictionary END | current time : `date +%Y-%m-%d-%T`======"
fi

if [ $stage -le 2 ]; then
  echo "======Train lm START | current time : `date +%Y-%m-%d-%T`=============="
  mkdir -p $lm_dir || exit 1;
  sed 's|\t| |' data/$train_combined/text |\
    cut -d " " -f 2- > $lm_dir/corpus.txt || exit 1;
  local/lm/train_lm.sh \
    --cmd "$train_cmd" --lm-order $lm_order \
    $lm_dir/corpus.txt $lm_dir || exit 1;
  echo "======Train lm END | current time : `date +%Y-%m-%d-%T`================"
fi

if [ $stage -le 3 ]; then
  echo "======Prepare lang START | current time : `date +%Y-%m-%d-%T`=========="
  utils/prepare_lang.sh $dict_dir \
    "<UNK>" data/local/lang_tmp data/lang || exit 1;

  utils/format_lm.sh data/lang $lm_dir/lm_${lm_order}gram.arpa.gz \
    $dict_dir/lexicon.txt data/lang_test || exit 1;
  echo "======Prepare lang END | current time : `date +%Y-%m-%d-%T`============"
fi

if [ $stage -le 4 ]; then
  echo "======Extract feat START | current time : `date +%Y-%m-%d-%T`=========="
  mfccdir=mfcc
  if [[  $(hostname -f) == tj1-asr-train-dev* ]]; then
    prefix1=data-tmp-TTL20/$(date +%Y%m%d)/$USER/kaldi-data
    prefix2=$(basename $(pwd))/$(hostname -f)_$(date +%Y%m%d_%H%M%S)_$$
    utils/create_split_dir.pl \
      /home/storage{{30..36},{40..49}}/$prefix1/$prefix2/storage/ \
      $mfccdir/storage || exit 1;
  fi

  for part in $test_sets $train_combined; do
    steps/make_mfcc.sh --cmd "$train_cmd" \
      --nj $train_nj data/$part exp/make_mfcc/$part $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh \
      data/$part exp/make_mfcc/$part $mfccdir || exit 1;
  done
  echo "======Extract feat END | current time : `date +%Y-%m-%d-%T`============"
fi

if [ $stage -le 5 ]; then
  echo "======Subset train data START | current time : `date +%Y-%m-%d-%T`====="
  # Make some small data subsets for early system-build stages.  Note, there are
  # 8283k utterances in the train directory which has 10000 hours of data. For
  # the monophone stages we select the shortest utterances, which should make it
  # easier to align the data from a flat start.
  total_num=`wc -l <data/$train_combined/utt2spk`

  subset_num=$((total_num/64))
  if [ $total_num -lt 20000 ]; then
    subset_num=$total_num
  else
    [ $subset_num -gt 100000 ] && subset_num=100000
    [ $subset_num -lt 20000 ] && subset_num=20000
  fi
  utils/subset_data_dir.sh --shortest \
    data/$train_combined $subset_num data/${train_combined}_1d64 || exit 1;

  subset_num=$((total_num/32))
  if [ $total_num -lt 60000 ]; then
    subset_num=$total_num
  else
    [ $subset_num -gt 250000 ] && subset_num=250000
    [ $subset_num -lt 60000 ] && subset_num=60000
  fi
  utils/subset_data_dir.sh \
    data/$train_combined $subset_num data/${train_combined}_1d32 || exit 1;

  subset_num=$((total_num/16))
  if [ $total_num -lt 125000 ]; then
    subset_num=$total_num
  else
    [ $subset_num -gt 500000 ] && subset_num=500000
    [ $subset_num -lt 125000 ] && subset_num=125000
  fi
  utils/subset_data_dir.sh \
    data/$train_combined $subset_num data/${train_combined}_1d16 || exit 1;

  subset_num=$((total_num/8))
  if [ $total_num -lt 250000 ]; then
    subset_num=$total_num
  else
    [ $subset_num -gt 1000000 ] && subset_num=1000000
    [ $subset_num -lt 250000 ] && subset_num=250000
  fi
  utils/subset_data_dir.sh \
    data/$train_combined $subset_num data/${train_combined}_1d8 || exit 1;
  echo "======Subset train data END | current time : `date +%Y-%m-%d-%T`======="
fi

if [ $stage -le 6 ]; then
  echo "======Train mono START | current time : `date +%Y-%m-%d-%T`============"
  steps/train_mono.sh \
    --boost-silence 1.25 --nj $train_nj --cmd "$train_cmd" \
    data/${train_combined}_1d64 data/lang exp/mono || exit 1;
  echo "======Train mono END | current time : `date +%Y-%m-%d-%T`=============="
  {
    utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1;
    for part in $test_sets; do
      [ ! -d data/$part ] &&\
        echo "$0: Decoder mono Error: no such dir data/$part" && exit 1;
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        exp/mono/graph data/${part} exp/mono/decode_${part} || exit 1;
      cat exp/mono/decode_${part}/wer_* | utils/best_wer.sh |\
        sed "s/^/mono\t/" > exp/mono/decode_${part}/wer.txt || exit 1;
    done
  } &
fi

if [ $stage -le 7 ]; then
  echo "======Train tri1b START | current time : `date +%Y-%m-%d-%T`==========="
  steps/align_si.sh \
    --boost-silence 1.25 --nj $train_nj --cmd "$train_cmd" \
    data/${train_combined}_1d32 data/lang \
    exp/mono exp/mono_ali_train_1d32 || exit 1;

  steps/train_deltas.sh \
    --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 \
    data/${train_combined}_1d32 data/lang \
    exp/mono_ali_train_1d32 exp/tri1b || exit 1;
  echo "======Train tri1b END | current time : `date +%Y-%m-%d-%T`============="
  {
    utils/mkgraph.sh data/lang_test exp/tri1b exp/tri1b/graph || exit 1;
    for part in $test_sets; do
      [ ! -d data/$part ] &&\
        echo "$0: Decoder tri1b Error: no such dir data/$part" && exit 1;
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        exp/tri1b/graph data/${part} exp/tri1b/decode_${part} || exit 1;
      cat exp/tri1b/decode_${part}/wer_* | utils/best_wer.sh |\
        sed "s/^/tri1b\t/" > exp/tri1b/decode_${part}/wer.txt || exit 1;
    done
  } &
fi

if [ $stage -le 8 ]; then
  echo "======Train tri2b START | current time : `date +%Y-%m-%d-%T`==========="
  steps/align_si.sh \
    --nj $train_nj --cmd "$train_cmd" \
    data/${train_combined}_1d16 data/lang \
    exp/tri1b exp/tri1b_ali_train_1d16 || exit 1;

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/${train_combined}_1d16 data/lang \
    exp/tri1b_ali_train_1d16 exp/tri2b || exit 1;
  echo "======Train tri2b END | current time : `date +%Y-%m-%d-%T`============="
  {
    utils/mkgraph.sh data/lang_test exp/tri2b exp/tri2b/graph || exit 1
    for part in $test_sets; do
      [ ! -d data/$part ] &&\
        echo "$0: Decoder tri2b Error: no such dir data/$part" && exit 1;
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        exp/tri2b/graph data/${part} exp/tri2b/decode_${part} || exit 1;
      cat exp/tri2b/decode_${part}/wer_* | utils/best_wer.sh |\
        sed "s/^/tri2b\t/" > exp/tri2b/decode_${part}/wer.txt || exit 1;
    done
  } &
fi

if [ $stage -le 9 ]; then
  echo "======Train tri3b START | current time : `date +%Y-%m-%d-%T`==========="
  steps/align_si.sh \
    --nj $train_nj --cmd "$train_cmd" --use-graphs true \
    data/${train_combined}_1d16 data/lang \
    exp/tri2b exp/tri2b_ali_train_1d16 || exit 1;

  steps/train_sat.sh \
    --cmd "$train_cmd" 2500 15000 \
    data/${train_combined}_1d16 data/lang \
    exp/tri2b_ali_train_1d16 exp/tri3b || exit 1;
  echo "======Train tri3b END | current time : `date +%Y-%m-%d-%T`============="
  {
    utils/mkgraph.sh data/lang_test exp/tri3b exp/tri3b/graph || exit 1;
    for part in $test_sets; do
      [ ! -d data/$part ] &&\
        echo "$0: Decoder tri3b Error: no such dir data/$part" && exit 1;
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        exp/tri3b/graph data/$part exp/tri3b/decode_${part} || exit 1;
      cat exp/tri3b/decode_${part}/wer_* | utils/best_wer.sh |\
        sed "s/^/tri3b\t/" > exp/tri3b/decode_${part}/wer.txt || exit 1;
    done
  } &
fi

if [ $stage -le 10 ]; then
  echo "======Train tri4b START | current time : `date +%Y-%m-%d-%T`==========="
  steps/align_fmllr.sh \
    --nj $train_nj --cmd "$train_cmd" \
    data/${train_combined}_1d8 data/lang \
    exp/tri3b exp/tri3b_ali_train_1d8 || exit 1;

  steps/train_sat.sh \
    --cmd "$train_cmd" 4200 40000 \
    data/${train_combined}_1d8 data/lang \
    exp/tri3b_ali_train_1d8 exp/tri4b || exit 1;
  echo "======Train tri4b END | current time : `date +%Y-%m-%d-%T`============="
  {
    utils/mkgraph.sh data/lang_test exp/tri4b exp/tri4b/graph || exit 1
    for part in $test_sets; do
      [ ! -d data/$part ] &&\
        echo "$0: Decoder tri4b Error: no such dir data/$part" && exit 1;
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        exp/tri4b/graph data/${part} exp/tri4b/decode_${part} || exit 1;
      cat exp/tri4b/decode_${part}/wer_* | utils/best_wer.sh |\
        sed "s/^/tri4b\t/" > exp/tri4b/decode_${part}/wer.txt || exit 1;
    done
  } &
fi

if [ $stage -le 11 ]; then
  echo "======Train chain START | current time : `date +%Y-%m-%d-%T`==========="
  local/chain/run_cnn_tdnn.sh \
    --stage 0 \
    --train-stage -10 \
    --get-egs-stage -10 \
    --train_set ${train_combined} \
    --gmm tri4b \
    --test-sets "$test_sets" \
    --frames_per_iter 3000000 \
    --num-jobs-initial 16 \
    --num-jobs-final 16 \
    --initial-effective-lrate 0.00015 \
    --final-effective-lrate 0.000015 || exit 1;
  echo "======Train chain END | current time : `date +%Y-%m-%d-%T`============="
fi

wait;
echo "$0: Done"
