#!/usr/bin/env bash

# Copyright 2021  ASLP, NWPU (Author: Hang Lyu)
#                 Mobvoi Inc (Author: Binbin Zhang)
# Apache 2.0


# you might not want to do this for interactive shells.
set -e
set -o pipefail


# ---- Configurations ----
stage=0

wenetspeech_root=/home/work_nfs4_ssd/zhyyao/corpus/wenetspeech
wenetspeech_train_subset_identifier=L  # L|M|S|W represents the
                                       # large|med|small size for strong
                                       # label data and weak label data

# Bear in mind, you can prepare different size training sets together as follows.
wenetspeech_train_subsets=
for label in $wenetspeech_train_subset_identifier; do
  wenetspeech_train_subsets="$wenetspeech_train_subsets train_${label,,?}"
done

# We assume only one 'train_subset' is specified.
# convert uppercase to lowercase
wenetspeech_train_set="train_${wenetspeech_train_subset_identifier,,?}"

# For aishell1, it has 20 speakers, we prefer to test it separately with
# local/wenetspeech_test_aishell.sh
wenetspeech_test_sets="dev test_meeting test_net"

train_nj=50
decode_nj=50
ngram_order=3

lm_dir=data/local/lm
dict_dir=data/local/dict
# ---- Configurations End ----

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $stage -le 1 ]; then
  # prepare the dict
  local/wenetspeech_dict_prep.sh $dict_dir
fi


if [ $stage -le 2 ]; then
  # format the data as Kaldi data directories
  local/wenetspeech_data_prep.sh \
    --stage 0 \
    --do-segmentation true \
    --add-dataset false \
    --train-subset "$wenetspeech_train_subset_identifier" \
    "$wenetspeech_root" data
fi


if [ $stage -le 3 ]; then
  # prepare the n-gram language model
  local/wenetspeech_train_lm.sh \
    --ngram-order $ngram_order \
    $dict_dir/lexicon.txt data/corpus/lm_text $lm_dir
fi

if [ $stage -le 4 ]; then
  # prepare lang
  utils/prepare_lang.sh $dict_dir \
    "<UNK>" data/local/lang_tmp data/lang
  utils/format_lm.sh data/lang $lm_dir/lm_${ngram_order}gram.arpa.gz \
    $dict_dir/lexicon.txt data/lang_test
fi

if [ $stage -le 5 ]; then
  mfcc_dir=mfcc
  for set in $wenetspeech_train_subsets $wenetspeech_test_sets; do
    steps/make_mfcc.sh --cmd "$train_cmd" \
      --nj $train_nj data/$set exp/make_mfcc/$set $mfcc_dir

    steps/compute_cmvn_stats.sh \
      data/$set exp/make_mfcc/$set $mfcc_dir

    utils/fix_data_dir.sh data/$set
  done
fi

# Here, make some sub-datasets for early system-build stages.
# The 'train_s' dataset has    151,600 utterances.    ~100hr
# The 'train_m' dataset has  1,514,500 utterances.  ~1,000hr
# The 'train_l' dataset has 14,625,245 utterances. ~10,000hr
# For monophone gmm-hmm model training, we select the shortest utterances,
# which should make it easier to align from a flat start.

if [ $stage -le 6 ]; then
  # split the subsets.
  total_num=`wc -l <data/$wenetspeech_train_set/utt2spk`

  # 50k utterances for monophone
  subset_num=50000
  if [ $subset_num -ge $total_num ]; then
    # use -sfr will be more readable, but macOS doesn't support '-r'.
    ln -sf $wenetspeech_train_set data/${wenetspeech_train_set}_50k
  else
    utils/subset_data_dir.sh --shortest \
      data/$wenetspeech_train_set $subset_num data/${wenetspeech_train_set}_50k
  fi

  # 125k utterances for tri1a (deltas 2500 20000)
  subset_num=125000
  if [ $subset_num -ge $total_num ]; then
    # use -sfr will be more readable, but macOS doesn't support '-r'.
    ln -sf $wenetspeech_train_set data/${wenetspeech_train_set}_125k
  else
    utils/subset_data_dir.sh \
      data/$wenetspeech_train_set $subset_num data/${wenetspeech_train_set}_125k
  fi

  # 250k utterances for tri1b (deltas 4500 36000)
  subset_num=250000
  if [ $subset_num -ge $total_num ]; then
    ln -sf $wenetspeech_train_set data/${wenetspeech_train_set}_250k
  else
    utils/subset_data_dir.sh \
      data/$wenetspeech_train_set $subset_num data/${wenetspeech_train_set}_250k
  fi

  # 500k utterances for tri2a (lda_mllt 5500 90000)
  subset_num=500000
  if [ $subset_num -ge $total_num ]; then
    ln -sf $wenetspeech_train_set data/${wenetspeech_train_set}_500k
  else
    utils/subset_data_dir.sh \
      data/$wenetspeech_train_set $subset_num data/${wenetspeech_train_set}_500k
  fi

  # 1m utterances for tri3a (sat 7000 110000)
  subset_num=1000000
  if [ $subset_num -ge $total_num ]; then
    ln -sf $wenetspeech_train_set data/${wenetspeech_train_set}_1m
  else
    utils/subset_data_dir.sh \
      data/$wenetspeech_train_set $subset_num data/${wenetspeech_train_set}_1m
  fi

  # 2m utterances for tri3b (sat 12000 190000)
  subset_num=2000000
  if [ $subset_num -ge $total_num ]; then
    ln -sf $wenetspeech_train_set data/${wenetspeech_train_set}_2m
  else
    utils/subset_data_dir.sh \
      data/$wenetspeech_train_set $subset_num data/${wenetspeech_train_set}_2m
  fi


fi


if [ $stage -le 7 ]; then
  # monophone training & decoding (50k utts)
  steps/train_mono.sh --nj $train_nj --cmd "$train_cmd" \
    --boost-silence 1.25 \
    data/${wenetspeech_train_set}_50k data/lang \
    exp/${wenetspeech_train_set}/mono

  {
    utils/mkgraph.sh data/lang_test exp/${wenetspeech_train_set}/mono \
      exp/${wenetspeech_train_set}/mono/graph || exit 1

    for data in $wenetspeech_test_sets; do
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        exp/${wenetspeech_train_set}/mono/graph \
        data/$data exp/${wenetspeech_train_set}/mono/decode_$data || exit 1
    done
  }
fi

if [ $stage -le 8 ]; then
  # tri1a : deltas + delta - delats training & decoding (125k utts)
  steps/align_si.sh --nj $train_nj --cmd "$train_cmd" --boost-silence 1.25 \
    data/${wenetspeech_train_set}_125k data/lang \
    exp/${wenetspeech_train_set}/mono \
    exp/${wenetspeech_train_set}/mono_ali_${wenetspeech_train_set}_125k

  steps/train_deltas.sh --cmd "$train_cmd" --boost-silence 1.25 \
    2500 20000 data/${wenetspeech_train_set}_125k data/lang \
    exp/${wenetspeech_train_set}/mono_ali_${wenetspeech_train_set}_125k \
    exp/${wenetspeech_train_set}/tri1a

  {
    utils/mkgraph.sh data/lang_test exp/${wenetspeech_train_set}/tri1a \
      exp/${wenetspeech_train_set}/tri1a/graph || exit 1

    for data in $wenetspeech_test_sets; do
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        exp/${wenetspeech_train_set}/tri1a/graph \
        data/$data exp/${wenetspeech_train_set}/tri1a/decode_$data || exit 1
    done
  }
fi


if [ $stage -le 9 ]; then
  # tri1b : deltas + delta - delats training & decoding (250k utts)
  steps/align_si.sh --nj $train_nj --cmd "$train_cmd" \
    data/${wenetspeech_train_set}_250k data/lang \
    exp/${wenetspeech_train_set}/tri1a \
    exp/${wenetspeech_train_set}/tri1a_ali_${wenetspeech_train_set}_250k

  steps/train_deltas.sh --cmd "$train_cmd" --boost-silence 1.25 \
    4500 36000 data/${wenetspeech_train_set}_250k data/lang \
    exp/${wenetspeech_train_set}/tri1a_ali_${wenetspeech_train_set}_250k \
    exp/${wenetspeech_train_set}/tri1b

  {
    utils/mkgraph.sh data/lang_test exp/${wenetspeech_train_set}/tri1b \
      exp/${wenetspeech_train_set}/tri1b/graph || exit 1

    for data in $wenetspeech_test_sets; do
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        exp/${wenetspeech_train_set}/tri1b/graph \
        data/$data exp/${wenetspeech_train_set}/tri1b/decode_$data || exit 1
    done
  }
fi


if [ $stage -le 10 ]; then
  # tri2a : lda + mllt training & decoding (500k utts)
  steps/align_si.sh --nj $train_nj --cmd "$train_cmd" \
    data/${wenetspeech_train_set}_500k data/lang \
    exp/${wenetspeech_train_set}/tri1b \
    exp/${wenetspeech_train_set}/tri1b_ali_${wenetspeech_train_set}_500k

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 5000 90000 \
    data/${wenetspeech_train_set}_500k data/lang \
    exp/${wenetspeech_train_set}/tri1b_ali_${wenetspeech_train_set}_500k \
    exp/${wenetspeech_train_set}/tri2a

  {
    utils/mkgraph.sh data/lang_test exp/${wenetspeech_train_set}/tri2a \
      exp/${wenetspeech_train_set}/tri2a/graph || exit 1

    for data in $wenetspeech_test_sets; do
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        exp/${wenetspeech_train_set}/tri2a/graph \
        data/$data exp/${wenetspeech_train_set}/tri2a/decode_$data || exit 1
    done
  }
fi


if [ $stage -le 11 ]; then
  # tri3a : sat training & decoding (1m utts)
  steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd" \
    data/${wenetspeech_train_set}_1m data/lang \
    exp/${wenetspeech_train_set}/tri2a \
    exp/${wenetspeech_train_set}/tri2a_ali_${wenetspeech_train_set}_1m

  steps/train_sat.sh --cmd "$train_cmd" 7000 110000 \
    data/${wenetspeech_train_set}_1m data/lang \
    exp/${wenetspeech_train_set}/tri2a_ali_${wenetspeech_train_set}_1m \
    exp/${wenetspeech_train_set}/tri3a

  {
    utils/mkgraph.sh data/lang_test exp/${wenetspeech_train_set}/tri3a \
      exp/${wenetspeech_train_set}/tri3a/graph || exit 1

    for data in $wenetspeech_test_sets; do
      steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        exp/${wenetspeech_train_set}/tri3a/graph \
        data/$data exp/${wenetspeech_train_set}/tri3a/decode_$data || exit 1
    done
  }
fi

if [ $stage -le 12 ]; then
  # tri3b : sat training & decoding (2m utts)
  steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd" \
    data/${wenetspeech_train_set}_2m data/lang \
    exp/${wenetspeech_train_set}/tri3a \
    exp/${wenetspeech_train_set}/tri3a_ali_${wenetspeech_train_set}_2m

  steps/train_sat.sh --cmd "$train_cmd" 12000 190000 \
    data/${wenetspeech_train_set}_2m data/lang \
    exp/${wenetspeech_train_set}/tri3a_ali_${wenetspeech_train_set}_2m \
    exp/${wenetspeech_train_set}/tri3b

  {
    utils/mkgraph.sh data/lang_test exp/${wenetspeech_train_set}/tri3b \
      exp/${wenetspeech_train_set}/tri3b/graph || exit 1

    for data in $wenetspeech_test_sets; do
      steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        exp/${wenetspeech_train_set}/tri3b/graph \
        data/$data exp/${wenetspeech_train_set}/tri3b/decode_$data || exit 1
    done
  }
fi

if [ $stage -le 13 ]; then
  local/chain/run_cnn_tdnn.sh \
    --stage 0 \
    --train-stage -10 \
    --get-egs-stage -10 \
    --train_set $wenetspeech_train_set \
    --gmm tri3b \
    --test-sets "$wenetspeech_test_sets" \
    --frames-per-iter 3000000 \
    --num-jobs-initial 8 \
    --num-jobs-final 8 \
    --initial-effective-lrate 0.00015 \
    --final-effective-lrate 0.000015 || exit 1;
fi

wait
echo "$0: Done"
exit 0
