#!/usr/bin/env bash

CORPUS=/pool/corpora/spgispeech/

stage=1

. ./cmd.sh
. ./path.sh
. parse_options.sh

set -e -o pipefail

if [ $stage -le 1 ] ; then
  lhotse prepare spgispeech $CORPUS ./
fi

if [ $stage -le 2 ] ; then
  lhotse kaldi export spgispeech_recordings_val.jsonl.gz spgispeech_supervisions_val.jsonl.gz  data/val
  utils/fix_data_dir.sh data/val
  utils/validate_data_dir.sh --no-feats data/val
  lhotse kaldi export spgispeech_recordings_train.jsonl.gz spgispeech_supervisions_train.jsonl.gz  data/train
  utils/fix_data_dir.sh data/train
  utils/validate_data_dir.sh --no-feats data/train
fi

mfccdir=./mfcc
if [ $stage -le 3 ] ; then
  for part in val train ; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/$part exp/make_mfcc/$part $mfccdir
    utils/fix_data_dir.sh data/$part
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
    utils/fix_data_dir.sh data/$part
    utils/validate_data_dir.sh data/$part
  done
fi

mkdir -p data/local/dict
if [ $stage -le 4 ] ; then
  cut -d ' ' -f 2- data/train/text  | sed 's/ /\n/g' | sort -u | grep -v  '[0-9]' > data/local/wordlist.txt
  wget http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7a -o $cmudict_dir/
  local/g2p/train_g2p.sh --cmd "$train_cmd" data/local/ data/local/g2p
  local/prepare_dict.sh  --cmd "$train_cmd" data/local/wordlist.txt data/local/g2p data/local/dict

fi

if [ $stage -le 5 ] ; then
  utils/prepare_lang.sh data/local/dict \
    "<UNK>" data/local/lang_tmp data/lang
fi

if [ $stage -le 6 ] ; then
  local/train_lms_srilm.sh --train-text data/train/text  data data/srilm
  utils/format_lm.sh data/lang data/srilm/3gram.me.gz  data/local/dict/lexicon.txt data/lang_test
  utils/build_const_arpa_lm.sh data/srilm/4gram.me.gz data/lang_test data/lang_test
fi

if [ $stage -le 7 ]; then
  # Make some small data subsets for early system-build stages.  Note, there are 29k
  # utterances in the train_clean_100 directory which has 100 hours of data.
  # For the monophone stages we select the shortest utterances, which should make it
  # easier to align the data from a flat start.

  utils/subset_data_dir.sh --shortest data/train 2000 data/train_2kshort
  utils/subset_data_dir.sh data/train 5000 data/train_5k
  utils/subset_data_dir.sh data/train 10000 data/train_10k
fi

if [ $stage -le 8 ]; then
  # train a monophone system
  steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
    data/train_2kshort data/lang exp/mono
fi

if [ $stage -le 9 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
    data/train_5k data/lang exp/mono exp/mono_ali_5k

  # train a first delta + delta-delta triphone system on a subset of 5000 utterances
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train_5k data/lang exp/mono_ali_5k exp/tri1
fi

if [ $stage -le 10 ]; then
  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
    data/train_10k data/lang exp/tri1 exp/tri1_ali_10k


  # train an LDA+MLLT system.
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/train_10k data/lang exp/tri1_ali_10k exp/tri2b
fi

if [ $stage -le 11 ]; then
  # Align a 10k utts subset using the tri2b model
  steps/align_si.sh  --nj 10 --cmd "$train_cmd" --use-graphs true \
    data/train_10k data/lang exp/tri2b exp/tri2b_ali_10k

  # Train tri3b, which is LDA+MLLT+SAT on 10k utts
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
    data/train_10k data/lang exp/tri2b_ali_10k exp/tri3b

fi

if [ $stage -le 12 ]; then
  # align the entire train_clean_100 subset using the tri3b model
  steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
    data/train data/lang \
    exp/tri3b exp/tri3b_ali

  # train another LDA+MLLT+SAT system on the entire 100 hour subset
  steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
    data/train data/lang \
    exp/tri3b_ali exp/tri4b
fi

if [ $stage -le 13 ]; then
  # align the new, combined set, using the tri4b model
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
    data/train data/lang exp/tri4b exp/tri4b_ali

  # create a larger SAT model, trained on the 460 hours of data.
  steps/train_sat.sh  --cmd "$train_cmd" 5000 100000 \
    data/train data/lang exp/tri4b_ali exp/tri5b
fi

## you can continue by
#./local/run_cleanup_segmentation.sh
#./local/chain/run_tdnn.sh
#./local/chain/run_tdnn_lstm.sh --stage 14
#
