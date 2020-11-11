#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh

stage=0

# the location of the LDC corpus; this location works for the CLSP grid.
datadir=/export/corpora5/LDC/LDC2006S37

# The corpus and lexicon are on openslr.org
#speech_url="http://www.openslr.org/resources/39/LDC2006S37.tar.gz"
lexicon_url="http://www.openslr.org/resources/34/santiago.tar.gz"

# Location of the Movie subtitles text corpus
subtitles_url="http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2018/en-es.txt.zip"

. utils/parse_options.sh

set -e
set -o pipefail
set -u


# don't change tmpdir, the location is used explicitly in scripts in local/.
tmpdir=data/local/tmp

if [ $stage -le 0 ]; then
  if [ ! -d $datadir ]; then
    echo "$0: please download and un-tar http://www.openslr.org/resources/39/LDC2006S37.tar.gz"
    echo "  and set $datadir to the directory where it is located."
    exit 1
  fi
  if [ ! -s santiago.txt ]; then
    echo "$0: downloading the lexicon"
    wget -c http://www.openslr.org/resources/34/santiago.tar.gz
    tar -xvzf santiago.tar.gz
  fi
  # Get data for lm training
  local/subs_download.sh $subtitles_url
fi

if [ $stage -le 1 ]; then
  echo "Making lists for building models."
  local/prepare_data.sh $datadir
fi

if [ $stage -le 2 ]; then
  mkdir -p data/local/dict $tmpdir/dict
  local/prepare_dict.sh
fi

if [ $stage -le 3 ]; then
  utils/prepare_lang.sh \
    data/local/dict "<UNK>" \
    data/local/lang data/lang
fi

if [ $stage -le 4 ]; then
  mkdir -p $tmpdir/subs/lm
  local/subs_prepare_data.pl
fi

if [ $stage -le 5 ]; then
  echo "point 1"
  local/prepare_lm.sh  $tmpdir/subs/lm/in_vocabulary.txt
fi

if [ $stage -le 6 ]; then
  echo "point 2"
  utils/format_lm.sh \
    data/lang data/local/lm/trigram.arpa.gz data/local/dict/lexicon.txt \
    data/lang_test
fi

if [ $stage -le 7 ]; then
  echo "$0: extracting acoustic features."
  mkdir -p exp

  for fld in native nonnative test devtest train; do
    if [ -e data/$fld/cmvn.scp ]; then
      rm data/$fld/cmvn.scp
    fi

    steps/make_mfcc.sh --cmd "$train_cmd" --nj 4 data/$fld
    utils/fix_data_dir.sh data/$fld
    steps/compute_cmvn_stats.sh data/$fld
    utils/fix_data_dir.sh data/$fld
  done
fi

if [ $stage -le 8 ]; then
  echo "$0 monophone training"
  steps/train_mono.sh --nj 8 --cmd "$train_cmd" data/train data/lang exp/mono || exit 1;

  # evaluation
  (
    # make decoding graph for monophones
    utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1;

    # test monophones
    for x in native nonnative devtest test; do
      steps/decode.sh --nj 8 exp/mono/graph data/$x exp/mono/decode_${x} || exit 1;
    done
  ) &
fi

if [ $stage -le 9 ]; then
  # align with monophones
  steps/align_si.sh --nj 8 --cmd "$train_cmd" \
    data/train data/lang exp/mono exp/mono_ali

  echo "$0 Starting  triphone training in exp/tri1"
  steps/train_deltas.sh  --cmd "$train_cmd" --cluster-thresh 100 \
    1500 25000 data/train data/lang exp/mono_ali exp/tri1

  wait  # wait for the previous decoding jobs to finish in case there's just one
        # machine.
  (
    utils/mkgraph.sh \
    data/lang_test exp/tri1 exp/tri1/graph || exit 1;

    for x in native nonnative devtest test; do
      steps/decode.sh --nj 8 exp/tri1/graph data/$x exp/tri1/decode_${x} || exit 1;
    done
  ) &

fi

if [ $stage -le 10 ]; then
  echo "$0: Starting delta system alignment"
  steps/align_si.sh \
    --nj 8 --cmd "$train_cmd" data/train data/lang exp/tri1 exp/tri1_ali

  echo "$0: starting lda+mllt triphone training in exp/tri2b"

  steps/train_lda_mllt.sh \
    --splice-opts "--left-context=3 --right-context=3" \
    2000 30000 data/train data/lang exp/tri1_ali exp/tri2b

  wait  # wait for the previous decoding jobs to finish in case there's just one
        # machine.

  (
    utils/mkgraph.sh data/lang_test exp/tri2b exp/tri2b/graph || exit 1;

    for x in native nonnative devtest test; do
      steps/decode.sh --nj 8 exp/tri2b/graph data/$x exp/tri2b/decode_${x} || exit 1;
    done
  ) &
fi

if  [ $stage -le 11 ]; then
  echo "$0: Starting LDA+MLLT system alignment"

  steps/align_si.sh \
    --use-graphs true --nj 8 --cmd "$train_cmd" \
    data/train data/lang exp/tri2b exp/tri2b_ali

  echo "$0 Starting (SAT) triphone training in exp/tri3b"
  steps/train_sat.sh \
    --cmd "$train_cmd" \
    3100 50000 data/train data/lang exp/tri2b_ali exp/tri3b

  echo "$0 Starting exp/tri3b_ali"
  steps/align_fmllr.sh \
    --nj 8 --cmd "$train_cmd" \
    data/train data/lang exp/tri3b exp/tri3b_ali

  wait  # wait for the previous decoding jobs to finish in case there's just one
        # machine.
  (
    # make decoding graphs for SAT models
    utils/mkgraph.sh \
      data/lang_test exp/tri3b exp/tri3b/graph ||  exit 1;

    for x in native nonnative devtest test; do
      echo "$0: decoding $x with tri3b models."
      steps/decode_fmllr.sh \
        --nj 8 --cmd "$decode_cmd"  exp/tri3b/graph data/$x exp/tri3b/decode_${x}
    done
  ) &
fi

if [ $stage -le 12 ]; then
  echo "$0: train and test chain models."
  local/chain/run_tdnn.sh
fi

wait
