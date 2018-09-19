#!/bin/bash

# Copyright 2017 John Morgan
# Apache 2.0.

. ./cmd.sh
. ./path.sh
stage=0
datadir=$1

. ./utils/parse_options.sh

set -e
set -o pipefail

tmpdir=data/local/tmp

# acoustic models are trained on the heroico corpus
# testing is done on the usma corpus
# heroico consists of 2 parts: answers and recordings (recited)

answers_transcripts=$datadir/data/transcripts/heroico-answers.txt
recordings_transcripts=$datadir/data/transcripts/heroico-recordings.txt

# usma is all recited
usma_transcripts=$datadir/data/transcripts/usma-prompts.txt

# make acoustic model training  lists
if [ $stage -le 0 ]; then
  mkdir -p $tmpdir/heroico $tmpdir/usma

  local/get_wav_list.sh $datadir/data

  # make separate lists for heroico answers and recordings
  # the transcripts are converted to UTF8
  export LC_ALL=en_US.UTF-8
  cat $answers_transcripts  | iconv -f ISO-8859-1 -t UTF-8 | \
    tr -d '\r' |  local/heroico_answers_make_lists.pl

  utils/fix_data_dir.sh $tmpdir/heroico/answers

  cat $recordings_transcripts | iconv -f ISO-8859-1 -t UTF-8 | \
    tr -d '\r' | local/heroico_recordings_make_lists.pl

  utils/fix_data_dir.sh $tmpdir/heroico/recordings/train
  utils/fix_data_dir.sh $tmpdir/heroico/recordings/devtest

  # consolidate heroico lists
  mkdir -p $tmpdir/heroico/lists/train $tmpdir/heroico/lists/devtest

  for x in wav.scp utt2spk text; do
    cat $tmpdir/heroico/answers/$x $tmpdir/heroico/recordings/train/$x | \
      tr -d '\r' | sort -k1,1 -u >$tmpdir/heroico/lists/train/$x
  done

  for x in wav.scp utt2spk text; do
    cat $tmpdir/heroico/recordings/devtest/$x | tr -d '\r' | \
      sort -k1,1 -u >$tmpdir/heroico/lists/devtest/$x
  done

  utils/fix_data_dir.sh $tmpdir/heroico/lists/train
  utils/fix_data_dir.sh $tmpdir/heroico/lists/devtest
fi

if [ $stage -le 1 ]; then
  #  make separate lists for usma (US military academy) native and nonnative
  cat $usma_transcripts | iconv -f ISO-8859-1 -t UTF-8 | \
    tr -d '\r' | dos2unix | local/usma_native_make_lists.pl

  cat $usma_transcripts | iconv -f ISO-8859-1 -t UTF-8 | \
    tr -d '\r' | local/usma_nonnative_make_lists.pl

  for n in native nonnative; do
    mkdir -p $tmpdir/usma/$n/lists
    for x in wav.scp utt2spk text; do
      sort $tmpdir/usma/$n/$x >$tmpdir/usma/$n/lists/$x
    done

    utils/fix_data_dir.sh  $tmpdir/usma/$n/lists
  done

  mkdir -p data/train $tmpdir/lists/train data/devtest $tmpdir/lists/devtest

  # get training lists
  for x in wav.scp utt2spk text; do
    cat $tmpdir/heroico/answers/${x} $tmpdir/heroico/recordings/train/${x} | \
      tr -d '\r' >$tmpdir/lists/train/$x
    sort $tmpdir/lists/train/$x >data/train/$x
  done

  # get devtest lists
  for x in wav.scp utt2spk text; do
    cat $tmpdir/heroico/lists/devtest/$x | \
       tr -d '\r' >$tmpdir/lists/devtest/$x
    sort $tmpdir/lists/devtest/$x >data/devtest/$x
  done

  utils/utt2spk_to_spk2utt.pl data/train/utt2spk | sort >data/train/spk2utt
  utils/utt2spk_to_spk2utt.pl data/devtest/utt2spk | sort >data/devtest/spk2utt

  utils/fix_data_dir.sh data/train
  utils/fix_data_dir.sh data/devtest

  # make testing  lists
  mkdir -p data/test data/native data/nonnative $tmpdir/usma/lists

  for x in wav.scp text utt2spk; do
    for n in native nonnative; do
      cat $tmpdir/usma/$n/lists/$x
    done >$tmpdir/usma/lists/$x

    cat $tmpdir/usma/lists/$x >data/test/$x

    for n in native nonnative; do
      sort $tmpdir/usma/$n/$x >data/$n/$x
    done
  done

  for n in native nonnative test; do
    utils/utt2spk_to_spk2utt.pl data/$n/utt2spk | sort >data/$n/spk2utt
    utils/fix_data_dir.sh data/$n
  done
fi
