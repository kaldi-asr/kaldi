#!/bin/bash

# Copyright 2016  Allen Guo
# Apache 2.0

. ./cmd.sh
. ./path.sh

# paths to corpora (see below for example)
ami=
fisher=
librispeech=
swbd=
tedlium2=
wsj0=
wsj1=
eval2000=
rt03=

# preset paths
case $(hostname -d) in
  clsp.jhu.edu)
    ami=/export/corpora4/ami/amicorpus
    fisher="/export/corpora3/LDC/LDC2004T19 /export/corpora3/LDC/LDC2005T19 \
      /export/corpora3/LDC/LDC2004S13 /export/corpora3/LDC/LDC2005S13"
    librispeech=/export/a15/vpanayotov/data
    swbd=/export/corpora3/LDC/LDC97S62
    tedlium2=/export/corpora5/TEDLIUM_release2
    wsj0=/export/corpora5/LDC/LDC93S6B
    wsj1=/export/corpora5/LDC/LDC94S13B
    eval2000="/export/corpora/LDC/LDC2002S09/hub5e_00 /export/corpora/LDC/LDC2002T43"
    rt03="/export/corpora/LDC/LDC2007S10"
    ;;
esac

# general options
stage=1
multi=multi_a  # This defines the "variant" we're using; see README.md
srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"

. utils/parse_options.sh

# prepare corpora data
if [ $stage -le 1 ]; then
  mkdir -p data/local
  # ami
  local/ami_text_prep.sh data/local/ami/downloads
  local/ami_ihm_data_prep.sh $ami
  local/ami_sdm_data_prep.sh $ami 1
  # fisher
  local/fisher_data_prep.sh $fisher
  utils/fix_data_dir.sh data/fisher/train
  # swbd
  local/swbd1_data_download.sh $swbd
  local/swbd1_data_prep.sh $swbd
  utils/fix_data_dir.sh data/swbd/train
  # librispeech
  local/librispeech_data_prep.sh $librispeech/LibriSpeech/train-clean-100 data/librispeech_100/train
  local/librispeech_data_prep.sh $librispeech/LibriSpeech/train-clean-360 data/librispeech_360/train
  local/librispeech_data_prep.sh $librispeech/LibriSpeech/train-other-500 data/librispeech_500/train
  local/librispeech_data_prep.sh $librispeech/LibriSpeech/test-clean data/librispeech/test
  # tedlium
  local/tedlium_prepare_data.sh $tedlium2
  # wsj
  local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?
  local/wsj_format_data.sh
  utils/copy_data_dir.sh --spk_prefix wsj_ --utt_prefix wsj_ data/wsj/train_si284 data/wsj/train
  rm -rf data/wsj/train_si284
fi

# prepare standalone eval data
if [ $stage -le 2 ]; then
  mkdir -p data/local
  # eval2000
  local/eval2000_data_prep.sh $eval2000
  utils/fix_data_dir.sh data/eval2000/test
  # rt03
  local/rt03_data_prep.sh $rt03
  utils/fix_data_dir.sh data/rt03/test
fi

# prepare initial CMUDict-based dict directory and normalize transcripts
if [ $stage -le 3 ]; then
  local/prepare_dict.sh $tedlium2
  for f in data/*/{train,test}/text; do
    echo Normalizing $f
    cp $f $f.orig
    local/normalize_transcript.py $f.orig > $f
  done
fi

# train G2P model using existing lexicon
# then synthesize pronounciations for all words (that do not include special characters)
# across all training transcripts that are not in the existing lexicon
if [ $stage -le 4 ]; then
  # will skip training if fifth-order model file already exists
  local/g2p/train_g2p.sh data/local/dict_nosp/lexicon2.txt data/local/g2p_model
  local/g2p/apply_g2p.sh data/local/g2p_model/model-5 data/local/g2p_tmp data/local/dict_nosp/lexicon.txt
fi

# prepare (and validate) lang directory
if [ $stage -le 5 ]; then
  rm -f data/local/dict_nosp/lexiconp.txt  # will be created
  utils/prepare_lang.sh data/local/dict_nosp "<unk>" data/local/tmp/lang_nosp data/lang_nosp
fi

# prepare LM and test lang directory
if [ $stage -le 6 ]; then
  mkdir -p data/local/lm
  cat data/{fisher,swbd}/train/text > data/local/lm/text
  local/train_lms.sh  # creates data/local/lm/3gram-mincount/lm_unpruned.gz
  utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
    data/lang_nosp data/local/lm/3gram-mincount/lm_unpruned.gz \
    data/local/dict_nosp/lexicon.txt data/lang_nosp_fsh_sw1_tg
fi

# make training features
if [ $stage -le 7 ]; then
  mfccdir=mfcc
  corpora="ami_ihm ami_sdm1 fisher librispeech_100 librispeech_360 librispeech_500 swbd tedlium wsj"
  for c in $corpora; do
    data=data/$c/train
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf \
      --cmd "$train_cmd" --nj 40 \
      $data exp/make_mfcc/$c/train || exit 1;
    steps/compute_cmvn_stats.sh \
      $data exp/make_mfcc/$c/train || exit 1;
  done
fi

# fix and validate training data directories
if [ $stage -le 8 ]; then
  # create segments file for wsj
  awk '{print $1, $1, 0, -1}' data/wsj/train/utt2spk > data/wsj/train/segments
  for f in `awk '{print $5}' data/wsj/train/wav.scp`; do
    head -c 1024 $f | grep sample_count | awk '{print $3/16000}'
  done > wsj_durations
  paste -d' ' <(cut -d' ' -f1-3 data/wsj/train/segments) wsj_durations > wsj_segments
  mv data/wsj/train/segments{,.bkp}
  mv wsj_segments data/wsj/train/segments
  rm -f wsj_segments wsj_durations
  # create segments files for librispeech
  for c in librispeech_100 librispeech_360 librispeech_500; do
    awk '{print $1, $1, 0, $2}' data/$c/train/utt2dur > data/$c/train/segments;
  done
  # get rid of spk2gender files because not all corpora have them
  rm -f data/*/train/spk2gender
  # create reco2channel_and_file files for wsj and librispeech
  for c in wsj librispeech_100 librispeech_360 librispeech_500; do
    awk '{print $1, $1, "A"}' data/$c/train/wav.scp > data/$c/train/reco2file_and_channel;
  done
  # apply standard fixes, then validate
  for f in data/*/train; do
    utils/fix_data_dir.sh $f
    utils/validate_data_dir.sh $f
  done
fi

# make test features
if [ $stage -le 9 ]; then
  mfccdir=mfcc
  corpora="tedlium eval2000 rt03 librispeech"
  for c in $corpora; do
    data=data/$c/test
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf \
      --cmd "$train_cmd" --nj 20 \
      $data exp/make_mfcc/$c/test || exit 1;
    steps/compute_cmvn_stats.sh \
      $data exp/make_mfcc/$c/test || exit 1;
  done
fi

# fix and validate test data directories
if [ $stage -le 10 ]; then
  for f in data/*/test; do
    utils/fix_data_dir.sh $f
    utils/validate_data_dir.sh $f
  done
fi

# train mono (monophone system)
if [ $stage -le 11 ]; then
  local/make_partitions.sh --multi $multi --stage 1 || exit 1;
  steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
    data/$multi/mono data/lang_nosp exp/$multi/mono || exit 1;
fi

# train tri1 (first triphone pass)
if [ $stage -le 12 ]; then
  local/make_partitions.sh --multi $multi --stage 2 || exit 1;
  steps/align_si.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
    data/$multi/mono_ali data/lang_nosp exp/$multi/mono exp/$multi/mono_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 \
    data/$multi/tri1 data/lang_nosp exp/$multi/mono_ali exp/$multi/tri1 || exit 1;
fi

# train tri2 (LDA+MLLT)
if [ $stage -le 13 ]; then
  local/make_partitions.sh --multi $multi --stage 3 || exit 1;
  steps/align_si.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
    data/$multi/tri1_ali data/lang_nosp exp/$multi/tri1 exp/$multi/tri1_ali || exit 1;
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 3500 30000 \
    data/$multi/tri2 data/lang_nosp exp/$multi/tri1_ali exp/$multi/tri2 || exit 1;
fi

# train tri3 (LDA+MLLT on more data)
if [ $stage -le 14 ]; then
  local/make_partitions.sh --multi $multi --stage 4 || exit 1;
  steps/align_si.sh --cmd "$train_cmd" --nj 30 \
    data/$multi/tri2_ali data/lang_nosp \
    exp/$multi/tri2 exp/$multi/tri2_ali  || exit 1;
  steps/train_lda_mllt.sh --cmd "$train_cmd" 8000 200000 \
    data/$multi/tri3 data/lang_nosp exp/$multi/tri2_ali exp/$multi/tri3 || exit 1;
fi

# reestimate LM with silprobs
if [ $stage -le 15 ]; then
#  steps/get_prons.sh --cmd "$train_cmd" data/$multi/tri3 data/lang_nosp exp/$multi/tri3
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp exp/$multi/tri3/pron_counts_nowb.txt \
    exp/$multi/tri3/sil_counts_nowb.txt exp/$multi/tri3/pron_bigram_counts_nowb.txt data/local/dict
  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
  utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
    data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
    data/local/dict/lexicon.txt data/lang_fsh_sw1_tg
fi

# train tri4 (SAT on almost all data)
if [ $stage -le 16 ]; then
  local/make_partitions.sh --multi $multi --stage 5 || exit 1;
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 60 \
    data/$multi/tri3_ali data/lang \
    exp/$multi/tri3 exp/$multi/tri3_ali || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" 11500 800000 \
    data/$multi/tri4 data/lang exp/$multi/tri3_ali exp/$multi/tri4 || exit 1;
fi

# decode
if [ $stage -le 17 ]; then
  graph_dir=exp/$multi/tri4/graph_tg
  utils/mkgraph.sh data/lang_fsh_sw1_tg \
    exp/$multi/tri4 $graph_dir || exit 1;
  for e in eval2000 rt03 librispeech; do
    steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config $graph_dir \
      data/$e/test exp/$multi/tri4/decode_tg_$e || exit 1;
  done
fi

# train tri5 (SAT on all data but ami_sdm)
if [ $stage -le 18 ]; then
  local/make_partitions.sh --multi $multi --stage 6 || exit 1;
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 100 \
    data/$multi/tri4_ali data/lang \
    exp/$multi/tri4 exp/$multi/tri4_ali || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" 11500 2000000 \
    data/$multi/tri5 data/lang exp/$multi/tri4_ali exp/$multi/tri5 || exit 1;
fi

# decode
if [ $stage -le 19 ]; then
  graph_dir=exp/$multi/tri5/graph_tg
  utils/mkgraph.sh data/lang_fsh_sw1_tg \
    exp/$multi/tri5 $graph_dir || exit 1;
  for e in eval2000 rt03 librispeech; do
    steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config $graph_dir \
      data/$e/test exp/$multi/tri5/decode_tg_$e || exit 1;
  done
fi

# train tdnn
if [ $stage -le 20 ]; then
  local/make_partitions.sh --multi $multi --stage 7 || exit 1;
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 100 \
    data/$multi/tri5_ali data/lang \
    exp/$multi/tri5 exp/$multi/tri5_ali || exit 1;
  local/nnet3/run_tdnn.sh --multi $multi
fi
