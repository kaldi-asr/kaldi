#!/usr/bin/env bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
train_nj=32
stage=0
# End configuration section
. ./utils/parse_options.sh

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

audio_data=/export/corpora/LDC/LDC98S74
transcript_data=/export/corpora/LDC/LDC98T29
eval_data=/export/corpora/LDC/LDC2001S91

boost_sil=0.5
numLeavesTri1=1000
numGaussTri1=10000
numLeavesTri2=1000
numGaussTri2=20000
numLeavesTri3=6000
numGaussTri3=75000
numLeavesMLLT=6000
numGaussMLLT=75000
numLeavesSAT=6000
numGaussSAT=75000
unk="<unk>"

. ./cmd.sh
. ./path.sh

if [ $stage -le 0 ]; then
  # Eval dataset preparation

  # prepare_data.sh does not really care about the order or number of the
  # corpus directories
  local/prepare_data.sh \
    $eval_data/HUB4_1997NE/doc/h4ne97sp.sgm \
    $eval_data/HUB4_1997NE/h4ne_sp/h4ne97sp.sph data/eval
  local/prepare_test_text.pl \
    "$unk" data/eval/text > data/eval/text.clean
  mv data/eval/text data/eval/text.old
  mv data/eval/text.clean data/eval/text
  utils/fix_data_dir.sh data/eval
fi


if [ $stage -le 1 ]; then
  ## Training dataset preparation
  local/prepare_data.sh $audio_data $transcript_data data/train
  local/prepare_training_text.pl \
    "$unk" data/train/text > data/train/text.clean
  mv data/train/text data/train/text.old
  mv data/train/text.clean data/train/text
  utils/fix_data_dir.sh data/train
fi

if [ $stage -le 2 ]; then
  # Graphemic lexicon
  mkdir -p data/local
  local/prepare_lexicon.sh data/train/text data/local
fi

if [ $stage -le 3 ]; then
  # Language model
  local/train_lms_srilm.sh  --oov-symbol "$unk"\
      --train-text data/train/text data data/srilm
  cp -R data/lang data/lang_test
  utils/format_lm.sh \
    data/lang data/srilm/lm.gz  data/local/lexicon.txt data/lang_test
fi

if [ $stage -le 4 ]; then
  # Training set features
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $train_nj data/train exp/make_mfcc_pitch/train mfcc
  utils/fix_data_dir.sh data/train
  steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train mfcc
  utils/fix_data_dir.sh data/train
fi

if [ $stage -le 4 ]; then
  # Eval dataset features
  steps/make_mfcc.sh --cmd "$decode_cmd" --nj 16  data/eval exp/make_mfcc_pitch/eval mfcc
  utils/fix_data_dir.sh data/eval
  steps/compute_cmvn_stats.sh data/eval exp/make_mfcc/eval mfcc
  utils/fix_data_dir.sh data/eval
fi


if [ $stage -le 5 ]; then
  # Subset the training data to speed up the early stages of training
  numutt=`cat data/train/feats.scp | wc -l`;
  utils/subset_data_dir.sh data/train  5000 data/train_sub1
  if [ $numutt -gt 10000 ] ; then
    utils/subset_data_dir.sh data/train 10000 data/train_sub2
  else
    (cd data; ln -s train train_sub2 )
  fi
  if [ $numutt -gt 20000 ] ; then
    utils/subset_data_dir.sh data/train 20000 data/train_sub3
  else
    (cd data; ln -s train train_sub3 )
  fi

fi

mkdir -p exp
if [ $stage -le 6 ]; then
  echo ---------------------------------------------------------------------
  echo "Stage 6: Starting (small) monophone training in exp/mono on" `date`
  echo ---------------------------------------------------------------------
  steps/train_mono.sh \
    --boost-silence $boost_sil --nj 8 --cmd "$train_cmd" \
    data/train_sub1 data/lang exp/mono
fi

if [ $stage -le 6 ]; then
  echo ---------------------------------------------------------------------
  echo "Stage 6: Starting (small) triphone training in exp/tri1 on" `date`
  echo ---------------------------------------------------------------------
  steps/align_si.sh \
    --boost-silence $boost_sil --nj 12 --cmd "$train_cmd" \
    data/train_sub2 data/lang exp/mono exp/mono_ali_sub2

  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri1 $numGaussTri1 \
    data/train_sub2 data/lang exp/mono_ali_sub2 exp/tri1
fi

if [ $stage -le 7 ]; then
  echo ---------------------------------------------------------------------
  echo "Stage 7: Starting (medium) triphone training in exp/tri2 on" `date`
  echo ---------------------------------------------------------------------
  steps/align_si.sh \
    --boost-silence $boost_sil --nj 24 --cmd "$train_cmd" \
    data/train_sub3 data/lang exp/tri1 exp/tri1_ali_sub3

  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri2 $numGaussTri2 \
    data/train_sub3 data/lang exp/tri1_ali_sub3 exp/tri2

  local/reestimate_langp.sh --cmd "$train_cmd" --unk "$unk" \
    data/train_sub3 data/lang data/local/ \
    exp/tri2 data/local/dictp/tri2 data/local/langp/tri2 data/langp/tri2
fi

if [ $stage -le 8 ]; then
  echo ---------------------------------------------------------------------
  echo "Stage 8: Starting (full) triphone training in exp/tri3 on" `date`
  echo ---------------------------------------------------------------------
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/langp/tri2 exp/tri2 exp/tri2_ali

  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesTri3 $numGaussTri3 data/train data/langp/tri2 exp/tri2_ali exp/tri3

  local/reestimate_langp.sh --cmd "$train_cmd" --unk "$unk" \
    data/train data/lang data/local/ \
    exp/tri3 data/local/dictp/tri3 data/local/langp/tri3 data/langp/tri3
fi

if [ $stage -le 9 ]; then
  echo ---------------------------------------------------------------------
  echo "Stage 9: Starting (lda_mllt) triphone training in exp/tri4 on" `date`
  echo ---------------------------------------------------------------------
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/langp/tri3 exp/tri3 exp/tri3_ali

  steps/train_lda_mllt.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesMLLT $numGaussMLLT data/train data/langp/tri3 exp/tri3_ali exp/tri4

  local/reestimate_langp.sh --cmd "$train_cmd" --unk "$unk" \
    data/train data/lang data/local \
    exp/tri4 data/local/dictp/tri4 data/local/langp/tri4 data/langp/tri4
fi

if [ $stage -le 10 ]; then
  echo ---------------------------------------------------------------------
  echo "Stage 10: Starting (SAT) triphone training in exp/tri5 on" `date`
  echo ---------------------------------------------------------------------

  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/langp/tri4 exp/tri4 exp/tri4_ali

  steps/train_sat.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesSAT $numGaussSAT data/train data/langp/tri4 exp/tri4_ali exp/tri5

  local/reestimate_langp.sh --cmd "$train_cmd" --unk "$unk" \
    data/train data/lang data/local \
    exp/tri5 data/local/dictp/tri5 data/local/langp/tri5 data/langp/tri5
fi


if [ $stage -le 11 ]; then
  echo ---------------------------------------------------------------------
  echo "Stage 11: Starting exp/tri5_ali on" `date`
  echo ---------------------------------------------------------------------
  steps/align_fmllr.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/langp/tri5 exp/tri5 exp/tri5_ali

  local/reestimate_langp.sh --cmd "$train_cmd" --unk "$unk" \
    data/train data/lang data/local \
    exp/tri5_ali data/local/dictp/tri5_ali data/local/langp/tri5_ali data/langp/tri5_ali
fi

if [ $stage -le 12 ]; then
  echo ---------------------------------------------------------------------
  echo "Stage 12: Building lang dir" `date`
  echo ---------------------------------------------------------------------
  cp -R data/langp/tri5_ali/ data/langp_test
  cp data/lang_test/G.fst data/langp_test
fi

if [ $stage -le 13 ]; then
  echo ---------------------------------------------------------------------
  echo "Stage 13: Running decoding with SAT models  on" `date`
  echo ---------------------------------------------------------------------
  decode=exp/tri5/decode_test
  utils/mkgraph.sh \
    data/langp_test exp/tri5 exp/tri5/graph |tee exp/tri5/mkgraph.log

  mkdir -p $decode
  steps/decode_fmllr_extra.sh  --beam 10 --lattice-beam 4\
    --nj 32 --cmd "$decode_cmd"\
    exp/tri5/graph data/eval/ ${decode} |tee ${decode}/decode.log
  touch ${decode}/.done
fi


#./local/chain/run_tdnn.sh
#./local/run_sgmm2.sh
