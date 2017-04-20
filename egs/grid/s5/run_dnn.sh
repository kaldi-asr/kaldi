#!/bin/bash

# Copyright 2017  Ruhr-University Bochum (Author: Hendrik Meutzner)
#
# Apache 2.0.

# DNN-based audio-visual speech recognition for CHiME-1/2 audio data and GRID video data.

echo ""
echo "$0"
date

. ./cmd.sh  # Needed for local or cluster-based processing
. ./path.sh # Needed for KALDI_ROOT, REC_ROOT and WAV_ROOT

. ./local/check_prerequisites.sh || exit 1

stage=0
# --stage <stage>       # 0: no skipping (default)
#                       # 1: skip data preparation
#                       # 2: skip lang preparation
#                       # 3: skip feature extraction
#                       # 4: skip training
#                       # 5: skip decoding
#                       # 6: skip DNN training and decoding

# Define number of parallel jobs
nj=8

eval_list="devel test" # sets used for decoding

# Define feature type
feat="mfcc"		# MFCC features only
#feat="fbank"	# Filterbank features only
#feat="video"	# Video features only
#feat="av"    # MFCC + video features (using early integration)
#feat="av2"   # Filterbank + video features (using early integration)


# Customize audio data, e.g., setup mixed training.
# You can provide a list of subfolders for each set (train, devel, test).
subfolders_train="isolated"
#subfolders_train="isolated reverberated"

subfolders_test="isolated"
subfolders_devel="isolated"

# do smbr decoding for DNN experiments (yield somewhat better results but takes pretty long)
dnn_do_smbr=true

# needed for GMM experiments
boost_silence=1.0

# ------------------------------------------------------
# --- script start (no more parameters from here on) ---
# ------------------------------------------------------

# Parse external parameters into bash variables
. ./utils/parse_options.sh || exit 1;

# Check for externally provided features
if [ $# -ge 1 ]; then
  feat=$1
fi

# Setup feature file directory
featdir=$REC_ROOT/feat/$feat


# Setup feature directories
mfcc="$REC_ROOT/feat/mfcc"
fbank="$REC_ROOT/feat/fbank"
video="$REC_ROOT/feat/video"
av="$REC_ROOT/feat/av"
av2="$REC_ROOT/feat/av2"

# Setup model and decoding directory
exp="$REC_ROOT/exp"
mkdir -p $exp

# Setup other relevant directories
data="$REC_ROOT/data"
lang="$data/lang"
dict="$data/local/dict"
langtmp="$data/local/lang"
mkdir -p $langtmp
steps="steps"
utils="utils"

# print setup
echo "Features: ${feat}"
echo "Training data: ${subfolders_train}"
echo "Test data: ${eval_list}"

echo ""
echo "Starting in 5 seconds... (CTRL+C to abort)"
sleep 5


# Data preparation
if [ $stage -le 0 ]; then
  echo ""
  echo "Stage ${stage}: Preparing data"

  # create a copy of the utilized audio data using symbolic links
  WAV_ROOT_TMP=$REC_ROOT/wav
  rm -rf $WAV_ROOT_TMP/*

  echo "Creating symbolic links to audio directories (train, test, devel) at $WAV_ROOT_TMP";
  mkdir -p $WAV_ROOT_TMP/train $WAV_ROOT_TMP/devel $WAV_ROOT_TMP/test

  for sf in $subfolders_train; do
    ln -snfv $WAV_ROOT/train/$sf $WAV_ROOT_TMP/train/$sf || exit 1
  done

  for sf in $subfolders_test; do
    ln -snfv $WAV_ROOT/test/$sf $WAV_ROOT_TMP/test/$sf || exit 1
  done

  for sf in $subfolders_devel; do
    ln -snfv $WAV_ROOT/devel/$sf $WAV_ROOT_TMP/devel/$sf || exit 1
  done

  #WAV_ROOT=$WAV_ROOT_TMP

  rm -rf $data/*
  local/chime1_prepare_data.sh --WAV_ROOT $WAV_ROOT_TMP || exit 1
fi


# Language model preparation
if [ $stage -le 1 ]; then
  echo ""
  echo "Stage ${stage}: Preparing language"
  local/chime1_prepare_dict.sh || exit 1
  $utils/prepare_lang.sh --num-sil-states 5 \
    --num-nonsil-states 3 \
    --position-dependent-phones true \
    --share-silence-phones true \
    $dict "A" $langtmp $lang || exit 1
  local/chime1_prepare_grammar.sh || exit 1
fi


# Feature extraction
if [ $stage -le 2 ]; then
  echo ""
  echo "Stage ${stage}: Extracting features"

  # TODO: Check how this can be cleaned
  rm -rf $featdir/*
  mkdir -p $featdir

  rm -rf $REC_ROOT/feat/$feat/*
  mkdir -p $REC_ROOT/feat/$feat

  rm -rf $REC_ROOT/tmp/*
  mkdir -p $REC_ROOT/tmp

  fe_list="train test devel" # sets used for feature extraction
  for x in $fe_list; do

    # extract regular features
    if [ "$feat" = "mfcc" ] || [ "$feat" = "av" ]; then
      mkdir -p $mfcc
      data2=$data/mfcc
      rm -rf $data2/$x/*
      mkdir -p $data2/$x
      cp -R $data/$x/* $data2/$x

      $steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" $data2/$x $exp/make_mfcc/$x $mfcc || exit 1
      # Compute CMVN stats
      $steps/compute_cmvn_stats.sh $data2/$x $exp/make_mfcc/$x $mfcc || exit 1
    fi

    if [ "$feat" = "fbank" ] || [ "$feat" = "av2" ]; then
      mkdir -p $fbank
      data2=$data/fbank
      rm -rf $data2/$x/*
      mkdir -p $data2/$x
      cp -R $data/$x/* $data2/$x

      $steps/make_fbank.sh --nj $nj --cmd "$train_cmd" --fbank_config conf/fbank.conf $data2/$x $exp/make_fbank/$x $fbank || exit 1
      # Compute CMVN stats
      $steps/compute_cmvn_stats.sh $data2/$x $exp/make_fbank/$x $fbank || exit 1
    fi

    if [ "$feat" = "video" ] || [ "$feat" = "av" ] || [ "$feat" = "av2" ]; then
      mkdir -p $video
      data2=$data/video
      rm -rf $data2/$x/*
      mkdir -p $data2/$x
      cp -R $data/$x/* $data2/$x

      echo "Running make_video.sh"
      local/make_video.sh --nj $nj \
                          --cmd "$train_cmd" \
                          --audioRoot $REC_ROOT/wav \
                          --videoRoot $VIDEO_ROOT \
                          $data2/$x \
                          $exp/make_video/$x \
                          $video || exit 1

      # Compute CMVN stats
      $steps/compute_cmvn_stats.sh $data2/$x $exp/make_video/$x $video || exit 1
    fi


    if [ "$feat" = "av" ] || [ "$feat" = "av2" ]; then

      data2=$data/$feat
      mkdir -p $data2

      if [ "$feat" = "av" ]; then
        # Append audio/video features
        $steps/append_feats.sh --nj $nj --cmd "$train_cmd" \
          $data/mfcc/$x $data/video/$x \
          $data2/$x $exp/make_${feat}/$x $featdir || exit 1
      elif [ "$feat" = "av2" ]; then
        # Append audio/video features
        $steps/append_feats.sh --nj $nj --cmd "$train_cmd" \
          $data/fbank/$x $data/video/$x \
          $data2/$x $exp/make_${feat}/$x $featdir || exit 1
      fi
      $steps/compute_cmvn_stats.sh $data2/$x $exp/make_${feat}/$x $featdir || exit 1
    fi

  done # for x in $fe_list
fi # stage 2

data=$data/$feat
exp=$exp/$feat
mkdir -p $exp

# GMM training
if [ $stage -le 3 ]; then
  echo ""
  echo "Stage ${stage}: Starting GMM training"
  #rm -rf $exp/* $data/train/split*
  rm $data/train/split*

  $steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
    --boost_silence $boost_silence \
    $data/train $lang $exp/mono0a || exit 1;

  #$utils/mkgraph.sh $lang $exp/mono0a $exp/mono0a/graph || exit 1;

  $steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    --boost_silence $boost_silence \
    $data/train $lang $exp/mono0a $exp/mono0a_ali || exit 1;

  $steps/train_deltas.sh --cmd "$train_cmd" \
    --boost_silence $boost_silence \
    2000 10000 $data/train $lang $exp/mono0a_ali $exp/tri1 || exit 1;

  #$utils/mkgraph.sh $lang $exp/tri1 $exp/tri1/graph || exit 1;

  $steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    $data/train $lang $exp/tri1 $exp/tri1_ali || exit 1;

  $steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    2500 15000 $data/train $lang $exp/tri1_ali $exp/tri2b || exit 1;

  $utils/mkgraph.sh $lang $exp/tri2b $exp/tri2b/graph || exit 1;

  $steps/align_si.sh --cmd "$train_cmd" --nj $nj \
    --use-graphs true $data/train $lang $exp/tri2b $exp/tri2b_ali || exit 1;

  $steps/train_sat.sh \
    2500 15000 $data/train $lang $exp/tri2b_ali $exp/tri3b || exit 1;

  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $data/train $lang $exp/tri3b $exp/tri3b_ali

  #$utils/mkgraph.sh $lang $exp/tri3b $exp/tri3b/graph || exit 1;
fi


# GMM decoding
if [ $stage -le 4 ]; then

  #model_list="mono0a tri1 tri2b tri3b"
  model_list="tri3b"

  echo ""
  echo "Stage ${stage}: Starting GMM decoding"
  for mdl in $model_list; do

    for x in $eval_list; do

      echo ""
      echo "Evaluating model $mdl for $x"

      if [ ! -f $exp/$mdl/graph/HCLG.fst ]; then
        $utils/mkgraph.sh $lang $exp/$mdl $exp/$mdl/graph || exit 1;
      fi

      $steps/decode_fmllr.sh --cmd "$decode_cmd" --nj $nj --num-threads 4 --skip_scoring true \
          $exp/$mdl/graph $data/$x $exp/$mdl/decode_$x  #&

      wait

      echo ""
      echo "---- $x set"
      local/score.sh $data/$x $exp/$mdl/graph $exp/$mdl/decode_$x
    done

  done #mdl
fi #stage


# DNN training
if [ $stage -le 5 ]; then
  echo "Stage ${stage}: Starting DNN training and decoding"

  local/nnet/run_dnn_fmllr.sh \
    --eval_list "$eval_list" \
    --do_smbr $dnn_do_smbr \
    --srcgmm tri3b \
    --dstdnn dnn_tri3b_fmllr \
    $REC_ROOT $feat

fi

echo ""
echo "----------- SCORE SUMMARY -----------"
local/collect_scores.sh $REC_ROOT/exp

echo ""
echo "All done."
date
exit 0
