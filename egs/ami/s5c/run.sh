#!/usr/bin/env bash
# Copyright   2020   Johns Hopkins University (Author: Desh Raj)
# Apache 2.0.
#
# This recipe performs diarization for the mix-headset data in the
# AMI dataset. The x-vector extractor we use is trained on VoxCeleb v2 
# corpus with simulated RIRs. We use oracle SAD in this recipe.
# This recipe demonstrates the following:
# 1. Diarization using x-vector and clustering (AHC, VBx, spectral)
# 2. Training an overlap detector (using annotations) and corresponding
# inference on full recordings.

# We do not provide training script for an x-vector extractor. You
# can download a pretrained extractor from:
# http://kaldi-asr.org/models/12/0012_diarization_v1.tar.gz
# and extract it.

. ./cmd.sh
. ./path.sh
set -euo pipefail
mfccdir=`pwd`/mfcc

stage=0
overlap_stage=0
diarizer_stage=0
nj=50
decode_nj=15

model_dir=exp/xvector_nnet_1a

train_set=train
test_sets="dev test"

diarizer_type=spectral  # must be one of (ahc, spectral, vbx)

. utils/parse_options.sh

# Path where AMI gets downloaded (or where locally available):
AMI_DIR=$PWD/wav_db # Default,
case $(hostname -d) in
  fit.vutbr.cz) AMI_DIR=/mnt/matylda5/iveselyk/KALDI_AMI_WAV ;; # BUT,
  clsp.jhu.edu) AMI_DIR=/export/corpora5/amicorpus ;; # JHU,
  cstr.ed.ac.uk) AMI_DIR= ;; # Edinburgh,
esac

# Download AMI corpus, You need around 130GB of free space to get whole data
if [ $stage -le 1 ]; then
  if [ -d $AMI_DIR ] && ! touch $AMI_DIR/.foo 2>/dev/null; then
    echo "$0: directory $AMI_DIR seems to exist and not be owned by you."
    echo " ... Assuming the data does not need to be downloaded.  Please use --stage 2 or more."
    exit 1
  fi
  if [ -e data/local/downloads/wget_$mic.sh ]; then
    echo "data/local/downloads/wget_$mic.sh already exists, better quit than re-download... (use --stage N)"
    exit 1
  fi
  local/ami_download.sh $mic $AMI_DIR
fi

# Prepare data directories. 
if [ $stage -le 2 ]; then
  # Download the data split and references from BUT's AMI setup
  if ! [ -d AMI-diarization-setup ]; then
    git clone https://github.com/BUTSpeechFIT/AMI-diarization-setup
  fi

  for dataset in train $test_sets; do
    echo "$0: preparing $dataset set.."
    mkdir -p data/$dataset
    # Prepare wav.scp and segments file from meeting lists and oracle SAD
    # labels, and concatenate all reference RTTMs into one file.
    local/prepare_data.py --sad-labels-dir AMI-diarization-setup/only_words/labs/${dataset} \
      AMI-diarization-setup/lists/${dataset}.meetings.txt \
      $AMI_DIR data/$dataset
    cat AMI-diarization-setup/only_words/rttms/${dataset}/*.rttm \
      > data/${dataset}/rttm.annotation

    awk '{print $1,$2}' data/$dataset/segments > data/$dataset/utt2spk
    utils/utt2spk_to_spk2utt.pl data/$dataset/utt2spk > data/$dataset/spk2utt
    utils/fix_data_dir.sh data/$dataset
  done
fi

# Feature extraction
if [ $stage -le 3 ]; then
  for dataset in train $test_sets; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $nj --cmd "$train_cmd" data/$dataset
    steps/compute_cmvn_stats.sh data/$dataset
    utils/fix_data_dir.sh data/$dataset
  done
fi

if [ $stage -le 4 ]; then
  echo "$0: preparing a AMI training data to train PLDA model"
  local/nnet3/xvector/prepare_feats.sh --nj $nj --cmd "$train_cmd" \
    data/train data/plda_train exp/plda_train_cmn
fi

if [ $stage -le 5 ]; then
  echo "$0: extracting x-vector for PLDA training data"
  utils/fix_data_dir.sh data/plda_train
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 10G" \
    --nj $nj --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false \
    --hard-min true $model_dir \
    data/plda_train $model_dir/xvectors_plda_train
fi

# Train PLDA models
if [ $stage -le 6 ]; then
  echo "$0: training PLDA model"
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $model_dir/xvectors_plda_train/log/compute_mean.log \
    ivector-mean scp:$model_dir/xvectors_plda_train/xvector.scp \
    $model_dir/xvectors_plda_train/mean.vec || exit 1;

  # Train the PLDA model.
  $train_cmd $model_dir/xvectors_plda_train/log/plda.log \
    ivector-compute-plda ark:$model_dir/xvectors_plda_train/spk2utt \
    "ark:ivector-subtract-global-mean scp:$model_dir/xvectors_plda_train/xvector.scp ark:- |\
     transform-vec $model_dir/xvectors_plda_train/transform.mat ark:- ark:- |\
      ivector-normalize-length ark:-  ark:- |" \
    $model_dir/xvectors_plda_train/plda || exit 1;
  
  cp $model_dir/xvectors_plda_train/plda $model_dir/
  cp $model_dir/xvectors_plda_train/transform.mat $model_dir/
  cp $model_dir/xvectors_plda_train/mean.vec $model_dir/
fi

if [ $stage -le 7 ]; then
  for datadir in ${test_sets}; do
    ref_rttm=data/${datadir}/rttm.annotation

    diarize_nj=$(wc -l < "data/$datadir/wav.scp")
    nj=$((decode_nj>diarize_nj ? diarize_nj : decode_nj))
    local/diarize_${diarizer_type}.sh --nj $nj --cmd "$train_cmd" --stage $diarizer_stage \
      $model_dir data/${datadir} exp/${datadir}_diarization_${diarizer_type}

    # Evaluate RTTM using md-eval.pl
    rttm_affix=
    if [ $diarizer_type == "vbx" ]; then
      rttm_affix=".vb"
    fi
    md-eval.pl -r $ref_rttm -s exp/${datadir}_diarization_${diarizer_type}/rttm${rttm_affix}
  done
fi

# These stages demonstrate how to perform training and inference
# for an overlap detector.
if [ $stage -le 8 ]; then
  echo "$0: training overlap detector"
  local/train_overlap_detector.sh --stage $overlap_stage --test-sets "$test_sets" $AMI_DIR
fi

overlap_affix=1a
if [ $stage -le 9 ]; then
  for dataset in $test_sets; do
    echo "$0: performing overlap detection on $dataset"
    local/detect_overlaps.sh --convert_data_dir_to_whole true \
      --output-scale "1 2 1" data/${dataset} \
      exp/overlap_$overlap_affix/tdnn_lstm_1a exp/overlap_$overlap_affix/$dataset

    echo "$0: evaluating output.."
    steps/overlap/get_overlap_segments.py data/$dataset/rttm.annotation | grep "overlap" |\
      md-eval.pl -r - -s exp/overlap_$overlap_affix/$dataset/rttm_overlap |\
      awk 'or(/MISSED SPEAKER TIME/,/FALARM SPEAKER TIME/)'
  done
fi

