#!/bin/bash

# Kaldi ASR baseline for the CHiME-4 Challenge (1ch track: single channel track)
#
# Copyright 2016 University of Sheffield (Jon Barker, Ricard Marxer)
#                Inria (Emmanuel Vincent)
#                Mitsubishi Electric Research Labs (Shinji Watanabe)
#           2017 JHU CLSP (Szu-Jui Chen)
#           2017 JHU CLSP (Aswin Shanmugam Subramanian)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

#####Baseline settings#####
# Usage: 
# Execute './run.sh' to get the models.
# We provide BLSTM masking based enhancement --enhancement single_BLSTMmask
#
# We stopped to support the old CHiME-3/4 baseline. If you want to reproduce the old results
# Please use the old version of Kaldi, e.g., git checkout 9e8ff73648917836d0870c8f6fdd2ff4bdde384f

# Config:
stage=0 # resume training with --stage N
enhancement=single_BLSTMmask #### or your method 

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#####check data and model paths################
# Set a main root directory of the CHiME4 data
# If you use scripts distributed in the CHiME4 package,
chime4_data=`pwd`/../..
# Otherwise, please specify it, e.g.,
chime4_data=/db/laputa1/data/processed/public/CHiME4
chime3_data=/data2/archive/speech-db/original/public/CHiME3

case $(hostname -f) in
  *.clsp.jhu.edu) 
      chime4_data=/export/corpora4/CHiME4/CHiME3 # JHU,
      chime3_data=/export/corpora5/CHiME3 
      ;;
esac 

if [ ! -d $chime4_data ]; then
  echo "$chime4_data does not exist. Please specify chime4 data root correctly" && exit 1;
fi
if [ ! -d $chime3_data ]; then
  echo "$chime3_data does not exist. Please specify chime4 data root correctly" && exit 1;
fi

#####main program start################
# You can execute run_init.sh only "once"
# This creates 3-gram LM, FSTs, and basic task files
if [ $stage -le 0 ]; then
  local/run_init.sh $chime4_data
fi

enhancement_method=isolated_1ch_track
enhancement_data=$chime4_data/data/audio/16kHz/$enhancement_method
# TODO Aswin 
#if [ $stage -le 1 ]; then
#  enhancement_method=single_BLSTMmask
#  enhancement_data=`pwd`/enhan/$enhancement
#  local/run_masking_blstm_1ch_track.sh --cmd "$train_cmd" --nj 20 $chime4_data $chime3_data $enhancement_data 5 
#fi

# Compute PESQ, STOI, eSTOI, and SDR scores
if [ $stage -le 2 ]; then
  if [ ! -f local/bss_eval_sources.m ] || [ ! -f local/stoi.m ] || [ ! -f local/estoi.m ] || [ ! -f local/PESQ ]; then
    # download and install speech enhancement evaluation tools
    local/download_se_eval_tool.sh
  fi
  chime4_rir_data=local/nn-gev/data/audio/16kHz/isolated_ext
  if [ ! -d $chime4_rir_data ]; then
    echo "$chime4_rir_dir does not exist. Please run 'blstm_gev' enhancement method first;" && exit 1;
  fi
  local/compute_PESQ.sh $enhancement $enhancement_data $chime4_rir_data $PWD
  local/compute_stoi_estoi_sdr.sh $enhancement $enhancement_data $chime4_rir_data
fi

# GMM based ASR experiment
# Please set a directory of your speech enhancement method.
# The directory structure and audio files must follow the attached baseline enhancement directory
if [ $stage -le 3 ]; then
  local/run_gmm.sh $enhancement $enhancement_data $chime4_data
fi

# DNN based ASR experiment
# Since it takes time to evaluate DNN, we make the GMM and DNN scripts separately.
# You may execute it after you would have promising results using GMM-based ASR experiments
if [ $stage -le 4 ]; then
  local/chain/run_tdnn.sh $enhancement
fi

# LM-rescoring experiment with 5-gram and RNN LMs
# It takes a few days to train a RNNLM.
if [ $stage -le 5 ]; then
  local/run_lmrescore_tdnn.sh $chime4_data $enhancement
fi

echo "Done."
