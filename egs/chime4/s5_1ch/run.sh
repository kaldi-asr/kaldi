#!/bin/bash

# Kaldi ASR baseline for the CHiME-4 Challenge (1ch track: single channel track)
#
# Copyright 2016 University of Sheffield (Jon Barker, Ricard Marxer)
#                Inria (Emmanuel Vincent)
#                Mitsubishi Electric Research Labs (Shinji Watanabe)
#           2017 JHU CLSP (Szu-Jui Chen)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh
#####Baseline settings#####
# Usage: 
# 1. For using original baseline, execute './run.sh --baseline chime4_official'. 
# We don't provide the function to train original baseline models anymore. Instead, we provided the
# trained original baseline models in tools/ASR_models for directly using.
#
# 2. For using advanced baseline, first execute './run.sh --baseline advanced --flatstart true' to
# get the models. If you want to use TDNN instead of DNN, add option "--tdnn true". If you want to
# use TDNN-LSTM instead of DNN, add option "--tdnn-lstm true".
# Then execute './run.sh --baseline advanced' for your experiments.

# Config:
stage=0 # resume training with --stage N

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


case $(hostname -f) in
  *.clsp.jhu.edu) chime4_data=/export/corpora4/CHiME4/CHiME3 ;; # JHU,
esac 

if [ ! -d $chime4_data ]; then
  echo "$chime4_data does not exist. Please specify chime4 data root correctly" && exit 1
fi
# Set a model directory for the CHiME4 data.
modeldir=`pwd`

#####check data and model paths finished#######


#####main program start################
# You can execute run_init.sh only "once"
# This creates 3-gram LM, FSTs, and basic task files
if [ $stage -le 0 ]; then
  local/run_init.sh $chime4_data
fi

# In this script, we use non-enhanced 6th microphone signals.
enhancement_method=isolated_1ch_track
enhancement_data=$chime4_data/data/audio/16kHz/$enhancement_method
#if [ $stage -le 1 ]; then
#  put your single channel enhancement
#fi

# GMM based ASR experiment without "retraining"
# Please set a directory of your speech enhancement method.
# run_gmm_recog.sh can be done every time when you change a speech enhancement technique.
# The directory structure and audio files must follow the attached baseline enhancement directory
if [ $stage -le 2 ]; then
  local/run_gmm.sh $enhancement_method $enhancement_data $chime4_data
fi

# DNN based ASR experiment
# Since it takes time to evaluate DNN, we make the GMM and DNN scripts separately.
# You may execute it after you would have promising results using GMM-based ASR experiments
if [ $stage -le 3 ]; then
  local/chain/run_tdnn.sh $enhancement_method
fi

# LM-rescoring experiment with 5-gram and RNN LMs
# It takes a few days to train a RNNLM.
if [ $stage -le 4 ]; then
  local/run_lmrescore_tdnn.sh $chime4_data $enhancement_method
fi

echo "Done."
