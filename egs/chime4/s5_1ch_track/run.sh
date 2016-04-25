#!/bin/bash

# Kaldi ASR baseline for the CHiME-4 Challenge (1ch track: single channel track)
#
# Copyright 2016 University of Sheffield (Jon Barker, Ricard Marxer)
#                Inria (Emmanuel Vincent)
#                Mitsubishi Electric Research Labs (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# Config:
stage=0 # resume training with --stage=N
flatstart=false

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
if [ ! -d $chime4_data ]; then
  echo "$chime4_data does not exist. Please specify chime4 data root correctly" && exit 1
fi
# Set a model directory for the CHiME4 data.
modeldir=$chime4_data/tools/ASR_models
for d in $modeldir $modeldir/data/{lang,lang_test_tgpr_5k,lang_test_5gkn_5k,lang_test_rnnlm_5k_h300,local} \
  $modeldir/exp/{tri3b_tr05_multi_noisy,tri4a_dnn_tr05_multi_noisy,tri4a_dnn_tr05_multi_noisy_smbr_i1lats}; do
  [ ! -d ] && echo "$0: no such directory $d. specify models correctly or execute './run.sh --flatstart true' first" && exit 1;
done
#####check data and model paths finished#######


#####main program start################
# You can execute run_init.sh only "once"
# This creates 3-gram LM, FSTs, and basic task files
if [ $stage -le 0 ] && $flatstart; then
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
  if $flatstart; then
    local/run_gmm.sh $enhancement_method $enhancement_data $chime4_data
  else
    local/run_gmm_recog.sh $enhancement_method $enhancement_data $modeldir
  fi
fi

# DNN based ASR experiment
# Since it takes time to evaluate DNN, we make the GMM and DNN scripts separately.
# You may execute it after you would have promising results using GMM-based ASR experiments
if [ $stage -le 3 ]; then
  if $flatstart; then
    local/run_dnn.sh $enhancement_method
  else
    local/run_dnn_recog.sh $enhancement_method $modeldir
  fi
fi

# LM-rescoring experiment with 5-gram and RNN LMs
# It takes a few days to train a RNNLM.
if [ $stage -le 4 ]; then
  if $flatstart; then
    local/run_lmrescore.sh $chime4_data $enhancement_method
  else
    local/run_lmrescore_recog.sh $enhancement_method $modeldir
  fi
fi

echo "Done."
