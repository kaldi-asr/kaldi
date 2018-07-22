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
# get the models. If you want to use DNN instead of TDNN, add option "--tdnn false".
# Then execute './run.sh --baseline advanced' for your experiments. 

# Config:
stage=0 # resume training with --stage N

baseline=advanced
flatstart=false
tdnn=true

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
case $baseline in
  chime4_official)
      if $flatstart; then
        echo "We don't support this anymore for 'chime4_official' baseline"
        echo " ... Automatically set it to false"
      fi
      modeldir=$chime4_data/tools/ASR_models
      flatstart=false
      ;;
  advanced)
      modeldir=`pwd`
      ;;
  *)
      echo "Usage: './run.sh --baseline chime4_official' or './run.sh --baseline advanced'"
      echo " ... If you haven't run flatstart for advanced baseline, please execute"
      echo " ... './run.sh --baseline advanced --flatstart true' first";
      exit 1;
esac

if [ "$flatstart" = false ]; then
  for d in $modeldir $modeldir/data/{lang,lang_test_tgpr_5k,lang_test_5gkn_5k,lang_test_rnnlm_5k_h300,local} \
    $modeldir/exp/{tri3b_tr05_multi_noisy,tri4a_dnn_tr05_multi_noisy,tri4a_dnn_tr05_multi_noisy_smbr_i1lats}; do
    [ ! -d $d ] && echo "$0: no such directory $d. specify models correctly" && \
    echo " or execute './run.sh --baseline advanced --flatstart true' first" && exit 1;
  done
fi
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
  if $tdnn; then
    if $flatstart; then
      local/chain/run_tdnn.sh $enhancement_method
    else
      local/chain/run_tdnn_recog.sh $enhancement_method $modeldir
    fi
  else
    if $flatstart; then
      local/run_dnn.sh $enhancement_method
    else
      local/run_dnn_recog.sh $enhancement_method $modeldir
    fi
  fi
fi

# LM-rescoring experiment with 5-gram and RNN LMs
# It takes a few days to train a RNNLM.
if [ $stage -le 4 ]; then
  if $flatstart; then
    if $tdnn; then
      local/run_lmrescore_tdnn.sh $chime4_data $enhancement_method
    else
      local/run_lmrescore.sh $chime4_data $enhancement_method
    fi
  else
    if $tdnn; then
      local/run_lmrescore_tdnn_recog.sh $enhancement_method $modeldir
    else
      local/run_lmrescore_recog.sh $enhancement_method $modeldir
    fi
  fi
fi

echo "Done."
