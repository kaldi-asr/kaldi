#!/usr/bin/env bash
#
# LibriCSS monoaural baseline recipe.
#
# Copyright  2020  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

# Begin configuration section.
nj=50
decode_nj=20
stage=0

# Different stages
asr_stage=1
diarizer_stage=0
decode_stage=0
rnnlm_rescore=true

data_affix=  # This can be used to distinguish between different data sources

use_oracle_segments=false
wpe=false

# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

test_sets="dev${data_affix} eval${data_affix}"

set -e # exit on error

# please change the path accordingly
libricss_corpus=/export/fs01/LibriCSS
librispeech_corpus=/export/corpora5/LibriSpeech/

##########################################################################
# We first prepare the LibriCSS data (monoaural) in the Kaldi data
# format. We use session 0 for dev and others for eval.
##########################################################################
if [ $stage -le 0 ]; then
  local/data_prep_mono.sh --data-affix "$data_affix" $libricss_corpus $librispeech_corpus
fi

#########################################################################
# ASR MODEL TRAINING
# In this stage, we prepare the Librispeech data and train our ASR model. 
# This part is taken from the librispeech recipe, with parts related to 
# decoding removed. We use the 100h clean subset to train most of the
# GMM models, except the SAT model, which is trained on the 460h clean
# subset. The nnet is trained on the full 960h (clean + other).
# To avoid training the whole ASR from scratch, you can download the
# chain model using:
# wget http://kaldi-asr.org/models/13/0013_librispeech_s5.tar.gz
# Once it is downloaded, extract using: tar -xvzf 0013_librispeech_s5.tar.gz
# and copy the contents of the exp/ directory to your exp/. 
#########################################################################
if [ $stage -le 1 ]; then
  local/train_asr.sh --stage $asr_stage --nj $nj $librispeech_corpus
fi

##########################################################################
# DIARIZATION MODEL TRAINING
# You can also download a pretrained diarization model using:
# wget http://kaldi-asr.org/models/12/0012_diarization_v1.tar.gz
# Once it is downloaded, extract using: tar -xvzf 0012_diarization_v1.tar.gz
# and copy the contents of the exp/ directory to your exp/
##########################################################################
if [ $stage -le 2 ]; then
  local/train_diarizer.sh --stage $diarizer_stage \
    --data-dir data/train_other_500 \
    --model-dir exp/xvector_nnet_1a
fi

##########################################################################
# RNNLM TRAINING
# We train a TDNN-LSTM based LM that will be used for rescoring the 
# decoded lattices.
##########################################################################
if [ $stage -le 3 ]; then
  local/rnnlm/train.sh --stage $rnnlm_stage
fi

##########################################################################
# DECODING: We assume that we are just given the raw recordings (approx 10
# mins each), without segments or speaker information, so we have to decode 
# the whole pipeline, i.e., SAD -> Diarization -> ASR. This is done in the 
# local/decode.sh script.
##########################################################################
if [ $stage -le 4 ]; then
  local/decode.sh --stage $decode_stage \
    --test-sets "$test_sets" \
    --use-oracle-segments $use_oracle_segments \
    --rnnlm-rescore $rnnlm_rescore
fi

exit 0;

