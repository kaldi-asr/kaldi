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
sad_stage=0
diarizer_stage=0
decode_stage=0

# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

test_sets="dev eval"

set -e # exit on error

# please change the path accordingly
libricss_corpus=/export/corpora/LibriCSS
librispeech_corpus=/export/corpora/LibriSpeech/

##########################################################################
# We first prepare the LibriCSS data (monoaural) in the Kaldi data
# format. We use sessions () for dev and () for eval.
##########################################################################
if [ $stage -le 0 ]; then
  local/data_prep_mono.sh $libricss_corpus
fi

#########################################################################
# ASR MODEL TRAINING
# In this stage, we prepare the Librispeech data and train our ASR model. 
# This part is taken from the librispeech recipe, with parts related to 
# decoding removed. We use the 100h clean subset to train most of the
# GMM models, except the SAT model, which is trained on the 460h clean
# subset. The nnet is trained on the full 960h (clean + other).
# To avoid training the whole ASR from scratch, you can just train the
# GMM parts (which is required for SAD training) and download the
# chain model using:
# wget http://kaldi-asr.org/models/13/0013_librispeech_s5.tar.gz
# Once it is downloaded, extract using: tar -xvzf 0013_librispeech_s5.tar.gz
# and copy the contents of the exp/ directory to your exp/. 
#########################################################################
if [ $stage -le 1 ]; then
  local/train_asr.sh --stage $asr_stage --nj $nj \
    --gmm-only true $librispeech_corpus
fi

#########################################################################
# SAD MODEL TRAINING 
# For training SAD, you first need to run stage 1 with `--gmm-only true` 
# since SAD training targets are obtained using alignments generated 
# from these GMM models.
#########################################################################
if [ $stage -le 2 ]; then
  local/train_sad.sh --stage $sad_stage --nj $nj \
    --data-dir data/train_other_500 \
    --sat-model-dir exp/tri6b \
    --model-dir exp/tri2b \
    --lang data/lang_nosp \
    --lang-test data/lang_test_tgsmall
fi

##########################################################################
# DIARIZATION MODEL TRAINING
# You can also download a pretrained diarization model using:
# wget http://kaldi-asr.org/models/12/0012_diarization_v1.tar.gz
# Once it is downloaded, extract using: tar -xvzf 0012_diarization_v1.tar.gz
# and copy the contents of the exp/ directory to your exp/
##########################################################################
if [ $stage -le 3 ]; then
  local/train_diarizer.sh --stage $diarizer_stage \
    --data-dir data/train_other_500 \
    --model-dir exp/xvector_nnet_1a
fi

##########################################################################
# DECODING: We assume that we are just given the raw recordings (approx 10
# mins each), without segments or speaker information, so we have to decode 
# the whole pipeline, i.e., SAD -> Diarization -> ASR. This is done in the 
# local/decode.sh script.
##########################################################################
if [ $stage -le 4 ]; then
  local/decode.sh --stage $decode_stage \
    --test-sets $test_sets
fi

exit 0;

