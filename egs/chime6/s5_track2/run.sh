  
#!/bin/bash
#
# Chime-6 Track 2 baseline. Based mostly on the Chime-5 recipe, with the exception
# that we are required to perform speech activity detection and speaker
# diarization before ASR, since we do not have access to the oracle SAD and 
# diarization labels.
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe and Yenda Trmal)
#            2019  Desh Raj, David Snyder, Ashish Arora
# Apache 2.0

# Begin configuration section.
nj=50
decode_nj=20
stage=0
chain_stage=0
sad_stage=0
diarizer_stage=0
decode_stage=0
enhancement=beamformit # for a new enhancement method,
                       # change this variable and decode stage
decode_only=false
# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

if [ $decode_only == "true" ]; then
  stage=17
fi

set -e # exit on error

# chime6 data is same as chime5 data
# chime5 main directory path
# please change the path accordingly
chime5_corpus=/export/corpora4/CHiME5
json_dir=${chime5_corpus}/transcriptions
audio_dir=${chime5_corpus}/audio

# training and test data
train_set=train_worn_simu_u400k
test_sets="dev_${enhancement}_dereverb_ref eval_${enhancement}_dereverb_ref"

# This script also needs the phonetisaurus g2p, srilm, beamformit
./local/check_tools.sh || exit 1

###########################################################################
# We prepare dict and lang in stages 1 to 3.
###########################################################################

if [ $stage -le 1 ]; then
  # skip u03 as they are missing
  for mictype in worn u01 u02 u04 u05 u06; do
    local/prepare_data.sh --mictype ${mictype} --train true \
        ${audio_dir}/train ${json_dir}/train data/train_${mictype}
  done
  for dataset in dev; do
    for mictype in worn; do
      local/prepare_data.sh --mictype ${mictype} --train true \
          ${audio_dir}/${dataset} ${json_dir}/${dataset} \
          data/${dataset}_${mictype}
    done
  done
fi

if [ $stage -le 2 ]; then
  local/prepare_dict.sh

  utils/prepare_lang.sh \
    data/local/dict "<unk>" data/local/lang data/lang

  local/train_lms_srilm.sh \
    --train-text data/train_worn/text --dev-text data/dev_worn/text \
    --oov-symbol "<unk>" --words-file data/lang/words.txt \
    data/ data/srilm
fi

LM=data/srilm/best_3gram.gz
if [ $stage -le 3 ]; then
  # Compiles G for chime5 trigram LM
  utils/format_lm.sh \
    data/lang $LM data/local/dict/lexicon.txt data/lang

fi

#########################################################################################
# In stages 4 to 6, we modify train and dev data for our training purpose. Here we
# use 400k utterances from array microphones and all the worn set utterances in train.
#########################################################################################

if [ $stage -le 4 ]; then
  # remove possibly bad sessions (P11_S03, P52_S19, P53_S24, P54_S24)
  # see http://spandh.dcs.shef.ac.uk/chime_challenge/data.html for more details
  utils/copy_data_dir.sh data/train_worn data/train_worn_org # back up
  grep -v -e "^P11_S03" -e "^P52_S19" -e "^P53_S24" -e "^P54_S24" data/train_worn_org/text > data/train_worn/text
  utils/fix_data_dir.sh data/train_worn
fi

if [ $stage -le 5 ]; then
  # combine mix array and worn mics
  # randomly extract first 400k utterances from all mics
  # if you want to include more training data, you can increase the number of array mic utterances
  utils/combine_data.sh data/train_uall data/train_u01 data/train_u02 data/train_u04 data/train_u05 data/train_u06
  utils/subset_data_dir.sh data/train_uall 400000 data/train_u400k
  utils/combine_data.sh data/${train_set} data/train_worn data/train_u400k

  # only use left channel for worn mic recognition
  # you can use both left and right channels for training
  for dset in train dev; do
    utils/copy_data_dir.sh data/${dset}_worn data/${dset}_worn_stereo
    grep "\.L-" data/${dset}_worn_stereo/text > data/${dset}_worn/text
    utils/fix_data_dir.sh data/${dset}_worn
  done
fi

if [ $stage -le 6 ]; then
  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  for dset in ${train_set}; do
    utils/copy_data_dir.sh data/${dset} data/${dset}_nosplit
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}_nosplit data/${dset}
  done
fi

##################################################################################
# Now make MFCC features. We use 40-dim "hires" MFCCs for all our systems.
##################################################################################

if [ $stage -le 7 ]; then
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  mfccdir=mfcc
  for x in ${train_set}; do
    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
      --mfcc-config conf/mfcc_hires.conf \
      data/$x exp/make_mfcc/$x $mfccdir
  done
fi

###################################################################################
# Stages 8 to 13 train monophone and triphone models. They will be used for 
# generating lattices for training the chain model and for obtaining targets
# for training the SAD system.
###################################################################################

if [ $stage -le 8 ]; then
  # make a subset for monophone training
  utils/subset_data_dir.sh --shortest data/${train_set} 100000 data/${train_set}_100kshort
  utils/subset_data_dir.sh data/${train_set}_100kshort 30000 data/${train_set}_30kshort
fi

if [ $stage -le 9 ]; then
  # Starting basic training on MFCC features
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
          data/${train_set}_30kshort data/lang exp/mono
fi

if [ $stage -le 10 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
        data/${train_set} data/lang exp/mono exp/mono_ali

  steps/train_deltas.sh --cmd "$train_cmd" \
      2500 30000 data/${train_set} data/lang exp/mono_ali exp/tri1
fi

if [ $stage -le 11 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
        data/${train_set} data/lang exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
        4000 50000 data/${train_set} data/lang exp/tri1_ali exp/tri2
fi

if [ $stage -le 12 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
        data/${train_set} data/lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd "$train_cmd" \
         5000 100000 data/${train_set} data/lang exp/tri2_ali exp/tri3
fi

if [ $stage -le 13 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set} data/lang exp/tri3 exp/tri3_ali
fi

##########################################################################
# CHAIN MODEL TRAINING
##########################################################################
if [ $stage -le 14 ]; then
  local/chain/run_tdnn.sh --stage $chain_stage
fi

##########################################################################
# SAD MODEL TRAINING
##########################################################################
# Now run the SAD script. This contains stages 15 to 19.
# If you just want to perform SAD decoding (without
# training), run from stage 19.
if [ $stage -le 15 ]; then
  local/train_sad.sh --stage $sad_stage \
    --data-dir data/${train_set} --test-sets "${test_sets}" \
    --sat-model-dir exp/tri3 \
    --model-dir exp/tri2
fi

##########################################################################
# DIARIZATION MODEL TRAINING
##########################################################################
if [ $stage -le 16 ]; then
  local/train_diarizer.sh --stage $diarizer_stage \
    --data-dir data/${train_set} \
    --model-dir exp/xvector_nnet_1a
fi

##########################################################################
# DECODING: In track 2, we are given raw utterances without segment
# or speaker information, so we have to decode the whole pipeline, i.e.,
# SAD -> Diarization -> ASR. This is done in the local/decode.sh
# script.
##########################################################################
if [ $stage -le 17 ]; then
  local/decode.sh --stage $decode_stage \
    --enhancement $enhancement \
    --test-sets "$test_sets"
fi

exit 0;

