#!/bin/bash

# Copyright 2013-2014 MERL (author: Felix Weninger and Shinji Watanabe)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# should run this on a machine that has sufficient memory.

# Requirements) matlab and tcsh
if [ ! `which tcsh` ]; then
  echo "Install tcsh, which is used in some REVERB scripts"
  exit 1
fi
if [ ! `which matlab` ]; then
  echo "Install matlab, which is used to generate multi-condition data"
  exit 1
fi

. ./cmd.sh
. ./path.sh

stage=1
. utils/parse_options.sh
# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail

# please make sure to set the paths of the REVERB and WSJ0 data
if [[ $(hostname -f) == *.clsp.jhu.edu ]] ; then
  REVERB_home=/export/corpora5/REVERB_2014/REVERB
  export wsjcam0=/export/corpora3/LDC/LDC95S24/wsjcam0
  # set LDC WSJ0 directory to obtain LMs
  # REVERB data directory only provides bi-gram (bcb05cnp), but this recipe also uses 3-gram (tcb05cnp.z)
  export wsj0=/export/corpora5/LDC/LDC93S6A/11-13.1 #LDC93S6A or LDC93S6B
  # It is assumed that there will be a 'wsj0' subdirectory
  # within the top-level corpus directory
elif [[ $(hostname -f) == *.merl.com ]] ; then
  REVERB_home=/db/laputa1/data/original/public/REVERB
  export wsjcam0=$REVERB_home/wsjcam0
  # set LDC WSJ0 directory to obtain LMs
  # REVERB data directory only provides bi-gram (bcb05cnp), but this recipe also uses 3-gram (tcb05cnp.z)
  export wsj0=/db/laputa1/data/original/public/WSJ0/11-13.1 #LDC93S6A or LDC93S6B
  # It is assumed that there will be a 'wsj0' subdirectory
  # within the top-level corpus directory
else
  echo "Set the data directory locations." && exit 1;
fi
export reverb_dt=$REVERB_home/REVERB_WSJCAM0_dt
export reverb_et=$REVERB_home/REVERB_WSJCAM0_et
export reverb_real_dt=$REVERB_home/MC_WSJ_AV_Dev
export reverb_real_et=$REVERB_home/MC_WSJ_AV_Eval

# set the directory of the multi-condition training data to be generated
reverb_tr=`pwd`/data_tr_cut/REVERB_WSJCAM0_tr_cut

# LDA context size (left/right) (4 is default)
context_size=4

# The language models with which to decode (tg_5k or bg_5k)
lm="tg_5k"

# number of jobs for feature extraction and model training
nj_train=30

# number of jobs for decoding
nj_decode=8

# set to true if you want the tri2a systems (re-implementation of the HTK baselines)
do_tri2a=true

if [ $stage -le 1 ]; then
  # Generate multi-condition training data
  # Note that utterance lengths match the original set.
  # This enables using clean alignments in multi-condition training (stereo training)
  local/REVERB_create_mcdata.sh $wsjcam0 $reverb_tr
fi

if [ $stage -le 2 ]; then
  # Prepare wsjcam0 clean data and wsj0 language model.
  local/wsjcam0_data_prep.sh $wsjcam0 $wsj0

  # Prepare merged BEEP/CMU dictionary.
  local/wsj_prepare_beep_dict.sh

  # Prepare wordlists, etc.
  utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang

  # Prepare directory structure for clean data. Apply some language model fixes.
  local/wsjcam0_format_data.sh

  # Now it's getting more interesting.
  # Prepare the multi-condition training data and the REVERB dt set.
  # This also extracts MFCC features (!!!)
  # This creates the data sets called REVERB_tr_cut and REVERB_dt.
  # If you have processed waveforms, this is a good starting point to integrate them.
  # For example, you could have something like
  # local/REVERB_wsjcam0_data_prep.sh /path/to/processed/REVERB_WSJCAM0_dt processed_REVERB_dt dt
  # The first argument is supposed to point to a folder that has the same structure
  # as the REVERB corpus.
  local/REVERB_wsjcam0_data_prep.sh $reverb_tr REVERB_tr_cut tr
  local/REVERB_wsjcam0_data_prep.sh $reverb_dt REVERB_dt dt
  local/REVERB_wsjcam0_data_prep.sh $reverb_et REVERB_et et

  # Prepare the REVERB "real" dt set from MCWSJAV corpus.
  # This corpus is *never* used for training.
  # This creates the data set called REVERB_Real_dt and its subfolders
  local/REVERB_mcwsjav_data_prep.sh $reverb_real_dt REVERB_Real_dt dt
  # The MLF file exists only once in the corpus, namely in the real_dt directory
  # so we pass it as 4th argument
  local/REVERB_mcwsjav_data_prep.sh $reverb_real_et REVERB_Real_et et $reverb_real_dt/mlf/WSJ.mlf
fi

if [ $stage -le 3 ]; then
  # Extract MFCC features for clean sets.
  # For the non-clean data sets, this is outsourced to the data preparation scripts.
  mfccdir=mfcc
  ### for x in si_tr si_dt; do it seems that the number of transcriptions of si_dt is not correct.
  for x in si_tr; do
   steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj_train \
     data/$x exp/make_mfcc/$x $mfccdir
   steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
  done
fi

if [ $stage -le 4 ]; then
  # Train monophone model on clean data (si_tr).
  echo "### TRAINING mono0a ###"
  steps/train_mono.sh --boost-silence 1.25 --nj $nj_train --cmd "$train_cmd" \
    data/si_tr data/lang exp/mono0a

  # Align monophones with clean data.
  echo "### ALIGNING mono0a_ali ###"
  steps/align_si.sh --boost-silence 1.25 --nj $nj_train --cmd "$train_cmd" \
    data/si_tr data/lang exp/mono0a exp/mono0a_ali

  # Create first triphone recognizer.
  echo "### TRAINING tri1 ###"
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/si_tr data/lang exp/mono0a_ali exp/tri1

  echo "### ALIGNING tri1_ali ###"
  # Re-align triphones.
  steps/align_si.sh --nj $nj_train --cmd "$train_cmd" \
    data/si_tr data/lang exp/tri1 exp/tri1_ali
fi

# The following code trains and evaluates a delta feature recognizer, which is similar to the HTK
# baseline (but using per-utterance basis fMLLR instead of batch MLLR). This is for reference only.
if $do_tri2a; then
if [ $stage -le 5 ]; then
  # Train tri2a, which is deltas + delta-deltas, on clean data.
  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 15000 data/si_tr data/lang exp/tri1_ali exp/tri2a

  # Re-align triphones using clean data. This gives a smallish performance gain.
  steps/align_si.sh --nj $nj_train --cmd "$train_cmd" \
    data/si_tr data/lang exp/tri2a exp/tri2a_ali

  # Train a multi-condition triphone recognizer.
  # This uses alignments on *clean* data, which is allowed for REVERB.
  # However, we have to use the "cut" version so that the length of the
  # waveforms match.
  # It is actually asserted by the Challenge that clean and multi-condition waves are aligned.
  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 15000 data/REVERB_tr_cut/SimData_tr_for_1ch_A data/lang exp/tri2a_ali exp/tri2a_mc

  # Prepare clean and mc tri2a models for decoding.
  utils/mkgraph.sh data/lang_test_bg_5k exp/tri2a exp/tri2a/graph_bg_5k &
  utils/mkgraph.sh data/lang_test_bg_5k exp/tri2a_mc exp/tri2a_mc/graph_bg_5k &
  wait
fi

if [ $stage -le 6 ]; then
  # decode REVERB dt using tri2a, clean
  for dataset in data/REVERB_*{dt,et}/*; do
    steps/decode.sh --nj $nj_decode --cmd "$decode_cmd" \
      exp/tri2a/graph_bg_5k $dataset exp/tri2a/decode_bg_5k_`echo $dataset | awk -F '/' '{print $2 "_" $3}'` &
  done

  # decode REVERB dt using tri2a, mc
  for dataset in data/REVERB_*{dt,et}/*; do
    steps/decode.sh --nj $nj_decode --cmd "$decode_cmd" \
      exp/tri2a_mc/graph_bg_5k $dataset exp/tri2a_mc/decode_bg_5k_`echo $dataset | awk -F '/' '{print $2 "_" $3}'` &
  done

  # basis fMLLR for tri2a_mc system
  # This computes a transform for every training utterance and computes a basis from that.
  steps/get_fmllr_basis.sh --cmd "$train_cmd" --per-utt true data/REVERB_tr_cut/SimData_tr_for_1ch_A data/lang exp/tri2a_mc

  # Recognition using fMLLR adaptation (per-utterance processing).
  for dataset in data/REVERB_*{dt,et}/*; do
    steps/decode_basis_fmllr.sh --nj $nj_decode --cmd "$decode_cmd" \
      exp/tri2a_mc/graph_bg_5k $dataset exp/tri2a_mc/decode_basis_fmllr_bg_5k_`echo $dataset | awk -F '/' '{print $2 "_" $3}'` &
  done
  wait
fi
fi

if [ $stage -le 7 ]; then
  # Train tri2b recognizer, which uses LDA-MLLT, using the default parameters from the WSJ recipe.
  echo "### TRAINING tri2b ###"
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=$context_size --right-context=$context_size" \
    2500 15000 data/si_tr data/lang exp/tri1_ali exp/tri2b

  # tri2b (LDA-MLLT system) with multi-condition training, using default parameters.
  echo "### TRAINING tri2b_mc ###"
  steps/train_lda_mllt.sh  --cmd "$train_cmd"\
    --splice-opts "--left-context=$context_size --right-context=$context_size" \
    2500 15000 data/REVERB_tr_cut/SimData_tr_for_1ch_A data/lang exp/tri1_ali exp/tri2b_mc
fi

# Prepare tri2b* systems for decoding.
if [ $stage -le 8 ]; then
  echo "### MAKING GRAPH {tri2b,tri2b_mc}/graph_$lm ###"
  for recog in tri2b tri2b_mc; do
    utils/mkgraph.sh data/lang_test_$lm exp/$recog exp/$recog/graph_$lm &
  done
  wait
fi

# discriminative training on top of multi-condition systems
# one could also add tri2b here to have a DT clean recognizer for reference
if [ $stage -le 9 ]; then
  base_recog=tri2b_mc
  bmmi_recog=${base_recog}_mmi_b0.1
  echo "### DT $base_recog --> $bmmi_recog ###"

  # get alignments from base recognizer
  steps/align_si.sh --nj $nj_train --cmd "$train_cmd" \
    --use-graphs true data/REVERB_tr_cut/SimData_tr_for_1ch_A data/lang exp/$base_recog exp/${base_recog}_ali

  # get lattices from base recognizer
  denlats_dir=${base_recog}_denlats
  subsplit=`echo $nj_train \* 2 | bc`
  # DT with multi-condition data ...
  steps/make_denlats.sh --sub-split $subsplit --nj $nj_train --cmd "$decode_cmd" \
    data/REVERB_tr_cut/SimData_tr_for_1ch_A data/lang exp/$base_recog exp/$denlats_dir

  # boosted MMI training
  steps/train_mmi.sh --boost 0.1 --cmd "$train_cmd" \
    data/REVERB_tr_cut/SimData_tr_for_1ch_A \
    data/lang \
    exp/${base_recog}_ali \
    exp/$denlats_dir \
    exp/$bmmi_recog
  cp exp/$base_recog/ali.* exp/$bmmi_recog
fi

# decoding using various recognizers
if [ $stage -le 10 ]; then
  # put tri2b last since it takes longest due to the large mismatch.
  for recog in tri2b_mc tri2b_mc_mmi_b0.1 tri2b; do
    # The graph from the ML directory is used in recipe
    recog2=`echo $recog | sed s/_mmi.*//`
    graph=exp/$recog2/graph_$lm

    echo "### DECODING with $recog, noadapt, $lm ###"
    for dataset in data/REVERB_*{dt,et}/*; do
      decode_suff=${lm}_`echo $dataset | awk -F '/' '{print $2 "_" $3}'`
      steps/decode.sh --nj $nj_decode --cmd "$decode_cmd" \
        $graph $dataset \
        exp/$recog/decode_$decode_suff &
    done
    wait

    echo " ## MBR RESCORING with $recog, noadapt ##"
    for dataset in data/REVERB_*{dt,et}/*; do
      decode_suff=${lm}_`echo $dataset | awk -F '/' '{print $2 "_" $3}'`
      mkdir -p exp/$recog/decode_mbr_$decode_suff
      cp exp/$recog/decode_$decode_suff/lat.*.gz exp/$recog/decode_mbr_$decode_suff
      local/score_mbr.sh --cmd "$decode_cmd" \
	$dataset data/lang_test_$lm/ exp/$recog/decode_mbr_$decode_suff &
    done
    wait

  done # loop recog
fi

# decoding using various recognizers with adaptation
if [ $stage -le 11 ]; then
  # put tri2b last since it takes longest due to the large mismatch.
  for recog in tri2b_mc tri2b_mc_mmi_b0.1 tri2b; do
    # The graph from the ML directory is used in recipe
    recog2=`echo $recog | sed s/_mmi.*//`
    graph=exp/$recog2/graph_$lm

    # set the adaptation data
    if [[ "$recog" =~ _mc ]]; then
      tr_dataset=REVERB_tr_cut/SimData_tr_for_1ch_A
    else
      tr_dataset=si_tr
    fi

    echo "### DECODING with $recog, basis_fmllr, $lm ###"
    steps/get_fmllr_basis.sh --cmd "$train_cmd" --per-utt true data/$tr_dataset data/lang exp/$recog
    for dataset in data/REVERB_*{dt,et}/*; do
      (
	decode_suff=${lm}_`echo $dataset | awk -F '/' '{print $2 "_" $3}'`
        steps/decode_basis_fmllr.sh --nj $nj_decode --cmd "$decode_cmd" \
          $graph $dataset \
          exp/$recog/decode_basis_fmllr_$decode_suff
      ) &
    done
    wait

    echo " ## MBR RESCORING with $recog, basis_fmllr ##"
    for dataset in data/REVERB_*{dt,et}/*; do
      decode_suff=${lm}_`echo $dataset | awk -F '/' '{print $2 "_" $3}'`
      mkdir -p exp/$recog/decode_mbr_basis_fmllr_$decode_suff
      cp exp/$recog/decode_basis_fmllr_$decode_suff/lat.*.gz exp/$recog/decode_mbr_basis_fmllr_$decode_suff
      local/score_mbr.sh --cmd "$decode_cmd" \
        $dataset data/lang_test_$lm/ exp/$recog/decode_mbr_basis_fmllr_$decode_suff &
    done
    wait

  done # loop recog
fi

# get all WERs with lmw=15
if [ $stage -le 12 ]; then
  local/get_results.sh
fi
