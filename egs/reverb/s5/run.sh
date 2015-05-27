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

# Requirements) matlab and tcsh
if [ ! `which tcsh` ]; then
    echo "Install tcsh, which is used in some REVERB scripts"
    exit 1
fi
if [ ! `which matlab` ]; then
    echo "Install matlab, which is used to generate multi-condition data"
    exit 1
fi

if [ ! -e path.sh ] || [ ! -e corpus.sh ]; then
    echo "ERROR: path.sh and/or corpus.sh not found"
    echo "You need to create these from {path,corpus}.sh.default to match your system"
    echo "Make sure you follow the instructions in ../README.txt"
    exit 1
fi

. ./cmd.sh 

# please make sure to set the paths of the REVERB and WSJ0 data           
. ./corpus.sh

# set the directory of the multi-condition training data generated
reverb_tr=`pwd`/data_tr_cut/REVERB_WSJCAM0_tr_cut

# LDA context size (left/right) (4 is default)
context_size=4

# The language models with which to decode (tg_5k or bg_5k or "tg_5k bg_5k" for
# both)
lms="bg_5k tg_5k"

# number of jobs for feature extraction and model training
nj_train=30

# number of jobs for decoding
# use less jobs for trigram model
# if you have enough RAM (~ 32 GB), you can use 8 jobs for trigram as well
nj_bg=8
nj_tg=8
nj_bg=25 ##
nj_tg=25 ##

# set to true if running from scratch
do_prep=true

# set to true if you want the tri2a systems (re-implementation of the HTK baselines)
do_tri2a=true


# The following are the settings determined by Gaussian Process optimization.
# However, they are not used in the final system.
# You can use the code below for training the "tri2c_mc" system.

# LDA parameters for MCT recognizer.
# Use significantly more context than the default (7 frames ~ 85 ms)
mct_lda_left_context=7
mct_lda_right_context=5

# Number of states and Gaussians for the MCT recognizer.
mct_nstates=7500
mct_ngauss=45000

## End of GP tuned settings

false && {
if $do_prep; then
  # Generate multi-condition training data
  # Note that utterance lengths match the original set.
  # This enables using clean alignments in multi-condition training (stereo training)
  #local/REVERB_create_mcdata.sh $wsjcam0 $reverb_tr

  # Prepare wsjcam0 clean data and wsj0 language model.
  local/wsjcam0_data_prep.sh $wsjcam0 $wsj0 || exit 1

  # Prepare merged BEEP/CMU dictionary.
  local/wsj_prepare_beep_dict.sh || exit 1;

  # Prepare wordlists, etc.
  utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;

  # Prepare directory structure for clean data. Apply some language model fixes.
  local/wsjcam0_format_data.sh || exit 1;

  # Now it's getting more interesting.
  # Prepare the multi-condition training data and the REVERB dt set.
  # This also extracts MFCC features (!!!)
  # This creates the data sets called REVERB_tr_cut and REVERB_dt.
  # If you have processed waveforms, this is a good starting point to integrate them.
  # For example, you could have something like
  # local/REVERB_wsjcam0_data_prep.sh /path/to/processed/REVERB_WSJCAM0_dt processed_REVERB_dt dt
  # The first argument is supposed to point to a folder that has the same structure
  # as the REVERB corpus.
  local/REVERB_wsjcam0_data_prep.sh $reverb_tr REVERB_tr_cut tr || exit 1;
  local/REVERB_wsjcam0_data_prep.sh $reverb_dt REVERB_dt dt     || exit 1;
  local/REVERB_wsjcam0_data_prep.sh $reverb_et REVERB_et et     || exit 1;

  # Prepare the REVERB "real" dt set from MCWSJAV corpus.
  # This corpus is *never* used for training.
  # This creates the data set called REVERB_Real_dt and its subfolders
  local/REVERB_mcwsjav_data_prep.sh $reverb_real_dt REVERB_Real_dt dt || exit 1;
  # The MLF file exists only once in the corpus, namely in the real_dt directory
  # so we pass it as 4th argument
  local/REVERB_mcwsjav_data_prep.sh $reverb_real_et REVERB_Real_et et $reverb_real_dt/mlf/WSJ.mlf || exit 1;

  # Extract MFCC features for clean sets.
  # For the non-clean data sets, this is outsourced to the data preparation scripts.
  mfccdir=mfcc
  ### for x in si_tr si_dt; do it seems that the number of transcriptions of si_dt is not correct.
  for x in si_tr; do 
   steps/make_mfcc.sh --nj $nj_train \
     data/$x exp/make_mfcc/$x $mfccdir || exit 1;
   steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  done
fi

# Train monophone model on clean data (si_tr).
if [ ! -e exp/mono0a/final.mdl ]; then
    echo "### TRAINING mono0a ###"
    steps/train_mono.sh --boost-silence 1.25 --nj $nj_train \
      data/si_tr data/lang exp/mono0a || exit 1;
fi

# Align monophones with clean data.
if [ ! -e exp/mono0a_ali/ali.1.gz ]; then
    echo "### ALIGNING mono0a_ali ###"
    steps/align_si.sh --boost-silence 1.25 --nj $nj_train \
       data/si_tr data/lang exp/mono0a exp/mono0a_ali || exit 1;
fi

# Create first triphone recognizer.
if [ ! -e exp/tri1/final.mdl ]; then
    echo "### TRAINING tri1 ###"
    steps/train_deltas.sh --boost-silence 1.25 \
        2000 10000 data/si_tr data/lang exp/mono0a_ali exp/tri1 || exit 1;
fi

# Prepare first triphone recognizer and decode clean si_dt for verification.
#utils/mkgraph.sh data/lang_test_bg_5k exp/tri1 exp/tri1/graph_bg_5k || exit 1;
#steps/decode.sh --nj 8 exp/tri1/graph_bg_5k data/si_dt exp/tri1/decode_si_dt

if [ ! -e exp/tri1_ali/ali.1.gz ]; then
    echo "### ALIGNING tri1_ali ###"
    # Re-align triphones.
    steps/align_si.sh --nj $nj_train \
      data/si_tr data/lang exp/tri1 exp/tri1_ali || exit 1;
fi


# The following code trains and evaluates a delta feature recognizer, which is similar to the HTK
# baseline (but using per-utterance basis fMLLR instead of batch MLLR). This is for reference only.
if $do_tri2a; then
  # Train tri2a, which is deltas + delta-deltas, on clean data.
  steps/train_deltas.sh \
    2500 15000 data/si_tr data/lang exp/tri1_ali exp/tri2a || exit 1;

  # Re-align triphones using clean data. This gives a smallish performance gain.
  steps/align_si.sh --nj $nj_train \
    data/si_tr data/lang exp/tri2a exp/tri2a_ali || exit 1;

  # Train a multi-condition triphone recognizer.
  # This uses alignments on *clean* data, which is allowed for REVERB.
  # However, we have to use the "cut" version so that the length of the 
  # waveforms match.
  # It is actually asserted by the Challenge that clean and multi-condition waves are aligned.
  steps/train_deltas.sh \
    2500 15000 data/REVERB_tr_cut/SimData_tr_for_1ch_A data/lang exp/tri2a_ali exp/tri2a_mc || exit 1;

  # Prepare clean and mc tri2a models for decoding.
  utils/mkgraph.sh data/lang_test_bg_5k exp/tri2a exp/tri2a/graph_bg_5k
  utils/mkgraph.sh data/lang_test_bg_5k exp/tri2a_mc exp/tri2a_mc/graph_bg_5k

  # decode REVERB dt using tri2a, clean
  for dataset in data/REVERB_dt/SimData_dt* data/REVERB_Real_dt/RealData_dt*; do
    steps/decode.sh --nj $nj_bg \
      exp/tri2a/graph_bg_5k $dataset exp/tri2a/decode_bg_5k_REVERB_dt_`basename $dataset` || exit 1;
  done

  # decode REVERB dt using tri2a, mc
  for dataset in data/REVERB_dt/SimData_dt* data/REVERB_Real_dt/RealData_dt*; do
    steps/decode.sh --nj $nj_bg \
      exp/tri2a_mc/graph_bg_5k $dataset exp/tri2a_mc/decode_bg_5k_REVERB_dt_`basename $dataset` || exit 1;
  done
  # basis fMLLR for tri2a_mc system
  # This computes a transform for every training utterance and computes a basis from that.
  steps/get_fmllr_basis.sh --per-utt true data/REVERB_tr_cut/SimData_tr_for_1ch_A data/lang exp/tri2a_mc || exit 1;

  # Recognition using fMLLR adaptation (per-utterance processing).
  for dataset in data/REVERB_dt/SimData_dt* data/REVERB_Real_dt/RealData_dt*; do
    steps/decode_basis_fmllr.sh --nj $nj_bg \
      exp/tri2a_mc/graph_bg_5k $dataset exp/tri2a_mc/decode_basis_fmllr_bg_5k_REVERB_dt_`basename $dataset` || exit 1;
  done

fi # train tri2a, tri2a_mc


# Train tri2b recognizer, which uses LDA-MLLT, using the default parameters from the WSJ recipe.
if [ ! -e exp/tri2b/final.mdl ]; then
    echo "### TRAINING tri2b ###"
    steps/train_lda_mllt.sh \
       --splice-opts "--left-context=$context_size --right-context=$context_size" \
       2500 15000 data/si_tr data/lang exp/tri1_ali exp/tri2b || exit 1;
fi

# tri2b (LDA-MLLT system) with multi-condition training, using default parameters.
if [ ! -e exp/tri2b_mc/final.mdl ]; then
    echo "### TRAINING tri2b_mc ###"
    steps/train_lda_mllt.sh \
       --splice-opts "--left-context=$context_size --right-context=$context_size" \
       2500 15000 data/REVERB_tr_cut/SimData_tr_for_1ch_A data/lang exp/tri1_ali exp/tri2b_mc || exit 1;
fi


# tri2c (LDA-MLLT system) with multi-condition training, optimized parameters.
# Disabled by default -- it only improves slightly, and tends to overfit.
if [ ! -e exp/tri2c_mc/final.mdl ]; then
    echo "### TRAINING tri2c_mc ###"
    steps/train_lda_mllt.sh \
       --splice-opts "--left-context=$mct_lda_left_context --right-context=$mct_lda_right_context" \
       $mct_nstates $mct_ngauss data/REVERB_tr_cut/SimData_tr_for_1ch_A data/lang exp/tri1_ali exp/tri2c_mc || exit 1;
fi


# Prepare tri2b* systems for decoding.
for recog in tri2b tri2b_mc; do
    for lm in $lms; do
        graph=exp/$recog/graph_$lm
        if [ ! -e "$graph" ]; then
            echo "### MAKING GRAPH $graph ###"
            utils/mkgraph.sh data/lang_test_$lm exp/$recog $graph || exit 1;
        fi
    done
done


# discriminative training on top of multi-condition systems
# one could also add tri2b here to have a DT clean recognizer for reference
for base_recog in tri2b_mc; do

    bmmi_recog=${base_recog}_mmi_b0.1
    echo "### DT $base_recog --> $bmmi_recog ###"

    # get alignments from base recognizer
    if [ ! -e exp/${base_recog}_ali/ali.1.gz ]; then
        steps/align_si.sh --nj $nj_train \
          --use-graphs true data/REVERB_tr_cut/SimData_tr_for_1ch_A data/lang exp/$base_recog exp/${base_recog}_ali || exit 1;
    fi

    # get lattices from base recognizer
    denlats_dir=${base_recog}_denlats
    subsplit=`echo $nj_train \* 2 | bc`
    if [ ! -e exp/$denlats_dir/.done.1 ]; then
        # DT with multi-condition data ...
        steps/make_denlats.sh --sub-split $subsplit --nj $nj_train \
          data/REVERB_tr_cut/SimData_tr_for_1ch_A data/lang exp/$base_recog exp/$denlats_dir || exit 1;
    fi

    # boosted MMI training
    if [ ! -e exp/$bmmi_recog/final.mdl ]; then
        steps/train_mmi.sh --boost 0.1 \
          data/REVERB_tr_cut/SimData_tr_for_1ch_A \
          data/lang \
          exp/${base_recog}_ali \
          exp/$denlats_dir \
          exp/$bmmi_recog  || exit 1;
        cp exp/$base_recog/ali.* exp/$bmmi_recog
    fi

done

}

# decoding using bigram / trigram and various recognizers
do_adapt=true
for lm in $lms; do
    if [[ "$lm" =~ tg ]]; then
        nj=$nj_tg
    else
        nj=$nj_bg
    fi
    # put tri2b last since it takes longest due to the large mismatch.
    for recog in tri2b_mc tri2b_mc_mmi_b0.1 tri2b; do
        # The graph from the ML directory is used in recipe
        recog2=`echo $recog | sed s/_mmi.*//`
        graph=exp/$recog2/graph_$lm
        for dataset in data/REVERB_dt/SimData_dt* \
                       data/REVERB_et/SimData_et* \
                       data/REVERB_Real_dt/RealData_dt* \
                       data/REVERB_Real_et/RealData_et*; do
            if [[ $dataset =~ _dt ]]; then
                pdataset=REVERB_dt
            elif [[ $dataset =~ _et ]]; then
                pdataset=REVERB_et
            else
                echo "$0: Cannot figure out what to do with: $dataset"
                exit 1
            fi
            #pdataset=$(basename $(dirname $dataset))
            #echo $pdataset
            decode_suff=${lm}_${pdataset}_`basename $dataset`
            if [ ! -e exp/$recog/decode_$decode_suff/wer_15 ]; then
                echo "### DECODING $dataset | $recog, noadapt, $lm ###"
                steps/decode.sh --nj $nj \
                    $graph $dataset \
                    exp/$recog/decode_$decode_suff || exit 1;
            fi
            if [ ! -e exp/$recog/decode_mbr_$decode_suff/wer_15 ]; then
                mkdir -p exp/$recog/decode_mbr_$decode_suff
                cp exp/$recog/decode_$decode_suff/lat.*.gz exp/$recog/decode_mbr_$decode_suff 
                echo " ## MBR RESCORING $dataset | $recog, noadapt ##"
                local/score_mbr.sh \
                    $dataset data/lang_test_$lm/ exp/$recog/decode_mbr_$decode_suff || exit 1
            fi
            if $do_adapt; then
                if [ ! -e exp/$recog/fmllr.basis ]; then
                    if [[ "$recog" =~ _mc ]]; then
                        tr_dataset=REVERB_tr_cut/SimData_tr_for_1ch_A
                    else
                        tr_dataset=si_tr
                    fi
                    steps/get_fmllr_basis.sh --per-utt true data/$tr_dataset data/lang exp/$recog || exit 1;
                fi
                if [ ! -e exp/$recog/decode_basis_fmllr_$decode_suff/wer_15 ]; then
                    echo "### DECODING $dataset | $recog, basis_fmllr, $lm ###"
                    steps/decode_basis_fmllr.sh --nj $nj \
                        $graph $dataset \
                        exp/$recog/decode_basis_fmllr_$decode_suff || exit 1;
                fi
                if [ ! -e exp/$recog/decode_mbr_basis_fmllr_$decode_suff/wer_15 ]; then
                    mkdir -p exp/$recog/decode_mbr_basis_fmllr_$decode_suff
                    cp exp/$recog/decode_basis_fmllr_$decode_suff/lat.*.gz exp/$recog/decode_mbr_basis_fmllr_$decode_suff 
                    echo " ## MBR RESCORING $dataset | $recog, basis_fmllr ##"
                    local/score_mbr.sh \
                        $dataset data/lang_test_$lm/ exp/$recog/decode_mbr_basis_fmllr_$decode_suff || exit 1
                fi
            fi

        done # loop data set
    done # loop recog
done # loop LM

# get all WERs with lmw=15
local/get_results.sh
