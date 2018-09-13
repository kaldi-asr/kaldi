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

./local/check_tools.sh || exit 1

. ./cmd.sh
. ./path.sh

stage=0

. utils/parse_options.sh
# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail

# please make sure to set the paths of the REVERB and WSJ0 data
if [[ $(hostname -f) == *.clsp.jhu.edu ]] ; then
  reverb=/export/corpora5/REVERB_2014/REVERB
  export wsjcam0=/export/corpora3/LDC/LDC95S24/wsjcam0
  # set LDC WSJ0 directory to obtain LMs
  # REVERB data directory only provides bi-gram (bcb05cnp), but this recipe also uses 3-gram (tcb05cnp.z)
  export wsj0=/export/corpora5/LDC/LDC93S6A/11-13.1 #LDC93S6A or LDC93S6B
  # It is assumed that there will be a 'wsj0' subdirectory
  # within the top-level corpus directory
else
  echo "Set the data directory locations." && exit 1;
fi

#training set and test set
train_set=tr_simu_8ch
test_sets="dt_real_1ch dt_simu_1ch et_real_1ch et_simu_1ch"

# LDA context size (left/right) (4 is default)
context_size=4

# The language models with which to decode (tg_5k or bg_5k)
lm="tg_5k"

# number of jobs for feature extraction and model training
nj=92
decode_nj=20

# number of jobs for decoding
nj_decode=8

# set to true if you want the tri2a systems (re-implementation of the HTK baselines)
do_tri2a=true

if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to make the following data preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    wavdir=$PWD/wav
    echo "stage 0: Data preparation"
    local/generate_data.sh --wavdir ${wavdir} ${wsjcam0}
    local/prepare_simu_data.sh --wavdir ${wavdir} ${reverb} ${wsjcam0}
    local/prepare_real_data.sh ${reverb}
fi

if [ $stage -le 2 ]; then
  # Prepare wsjcam0 clean data and wsj0 language model.
  # local/wsjcam0_data_prep.sh $wsjcam0 $wsj0

  # # Prepare merged BEEP/CMU dictionary.
  # local/wsj_prepare_beep_dict.sh

  # # Prepare wordlists, etc.
 
  # utils/prepare_lang.sh data/local/dict "<NOISE>" data/local/lang_tmp data/lang

  # # Prepare directory structure for clean data. Apply some language model fixes.
  # local/wsjcam0_format_data.sh
  
  local/train_lms_srilm.sh \
    --train-text data/${train_set}/text --dev-text data/dt_simu_8ch/text \
    --oov-symbol "<NOISE>" --words-file data/lang/words.txt \
    data/ data/srilm
  
  LM=data/srilm/best_3gram.gz
  # Compiles G for reverb trigram LM
  utils/format_lm.sh \
		data/lang $LM data/local/dict/lexicon.txt data/lang

  # Now it's getting more interesting.
  # Prepare the multi-condition training data and the REVERB dt set.
  # This also extracts MFCC features (!!!)
  # This creates the data sets called REVERB_tr_cut and REVERB_dt.
  # If you have processed waveforms, this is a good starting point to integrate them.
  # For example, you could have something like
  # local/REVERB_wsjcam0_data_prep.sh /path/to/processed/REVERB_WSJCAM0_dt processed_REVERB_dt dt
  # The first argument is supposed to point to a folder that has the same structure
  # as the REVERB corpus.
  # local/REVERB_wsjcam0_data_prep.sh $reverb_tr REVERB_tr_cut tr
  # local/REVERB_wsjcam0_data_prep.sh $reverb_dt REVERB_dt dt
  # local/REVERB_wsjcam0_data_prep.sh $reverb_et REVERB_et et

  # # Prepare the REVERB "real" dt set from MCWSJAV corpus.
  # # This corpus is *never* used for training.
  # # This creates the data set called REVERB_Real_dt and its subfolders
  # local/REVERB_mcwsjav_data_prep.sh $reverb_real_dt REVERB_Real_dt dt
  # # The MLF file exists only once in the corpus, namely in the real_dt directory
  # # so we pass it as 4th argument
  # local/REVERB_mcwsjav_data_prep.sh $reverb_real_et REVERB_Real_et et $reverb_real_dt/mlf/WSJ.mlf
fi

if [ $stage -le 3 ]; then
  for dset in ${train_set} ${test_sets}; do
    utils/copy_data_dir.sh data/${dset} data/${dset}_nosplit
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}_nosplit data/${dset}
  done
fi

if [ $stage -le 4 ]; then
  # Extract MFCC features for clean sets.
  # For the non-clean data sets, this is outsourced to the data preparation scripts.
  mfccdir=mfcc
  ### for x in si_tr si_dt; do it seems that the number of transcriptions of si_dt is not correct.
  for x in ${train_set} ${test_sets}; do
   steps/make_mfcc.sh --cmd "$train_cmd" --nj 30 \
     data/$x exp/make_mfcc/$x $mfccdir
   steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
  done
fi

# if [ $stage -le 5 ]; then
  # make a subset for monophone training
  #utils/subset_data_dir.sh --shortest data/${train_set} 30000 data/${train_set}_30kshort
  #utils/subset_data_dir.sh data/${train_set}_10kshort 4000 data/${train_set}_4kshort
# fi

if [ $stage -le 6 ]; then
  # Starting basic training on MFCC features
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
		      data/${train_set} data/lang exp/mono
fi

if [ $stage -le 7 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
		    data/${train_set} data/lang exp/mono exp/mono_ali

  steps/train_deltas.sh --cmd "$train_cmd" \
			2500 30000 data/${train_set} data/lang exp/mono_ali exp/tri1
fi

if [ $stage -le 8 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
		    data/${train_set} data/lang exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
			  4000 50000 data/${train_set} data/lang exp/tri1_ali exp/tri2
fi

if [ $stage -le 9 ]; then
  utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph
  for dset in ${test_sets}; do
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
		    exp/tri2/graph data/${dset} exp/tri2/decode_${dset} &
  done
  wait
fi

if [ $stage -le 10 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
		    data/${train_set} data/lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd "$train_cmd" \
		     5000 100000 data/${train_set} data/lang exp/tri2_ali exp/tri3
fi

if [ $stage -le 11 ]; then
  utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph
  for dset in ${test_sets}; do
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
			  exp/tri3/graph data/${dset} exp/tri3/decode_${dset} &
  done
  wait
fi

if [ $stage -le 12 ]; then
  # chain TDNN
  local/chain/run_tdnn.sh --nj ${nj} --train-set ${train_set} --test-sets "$test_sets" --gmm tri3 --nnet3-affix _${train_set}
fi
exit 1
# # decoding using various recognizers
# if [ $stage -le 16 ]; then
  # # put tri2b last since it takes longest due to the large mismatch.
  # for recog in tri2b_mc tri2b_mc_mmi_b0.1 tri2b; do
    # # The graph from the ML directory is used in recipe
    # recog2=`echo $recog | sed s/_mmi.*//`
    # graph=exp/$recog2/graph_$lm

    # echo "### DECODING with $recog, noadapt, $lm ###"
    # for dataset in data/REVERB_*{dt,et}/*; do
      # decode_suff=${lm}_`echo $dataset | awk -F '/' '{print $2 "_" $3}'`
      # steps/decode.sh --nj $nj_decode --cmd "$decode_cmd" \
        # $graph $dataset \
        # exp/$recog/decode_$decode_suff &
    # done
    # wait

    # echo " ## MBR RESCORING with $recog, noadapt ##"
    # for dataset in data/REVERB_*{dt,et}/*; do
      # decode_suff=${lm}_`echo $dataset | awk -F '/' '{print $2 "_" $3}'`
      # mkdir -p exp/$recog/decode_mbr_$decode_suff
      # cp exp/$recog/decode_$decode_suff/lat.*.gz exp/$recog/decode_mbr_$decode_suff
      # local/score_mbr.sh --cmd "$decode_cmd" \
        # $dataset data/lang_test_$lm/ exp/$recog/decode_mbr_$decode_suff &
    # done
    # wait

  # done # loop recog
# fi

# # decoding using various recognizers with adaptation
# if [ $stage -le 11 ]; then
  # # put tri2b last since it takes longest due to the large mismatch.
  # for recog in tri2b_mc tri2b_mc_mmi_b0.1 tri2b; do
    # # The graph from the ML directory is used in recipe
    # recog2=`echo $recog | sed s/_mmi.*//`
    # graph=exp/$recog2/graph_$lm

    # # set the adaptation data
    # if [[ "$recog" =~ _mc ]]; then
      # tr_dataset=REVERB_tr_cut/SimData_tr_for_1ch_A
    # else
      # tr_dataset=si_tr
    # fi

    # echo "### DECODING with $recog, basis_fmllr, $lm ###"
    # steps/get_fmllr_basis.sh --cmd "$train_cmd" --per-utt true data/$tr_dataset data/lang exp/$recog
    # for dataset in data/REVERB_*{dt,et}/*; do
      # (
        # decode_suff=${lm}_`echo $dataset | awk -F '/' '{print $2 "_" $3}'`
        # steps/decode_basis_fmllr.sh --nj $nj_decode --cmd "$decode_cmd" \
          # $graph $dataset \
          # exp/$recog/decode_basis_fmllr_$decode_suff
      # ) &
    # done
    # wait

    # echo " ## MBR RESCORING with $recog, basis_fmllr ##"
    # for dataset in data/REVERB_*{dt,et}/*; do
      # decode_suff=${lm}_`echo $dataset | awk -F '/' '{print $2 "_" $3}'`
      # mkdir -p exp/$recog/decode_mbr_basis_fmllr_$decode_suff
      # cp exp/$recog/decode_basis_fmllr_$decode_suff/lat.*.gz exp/$recog/decode_mbr_basis_fmllr_$decode_suff
      # local/score_mbr.sh --cmd "$decode_cmd" \
        # $dataset data/lang_test_$lm/ exp/$recog/decode_mbr_basis_fmllr_$decode_suff &
    # done
    # wait

  # done # loop recog
# fi

# get all WERs with lmw=15
if [ $stage -le 12 ]; then
  local/get_results.sh
fi
