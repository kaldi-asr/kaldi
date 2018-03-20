#!/bin/bash
#
# Based mostly on the WSJ/Librispeech recipe. The training database is #####,
# it consists of 51hrs korean speech with cleaned automatic transcripts:
#
# http://www.openslr.org/resources (Mirror).
#
# Copyright  2018  Atlas Guide (Author : Lucas Jo)
#            2018  Gridspace Inc. (Author: Wonkyum Lee)
#
# Apache 2.0
#

# Check list before start
# 1. locale setup
# 2. pre-installed package: awscli, Morfessor-2.0.1, flac, sox, same cuda library, unzip
# 3. pre-install or symbolic link for easy going: rirs_noises.zip (takes pretty long time)
# 4. parameters: nCPU, num_jobs_initial, num_jobs_final, --max-noises-per-minute

db_dir=./db
nCPU=16

. ./cmd.sh
. ./path.sh

# you might not want to do this for interactive shells.
set -e

startTime=$(date +'%F-%H-%M')
echo "started at" $startTime

# download the data.  
local/download_and_untar.sh $db_dir

# format the data as Kaldi data directories
for part in train_data_01 test_data_01; do
	# use underscore-separated names in data directories.
	local/data_prep.sh $db_dir/$part data/$(echo $part | sed s/-/_/g)
done

# update segmentation of transcripts
for part in train_data_01 test_data_01; do
	local/updateSegmentation.sh data/$part data/local/lm
done

# prepare dictionary and language model 
local/prepare_dict.sh data/local/lm data/local/dict_nosp

utils/prepare_lang.sh data/local/dict_nosp \
	"<UNK>" data/local/lang_tmp_nosp data/lang_nosp

local/format_lms.sh --src-dir data/lang_nosp data/local/lm

# Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
# it takes long time and do this again after computing silence prob.
# you can do comment out here this time

#utils/build_const_arpa_lm.sh data/local/lm/zeroth.lm.tg.arpa.gz \
#	data/lang_nosp data/lang_nosp_test_tglarge
#utils/build_const_arpa_lm.sh data/local/lm/zeroth.lm.fg.arpa.gz \
#	  data/lang_nosp data/lang_nosp_test_fglarge

# Feature extraction (MFCC)
mfccdir=mfcc
hostInAtlas="ares hephaestus jupiter neptune"
if [[ ! -z $(echo $hostInAtlas | grep -o $(hostname -f)) ]]; then
  mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
  utils/create_split_dir.pl /mnt/{ares,hephaestus,jupiter,neptune}/$USER/kaldi-data/zeroth-kaldi/s5/$mfcc/storage \
    $mfccdir/storage
fi
for part in train_data_01 test_data_01; do
	steps/make_mfcc.sh --cmd "$train_cmd" --nj $nCPU data/$part exp/make_mfcc/$part $mfccdir
	steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
done

# ... and then combine data sets into one (for later extension)
utils/combine_data.sh \
  data/train_clean data/train_data_01

utils/combine_data.sh \
  data/test_clean data/test_data_01

# Make some small data subsets for early system-build stages.
utils/subset_data_dir.sh --shortest data/train_clean 2000 data/train_2kshort
utils/subset_data_dir.sh data/train_clean 5000 data/train_5k
utils/subset_data_dir.sh data/train_clean 10000 data/train_10k

echo "#### Monophone Training ###########"
# train a monophone system & align
steps/train_mono.sh --boost-silence 1.25 --nj $nCPU --cmd "$train_cmd" \
	data/train_2kshort data/lang_nosp exp/mono
steps/align_si.sh --boost-silence 1.25 --nj $nCPU --cmd "$train_cmd" \
	data/train_5k data/lang_nosp exp/mono exp/mono_ali_5k

echo "#### Triphone Training, delta + delta-delta ###########"
# train a first delta + delta-delta triphone system on a subset of 5000 utterancesa
# number of maximum pdf, gaussian (under/over fitting)
#  recognition result 
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train_5k data/lang_nosp exp/mono_ali_5k exp/tri1
steps/align_si.sh --nj $nCPU --cmd "$train_cmd" \
  data/train_10k data/lang_nosp exp/tri1 exp/tri1_ali_10k

echo "#### Triphone Training, LDA+MLLT ###########"
# train an LDA+MLLT system.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
   data/train_10k data/lang_nosp exp/tri1_ali_10k exp/tri2b

# Align a 10k utts subset using the tri2b model
steps/align_si.sh  --nj $nCPU --cmd "$train_cmd" --use-graphs true \
  data/train_clean data/lang_nosp exp/tri2b exp/tri2b_ali_train_clean

echo "#### Triphone Training, LDA+MLLT+SAT ###########"
# Train tri3b, which is LDA+MLLT+SAT on 10k utts
#steps/train_sat.sh --cmd "$train_cmd" 3000 25000 \
steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
  data/train_clean data/lang_nosp exp/tri2b_ali_train_clean exp/tri3b

# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory.
# silence transition probability ...
steps/get_prons.sh --cmd "$train_cmd" \
      data/train_clean data/lang_nosp exp/tri3b

utils/dict_dir_add_pronprobs.sh --max-normalize true \
      data/local/dict_nosp \
        exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
          exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict

utils/prepare_lang.sh data/local/dict \
      "<UNK>" data/local/lang_tmp data/lang

local/format_lms.sh --src-dir data/lang data/local/lm

utils/build_const_arpa_lm.sh \
      data/local/lm/zeroth.lm.tg.arpa.gz data/lang data/lang_test_tglarge
utils/build_const_arpa_lm.sh \
      data/local/lm/zeroth.lm.fg.arpa.gz data/lang data/lang_test_fglarge

# align the entire train_clean using the tri3b model
steps/align_fmllr.sh --nj $nCPU --cmd "$train_cmd" \
  data/train_clean data/lang exp/tri3b exp/tri3b_ali_train_clean

echo "#### SAT again on train_clean ###########"
# train another LDA+MLLT+SAT system on the entire subset
steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
  data/train_clean data/lang exp/tri3b_ali_train_clean exp/tri4b

# decode using the tri4b model with pronunciation and silence probabilities
utils/mkgraph.sh \
  data/lang_test_tgsmall exp/tri4b exp/tri4b/graph_tgsmall

# the size is properly set?
utils/subset_data_dir.sh data/test_clean 200 data/test_200

for test in test_200; do
  nspk=$(wc -l <data/${test}/spk2utt)
  steps/decode_fmllr.sh --nj $nspk --cmd "$decode_cmd" \
    exp/tri4b/graph_tgsmall data/$test \
    exp/tri4b/decode_tgsmall_$test
  #steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
  #  data/$test exp/tri4b/decode_{tgsmall,tgmed}_$test
  steps/lmrescore_const_arpa.sh \
    --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
    data/$test exp/tri4b/decode_{tgsmall,tglarge}_$test
  steps/lmrescore_const_arpa.sh \
    --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
    data/$test exp/tri4b/decode_{tgsmall,fglarge}_$test
done

# align train_clean_100 using the tri4b model
steps/align_fmllr.sh --nj $nCPU --cmd "$train_cmd" \
	  data/train_clean data/lang exp/tri4b exp/tri4b_ali_train_clean

finishTime=$(date +'%F-%H-%M')
echo "GMM trainig is finished at" $finishTime
exit
## online chain recipe using only clean data set
echo "#### online chain training  ###########"
## check point: sudo nvidia-smi --compute-mode=3 if you have multiple GPU's
#local/chain/run_tdnn_1a.sh
#local/chain/run_tdnn_1b.sh
#local/chain/multi_condition/run_tdnn_lstm_1e.sh --nj $nCPU
local/chain/multi_condition/run_tdnn_1n.sh --nj $nCPU 
#local/chain/run_tdnn_opgru_1c.sh --nj $nCPU


finishTime=$(date +'%F-%H-%M')
echo "DNN trainig is finished at" $finishTime
echo "started at" $startTime
echo "finished at" $finishTime
exit 0;

