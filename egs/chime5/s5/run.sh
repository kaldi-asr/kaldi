#!/usr/bin/env bash
#
# Based mostly on the TED-LIUM and Switchboard recipe
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe and Yenda Trmal)
# Apache 2.0
#

# Begin configuration section.
nj=96
decode_nj=20
stage=0
enhancement=beamformit # for a new enhancement method,
                       # change this variable and stage 4
# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh


set -e # exit on error

# chime5 main directory path
# please change the path accordingly
chime5_corpus=/export/corpora5/CHiME5
json_dir=${chime5_corpus}/transcriptions
audio_dir=${chime5_corpus}/audio

# training and test data
train_set=train_worn_u100k
test_sets="dev_worn dev_${enhancement}_ref eval_${enhancement}_ref"

# This script also needs the phonetisaurus g2p, srilm, beamformit
./local/check_tools.sh || exit 1

if [ $stage -le 1 ]; then
  # skip u03 as they are missing
  for mictype in worn u01 u02 u04 u05 u06; do
    local/prepare_data.sh --mictype ${mictype} \
			  ${audio_dir}/train ${json_dir}/train data/train_${mictype}
  done
  for dataset in dev; do
    for mictype in worn; do
      local/prepare_data.sh --mictype ${mictype} \
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

if [ $stage -le 4 ]; then
  # Beamforming using reference arrays
  # enhanced WAV directory
  enhandir=enhan
  for dset in dev eval; do
    for mictype in u01 u02 u03 u04 u05 u06; do
      local/run_beamformit.sh --cmd "$train_cmd" \
			      ${audio_dir}/${dset} \
			      ${enhandir}/${dset}_${enhancement}_${mictype} \
			      ${mictype}
    done
  done

  for dset in dev eval; do
    local/prepare_data.sh --mictype ref "$PWD/${enhandir}/${dset}_${enhancement}_u0*" \
			  ${json_dir}/${dset} data/${dset}_${enhancement}_ref
  done
fi

if [ $stage -le 5 ]; then
  # remove possibly bad sessions (P11_S03, P52_S19, P53_S24, P54_S24)
  # see http://spandh.dcs.shef.ac.uk/chime_challenge/data.html for more details
  utils/copy_data_dir.sh data/train_worn data/train_worn_org # back up
  grep -v -e "^P11_S03" -e "^P52_S19" -e "^P53_S24" -e "^P54_S24" data/train_worn_org/text > data/train_worn/text
  utils/fix_data_dir.sh data/train_worn

  # combine mix array and worn mics
  # randomly extract first 100k utterances from all mics
  # if you want to include more training data, you can increase the number of array mic utterances
  utils/combine_data.sh data/train_uall data/train_u01 data/train_u02 data/train_u04 data/train_u05 data/train_u06
  utils/subset_data_dir.sh data/train_uall 100000 data/train_u100k
  utils/combine_data.sh data/${train_set} data/train_worn data/train_u100k

  # only use left channel for worn mic recognition
  # you can use both left and right channels for training
  for dset in train dev; do
    utils/copy_data_dir.sh data/${dset}_worn data/${dset}_worn_stereo
    grep "\.L-" data/${dset}_worn_stereo/text > data/${dset}_worn/text
    utils/fix_data_dir.sh data/${dset}_worn
  done
fi

if [ $stage -le 6 ]; then
  # fix speaker ID issue (thanks to Dr. Naoyuki Kanda)
  # add array ID to the speaker ID to avoid the use of other array information to meet regulations
  # Before this fix
  # $ head -n 2 data/eval_beamformit_ref_nosplit/utt2spk
  # P01_S01_U02_KITCHEN.ENH-0000192-0001278 P01
  # P01_S01_U02_KITCHEN.ENH-0001421-0001481 P01
  # After this fix
  # $ head -n 2 data/eval_beamformit_ref_nosplit_fix/utt2spk
  # P01_S01_U02_KITCHEN.ENH-0000192-0001278 P01_U02
  # P01_S01_U02_KITCHEN.ENH-0001421-0001481 P01_U02
  for dset in dev_${enhancement}_ref eval_${enhancement}_ref; do
    utils/copy_data_dir.sh data/${dset} data/${dset}_nosplit
    mkdir -p data/${dset}_nosplit_fix
    cp data/${dset}_nosplit/{segments,text,wav.scp} data/${dset}_nosplit_fix/
    awk -F "_" '{print $0 "_" $3}' data/${dset}_nosplit/utt2spk > data/${dset}_nosplit_fix/utt2spk
    utils/utt2spk_to_spk2utt.pl data/${dset}_nosplit_fix/utt2spk > data/${dset}_nosplit_fix/spk2utt
  done

  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  for dset in ${train_set} dev_worn; do
    utils/copy_data_dir.sh data/${dset} data/${dset}_nosplit
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}_nosplit data/${dset}
  done
  for dset in dev_${enhancement}_ref eval_${enhancement}_ref; do
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}_nosplit_fix data/${dset}
  done
fi

if [ $stage -le 7 ]; then
  # Now make MFCC features.
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  mfccdir=mfcc
  for x in ${train_set} ${test_sets}; do
    steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" \
		       data/$x exp/make_mfcc/$x $mfccdir
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
    utils/fix_data_dir.sh data/$x
  done
fi

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
  utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph
  for dset in ${test_sets}; do
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
		    exp/tri2/graph data/${dset} exp/tri2/decode_${dset} &
  done
  wait
fi

if [ $stage -le 14 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
		    data/${train_set} data/lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd "$train_cmd" \
		     5000 100000 data/${train_set} data/lang exp/tri2_ali exp/tri3
fi

if [ $stage -le 15 ]; then
  utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph
  for dset in ${test_sets}; do
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
			  exp/tri3/graph data/${dset} exp/tri3/decode_${dset} &
  done
  wait
fi

if [ $stage -le 16 ]; then
  # The following script cleans the data and produces cleaned data
  steps/cleanup/clean_and_segment_data.sh --nj ${nj} --cmd "$train_cmd" \
    --segmentation-opts "--min-segment-length 0.3 --min-new-segment-length 0.6" \
    data/${train_set} data/lang exp/tri3 exp/tri3_cleaned data/${train_set}_cleaned
fi

if [ $stage -le 17 ]; then
  # chain TDNN
  local/chain/run_tdnn.sh --nj ${nj} --train-set ${train_set}_cleaned --test-sets "$test_sets" --gmm tri3_cleaned --nnet3-affix _${train_set}_cleaned
fi

if [ $stage -le 18 ]; then
  # final scoring to get the official challenge result
  # please specify both dev and eval set directories so that the search parameters
  # (insertion penalty and language model weight) will be tuned using the dev set
  local/score_for_submit.sh \
      --dev exp/chain_${train_set}_cleaned/tdnn1a_sp/decode_dev_${enhancement}_ref \
      --eval exp/chain_${train_set}_cleaned/tdnn1a_sp/decode_eval_${enhancement}_ref
fi
