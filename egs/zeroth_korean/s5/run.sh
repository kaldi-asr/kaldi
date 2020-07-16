#!/usr/bin/env bash
#
# Based mostly on the WSJ/Librispeech recipe. 
# The training/testing database is described in http://www.openslr.org/40/
# This corpus consists of 51hrs korean speech with cleaned automatic transcripts:
#
# Copyright  2018  Atlas Guide (Author : Lucas Jo)
#            2018  Gridspace Inc. (Author: Wonkyum Lee)
#
# Apache 2.0
#

# Check list before start
# 1. required software: Morfessor-2.0.1 (see tools/extras/install_morfessor.sh)

stage=0
db_dir=./db
nj=16

chain_train=true
decode=true # set false if you don't want to decode each GMM model
decode_rescoring=true # set false if you don't want to rescore with large language model
test_set="test_clean"

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

# you might not want to do this for interactive shells.
set -e

if [ $stage -le 0 ]; then
  # download the data.  
  local/download_and_untar.sh $db_dir
fi

if [ $stage -le 1 ]; then
  # format the data as Kaldi data directories
  for part in train_data_01 test_data_01; do
  	# use underscore-separated names in data directories.
  	local/data_prep.sh $db_dir $part
  done
fi

if [ $stage -le 2 ]; then
  # update segmentation of transcripts
  for part in train_data_01 test_data_01; do
  	local/update_segmentation.sh data/$part data/local/lm
  done
fi

if [ $stage -le 3 ]; then
  # prepare dictionary and language model 
  local/prepare_dict.sh data/local/lm data/local/dict_nosp
  
  utils/prepare_lang.sh data/local/dict_nosp \
  	"<UNK>" data/local/lang_tmp_nosp data/lang_nosp
fi

if [ $stage -le 4 ]; then
  # build testing language model
  local/format_lms.sh --src-dir data/lang_nosp data/local/lm

  # re-scoring language model
  if $decode_rescoring ; then
    utils/build_const_arpa_lm.sh data/local/lm/zeroth.lm.tg.arpa.gz \
    	data/lang_nosp data/lang_nosp_test_tglarge
    utils/build_const_arpa_lm.sh data/local/lm/zeroth.lm.fg.arpa.gz \
    	  data/lang_nosp data/lang_nosp_test_fglarge
  fi
fi


if [ $stage -le 5 ]; then
  # Feature extraction (MFCC)
  mfccdir=mfcc
  for part in train_data_01 test_data_01; do
  	steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/$part exp/make_mfcc/$part $mfccdir
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
fi

if [ $stage -le 6 ]; then
  echo "$0: #### Monophone Training ###########"
  # train a monophone system with 2k short utts
  steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
  	data/train_2kshort data/lang_nosp exp/mono
  if $decode; then
    utils/mkgraph.sh data/lang_nosp_test_tgsmall exp/mono exp/mono/graph_nosp_tgsmall
    nspk=$(wc -l <data/${test_set}/spk2utt)
    steps/decode.sh --nj $nspk --cmd "$decode_cmd" \
      exp/mono/graph_nosp_tgsmall data/${test_set} exp/mono/decode_nosp_tgsmall_${test_set}
    if $decode_rescoring; then
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
        data/$test_set exp/mono/decode_nosp_{tgsmall,tglarge}_$test_set
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,fglarge} \
        data/$test_set exp/mono/decode_nosp_{tgsmall,fglarge}_$test_set
    fi 
  fi
fi

if [ $stage -le 7 ]; then
  echo "$0: #### Triphone Training, delta + delta-delta ###########"
  steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
  	data/train_5k data/lang_nosp exp/mono exp/mono_ali_5k
  # train a first delta + delta-delta triphone system on a subset of 5000 utterances
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
      2000 10000 data/train_5k data/lang_nosp exp/mono_ali_5k exp/tri1
  if $decode; then
    utils/mkgraph.sh data/lang_nosp_test_tgsmall exp/tri1 exp/tri1/graph_nosp_tgsmall
    nspk=$(wc -l <data/${test_set}/spk2utt)
    steps/decode.sh --nj $nspk --cmd "$decode_cmd" \
      exp/tri1/graph_nosp_tgsmall data/${test_set} exp/tri1/decode_nosp_tgsmall_${test_set}
    if $decode_rescoring; then
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
        data/$test_set exp/tri1/decode_nosp_{tgsmall,tglarge}_$test_set
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,fglarge} \
        data/$test_set exp/tri1/decode_nosp_{tgsmall,fglarge}_$test_set
    fi
  fi
fi

if [ $stage -le 8 ]; then
  echo "$0: #### Triphone Training, LDA+MLLT ###########"
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train_10k data/lang_nosp exp/tri1 exp/tri1_ali_10k
  # train an LDA+MLLT system.
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
     --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
     data/train_10k data/lang_nosp exp/tri1_ali_10k exp/tri2
  if $decode; then
    utils/mkgraph.sh data/lang_nosp_test_tgsmall exp/tri2 exp/tri2/graph_nosp_tgsmall
    nspk=$(wc -l <data/${test_set}/spk2utt)
    steps/decode.sh --nj $nspk --cmd "$decode_cmd" \
      exp/tri2/graph_nosp_tgsmall data/${test_set} exp/tri2/decode_nosp_tgsmall_${test_set}
    if $decode_rescoring; then
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
        data/$test_set exp/tri2/decode_nosp_{tgsmall,tglarge}_$test_set
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,fglarge} \
        data/$test_set exp/tri2/decode_nosp_{tgsmall,fglarge}_$test_set
    fi
  fi
fi


if [ $stage -le 9 ]; then
  echo "$0: #### Triphone Training, LDA+MLLT+SAT ###########"
  # Align the entire train_clean using the tri2 model
  steps/align_si.sh  --nj $nj --cmd "$train_cmd" --use-graphs true \
    data/train_clean data/lang_nosp exp/tri2 exp/tri2_ali_train_clean
  
  # Train tri3, which is LDA+MLLT+SAT on the entire train_clean
  steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
    data/train_clean data/lang_nosp exp/tri2_ali_train_clean exp/tri3
  if $decode; then
    utils/mkgraph.sh data/lang_nosp_test_tgsmall exp/tri3 exp/tri3/graph_nosp_tgsmall
    nspk=$(wc -l <data/${test_set}/spk2utt)
    steps/decode_fmllr.sh --nj $nspk --cmd "$decode_cmd" \
      exp/tri3/graph_nosp_tgsmall data/${test_set} exp/tri3/decode_nosp_tgsmall_${test_set}
    if $decode_rescoring; then
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
        data/$test_set exp/tri3/decode_nosp_{tgsmall,tglarge}_$test_set
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,fglarge} \
        data/$test_set exp/tri3/decode_nosp_{tgsmall,fglarge}_$test_set
    fi
  fi
fi 

if [ $stage -le 10 ]; then
  echo "$0: #### Re-computing pronunciation model using tri3 model ###########"
  # Now we compute the pronunciation and silence probabilities from training data,
  # and re-create the lang directory.
  # silence transition probability ...
  steps/get_prons.sh --cmd "$train_cmd" \
        data/train_clean data/lang_nosp exp/tri3
  
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
        data/local/dict_nosp \
          exp/tri3/pron_counts_nowb.txt exp/tri3/sil_counts_nowb.txt \
            exp/tri3/pron_bigram_counts_nowb.txt data/local/dict
  
  utils/prepare_lang.sh data/local/dict \
        "<UNK>" data/local/lang_tmp data/lang
  
  local/format_lms.sh --src-dir data/lang data/local/lm
  
  utils/build_const_arpa_lm.sh \
        data/local/lm/zeroth.lm.tg.arpa.gz data/lang data/lang_test_tglarge
  utils/build_const_arpa_lm.sh \
        data/local/lm/zeroth.lm.fg.arpa.gz data/lang data/lang_test_fglarge

  if $decode; then
    utils/mkgraph.sh data/lang_test_tgsmall exp/tri3 exp/tri3/graph_tgsmall
    nspk=$(wc -l <data/${test_set}/spk2utt)
    steps/decode_fmllr.sh --nj $nspk --cmd "$decode_cmd" \
      exp/tri3/graph_tgsmall data/${test_set} exp/tri3/decode_tgsmall_${test_set}
    if $decode_rescoring; then
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
        data/$test_set exp/tri3/decode_{tgsmall,tglarge}_$test_set
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
        data/$test_set exp/tri3/decode_{tgsmall,fglarge}_$test_set
    fi
  fi
fi

if [ $stage -le 11 ]; then

  echo "$0: #### SAT again on train_clean ###########"
  # align the entire train_clean using the tri3 model
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train_clean data/lang exp/tri3 exp/tri3_ali_train_clean
  
  # train another LDA+MLLT+SAT system on the entire train_clean
  steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
    data/train_clean data/lang exp/tri3_ali_train_clean exp/tri4
 
  if $decode; then
    utils/mkgraph.sh data/lang_test_tgsmall exp/tri4 exp/tri4/graph_tgsmall
    nspk=$(wc -l <data/${test_set}/spk2utt)
    steps/decode_fmllr.sh --nj $nspk --cmd "$decode_cmd" \
      exp/tri4/graph_tgsmall data/${test_set} exp/tri4/decode_tgsmall_${test_set}
    if $decode_rescoring; then
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
        data/$test_set exp/tri4/decode_{tgsmall,tglarge}_$test_set
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
        data/$test_set exp/tri4/decode_{tgsmall,fglarge}_$test_set
    fi
  fi 
fi 

echo "$0: GMM trainig is Done"

if $chain_train; then
  ## Training Chain Acoustic model using clean data set
  echo "$0: #### chain training  ###########"
  local/chain/run_tdnn.sh
fi 

exit 0;

