#! /bin/bash

# Copyright Johns Hopkins University
#   2019 Fei Wu

set -eo

stage=0
cmu_kids=               # path to cmu_kids corpus
cslu_kids=              # path to cslu_kids corpus
lm_src=                 # path of existing librispeech lm 
extra_features=false    # Extra features for GMM model (MMI, boosting and MPE)
vtln=false              # Optional, run VLTN on gmm and tdnnf models if set true 
email=                  # Reporting email for tdnn-f training

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

lm_url=www.openslr.org/resources/11
mkdir -p data
mkdir -p data/local

# Prepare data
if [ $stage -le 0 ]; then
  # Make soft link to the corpora
  if [ ! -e cmu_kids ]; then
    if [ ! -d $cmu_kids/kids ]; then echo "ERROR: Expected to find a directory called 'kids' in $cmu_kids. Exiting." && exit 1; fi
    ln -sf $cmu_kids cmu_kids
  fi
  if [ ! -e cslu ]; then
    if [ ! -d $cslu_kids/speech ]; then echo "ERROR: Expected to find a directory called 'speech' in $cslu_kids. Exiting." && exit 1; fi
    ln -sf $cslu_kids cslu
  fi
  
  # Make softlink to lm, if lm_src provided
  if [ ! -z "$lm_src" ] && [ ! -e data/local/lm ] ; then
    ln -sf $lm_src data/local/lm
  fi
  
  # Remove old data dirs
  rm -rf data/data_cmu
  rm -rf data/data_cslu

  # Data Prep
  ./local/cmu_prepare_data.sh --corpus cmu_kids/kids --data data/data_cmu
  ./local/cslu_prepare_data.sh --corpus cslu --data data/data_cslu 
fi

# Combine data
if [ $stage -le 1 ]; then
   mkdir -p data/train
   mkdir -p data/test
   rm -rf data/train/*
   rm -rf data/test/*
   ./utils/combine_data.sh data/train data/data_cmu/train data/data_cslu/train
   ./utils/combine_data.sh data/test data/data_cmu/test data/data_cslu/test
fi

# LM, WFST Preparation
if [ $stage -le 2 ]; then
  if [ ! -d data/local/dict ]; then
      ./local/download_cmu_dict.sh
  fi

  if [ ! -e data/local/lm ]; then
    echo "lm_src not provided. Downloading lm from openslr."
    ./local/download_lm.sh $lm_url data/local/lm
  fi

  utils/prepare_lang.sh data/local/dict "<UNK>"  data/local/lang data/lang
  local/format_lms.sh --src_dir data/lang  data/local/lm 
   
  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge
  utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz data/lang data/lang_test_fglarge 
fi

# Make MFCC features
if [ $stage -le 3 ]; then
  mkdir -p mfcc
  mkdir -p exp
  steps/make_mfcc.sh --nj 40 --cmd "$train_cmd" data/test exp/make_feat/test mfcc
  steps/compute_cmvn_stats.sh data/test exp/make_feat/test mfcc
  steps/make_mfcc.sh --nj 40 --cmd "$train_cmd" data/train exp/make_feat/train mfcc 
  steps/compute_cmvn_stats.sh data/train exp/make_feat/train mfcc
fi

# Mono-phone 
if [ $stage -le 4 ]; then
  # Train
  steps/train_mono.sh --nj 40 --cmd "$train_cmd" data/train data/lang exp/mono 
  #Decode
  utils/mkgraph.sh data/lang_test_tgsmall exp/mono exp/mono/graph
  steps/decode.sh --config conf/decode.config --nj 40 --cmd "$decode_cmd" exp/mono/graph data/test exp/mono/decode
  #Align
  steps/align_si.sh --nj 20 --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_ali
fi

# Tri1 [Vanilla tri phone model]
if [ $stage -le 5 ]; then
  # Train
  steps/train_deltas.sh --cmd "$train_cmd" 1800 9000 data/train data/lang exp/mono_ali exp/tri1
  # Decode 
  utils/mkgraph.sh data/lang_test_tgmed exp/tri1 exp/tri1/graph 
  steps/decode.sh --config conf/decode.config --nj 40 --cmd "$decode_cmd" exp/tri1/graph data/test exp/tri1/decode
  # Align - make graph - decode again   
  steps/align_si.sh --nj 20 --cmd "$train_cmd" --use-graphs true data/train data/lang_test_tgmed exp/tri1 exp/tri1_ali
  utils/mkgraph.sh data/lang_test_tgmed exp/tri1_ali exp/tri1_ali/graph
  steps/decode.sh --config conf/decode.config --nj 40 --cmd "$decode_cmd" exp/tri1_ali/graph data/test exp/tri1_ali/decode
fi

# Add LDA and MLLT
if [ $stage -le 6 ]; then
  # Train
  steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3" 1800 9000 data/train data/lang exp/tri1_ali exp/tri2
  utils/mkgraph.sh data/lang_test_tgmed exp/tri2 exp/tri2/graph
  # Decode
  steps/decode.sh --config conf/decode.config --nj 40 --cmd "$decode_cmd" exp/tri2/graph data/test exp/tri2/decode
  # Align - make graph - dcode again 
  steps/align_si.sh --nj 20 --cmd "$train_cmd" --use-graphs true data/train data/lang_test_tgmed exp/tri2 exp/tri2_ali
  utils/mkgraph.sh data/lang_test_tgmed exp/tri2_ali exp/tri2_ali/graph
  steps/decode_fmllr.sh --config conf/decode.config --nj 40 --cmd "$decode_cmd" exp/tri2_ali/graph data/test exp/tri2_ali/decode
fi 

# Add other features
if [ $stage -le 7 ]; then
  if [ $extra_features = true ]; then
    # Add MMI
    steps/make_denlats.sh --nj 20 --cmd "$train_cmd" data/train data/lang exp/tri2 exp/tri2_denlats
    steps/train_mmi.sh data/train data/lang exp/tri2_ali exp/tri2_denlats exp/tri2_mmi
    steps/decode.sh --config conf/decode.config --iter 4 --nj 20 --cmd "$decode_cmd" exp/tri2/graph data/test exp/tri2_mmi/decode_it4
    steps/decode.sh --config conf/decode.config --iter 3 --nj 20 --cmd "$decode_cmd" exp/tri2/graph data/test exp/tri2_mmi/decode_it3
    
    # Add Boosting 
    steps/train_mmi.sh --boost 0.05 data/train data/lang exp/tri2_ali exp/tri2_denlats exp/tri2_mmi_b0.05
    steps/decode.sh --config conf/decode.config --iter 4 --nj 20 --cmd "$decode_cmd" exp/tri2/graph data/test exp/tri2_mmi_b0.05/decode_it4
    steps/decode.sh --config conf/decode.config --iter 3 --nj 20 --cmd "$decode_cmd" exp/tri2/graph data/test exp/tri2_mmi_b0.05/decode_it3
    
    # Add MPE 
    steps/train_mpe.sh data/train data/lang exp/tri2_ali exp/tri2_denlats exp/tri2_mpe
    steps/decode.sh --config conf/decode.config --iter 4 --nj 20 --cmd "$decode_cmd" exp/tri2/graph data/test exp/tri2_mpe/decode_it4
    steps/decode.sh --config conf/decode.config --iter 3 --nj 20 --cmd "$decode_cmd" exp/tri2/graph data/test exp/tri2_mpe/decode_it3
  fi
fi

# Add SAT
if [ $stage -le 8 ]; then 
  # Do LDA+MLLT+SAT, and decode.
  steps/train_sat.sh 1800 9000 data/train data/lang exp/tri2_ali exp/tri3
  utils/mkgraph.sh data/lang_test_tgmed exp/tri3 exp/tri3/graph
  steps/decode_fmllr.sh --config conf/decode.config --nj 40 --cmd "$decode_cmd" exp/tri3/graph data/test exp/tri3/decode
fi

if [ $stage -le 9 ]; then
  # Align all data with LDA+MLLT+SAT system (tri3)
  steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" --use-graphs true data/train data/lang_test_tgmed exp/tri3 exp/tri3_ali
  utils/mkgraph.sh data/lang_test_tgmed exp/tri3_ali exp/tri3_ali/graph   
  steps/decode_fmllr.sh --config conf/decode.config --nj 40 --cmd "$decode_cmd" exp/tri3_ali/graph data/test exp/tri3_ali/decode
fi

if [ $stage -le 10 ]; then 
    # Uncomment reporting email option to get training progress updates by email
  ./local/chain/run_tdnnf.sh --train_set train \
      --test_sets test --gmm tri3  # --reporting_email $email 
fi


# Optional VTLN. Run if vtln is set to true
if [ $stage -le 11 ]; then
  if [ $vtln = true ]; then
    ./local/vtln.sh
    ./local/chain/run_tdnnf.sh --nnet3_affix vtln --train_set train_vtln \
        --test_sets test_vtln --gmm tri5 # --reporting_email $email
  fi
fi

# Collect and resport WER results for all models
./local/sort_result.sh
