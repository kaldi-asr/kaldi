#!/usr/bin/env bash

# Copyright 2020  Johns Hopkins University (author: Piotr Żelasko)
# Apache 2.0

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
data=/export/corpora5/nsc

# base url for downloads.
mfccdir=mfcc
stage=2

. ./cmd.sh
. ./path.sh
. parse_options.sh

# Bash strict mode; you might not want to do this for interactive shells.
set -eou pipefail


if [ $stage -le 1 ]; then
  # Creates data/nsc and splits it into data/nsc/train and data/nsc/test
  local/nsc_data_prep.sh
fi

if [ $stage -le 2 ]; then
  # spread the mfccs over various machines, as this data-set is quite large.
  if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
    mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
    utils/create_split_dir.pl /export/c{23,24,25,27}/$USER/kaldi-data/egs/nsc/s5/$mfcc/storage \
     $mfccdir/storage
  fi
fi


if [ $stage -le 3 ]; then
  for part in train test; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 60 data/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
  done
fi

if [ $stage -le 4 ]; then
  # Make some small data subsets for early system-build stages.
  # For the monophone stages we select the shortest utterances, which should make it
  # easier to align the data from a flat start.

  utils/subset_data_dir.sh --shortest data/train 2000 data/train_2kshort
  utils/subset_data_dir.sh data/train 5000 data/train_5k
  utils/subset_data_dir.sh data/train 10000 data/train_10k
  utils/subset_data_dir.sh data/train 30000 data/train_30k
  utils/subset_data_dir.sh data/train 100000 data/train_100k
fi

if [ $stage -le 5 ]; then
  # train a monophone system
  steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
                      data/train_2kshort data/lang exp/mono
fi

if [ $stage -le 6 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
                    data/train_5k data/lang exp/mono exp/mono_ali_5k

  # train a first delta + delta-delta triphone system on a subset of 5000 utterances
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
                        2000 10000 data/train_5k data/lang exp/mono_ali_5k exp/tri1
fi

if [ $stage -le 7 ]; then
  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
                    data/train_10k data/lang exp/tri1 exp/tri1_ali_10k


  # train an LDA+MLLT system.
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
                          --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
                          data/train_10k data/lang exp/tri1_ali_10k exp/tri2b
fi

if [ $stage -le 8 ]; then
  # Align a 10k utts subset using the tri2b model
  steps/align_si.sh  --nj 10 --cmd "$train_cmd" --use-graphs true \
                     data/train_10k data/lang exp/tri2b exp/tri2b_ali_10k

  # Train tri3b, which is LDA+MLLT+SAT on 10k utts
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
                     data/train_10k data/lang exp/tri2b_ali_10k exp/tri3b

fi

if [ $stage -le 9 ]; then
  # align the entire train_clean_100 subset using the tri3b model
  steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
    data/train_30k data/lang \
    exp/tri3b exp/tri3b_ali_30k

  # train another LDA+MLLT+SAT system on the entire 100 hour subset
  steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
                      data/train_30k data/lang \
                      exp/tri3b_ali_30k exp/tri4b
fi

if [ $stage -le 13 ]; then
  echo "skipped"
  # Now we compute the pronunciation and silence probabilities from training data,
  # and re-create the lang directory.
  #steps/get_prons.sh --cmd "$train_cmd" \
  #                   data/train_30k data/lang exp/tri4b
  #utils/dict_dir_add_pronprobs.sh --max-normalize true \
  #                                data/local/dict \
  #                                exp/tri4b/pron_counts_nowb.txt exp/tri4b/sil_counts_nowb.txt \
  #                                exp/tri4b/pron_bigram_counts_nowb.txt data/local/dict

  #utils/prepare_lang.sh data/local/dict \
  #                      "<UNK>" data/local/lang_tmp data/lang
  #local/format_lms.sh --src-dir data/lang data/local/lm

  #utils/build_const_arpa_lm.sh \
  #  data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge
  #utils/build_const_arpa_lm.sh \
  #  data/local/lm/lm_fglarge.arpa.gz data/lang data/lang_test_fglarge
fi

if [ $stage -le 16 ]; then
  # align the new, combined set, using the tri4b model
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
                       data/train_100k data/lang exp/tri4b exp/tri4b_ali_100k

  # create a larger SAT model, trained on the 460 hours of data.
  steps/train_sat.sh  --cmd "$train_cmd" 5000 100000 \
                      data/train_100k data/lang exp/tri4b_ali_100k exp/tri5b
fi


# The following command trains an nnet3 model on the 460 hour setup.  This
# is deprecated now.
## train a NN model on the 460 hour set
#local/nnet2/run_6a_clean_460.sh

if [ $stage -le 18 ]; then
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
                       data/train data/lang exp/tri5b exp/tri5b_ali

  # train a SAT model on the 960 hour mixed data.  Use the train_quick.sh script
  # as it is faster.
  steps/train_quick.sh --cmd "$train_cmd" \
                       7000 150000 data/train data/lang exp/tri5b_ali exp/tri6b
fi

if [ $stage -le 19 ]; then
  # decode using the tri6b model
  utils/mkgraph.sh data/lang_test_tg \
                   exp/tri6b exp/tri6b/graph_tg
  for test in test; do
      steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
                            exp/tri6b/graph_tg data/$test exp/tri6b/decode_tg_$test
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_test_{tg,fg} \
        data/$test exp/tri6b/decode_{tg,fg}_$test
  done
fi


if [ $stage -le 20 ]; then
  # this does some data-cleaning. The cleaned data should be useful when we add
  # the neural net and chain systems.  (although actually it was pretty clean already.)
  echo "Cleanup segmentation skipped for now due to encoding/non-ascii characters issues"
  #local/run_cleanup_segmentation.sh
fi

# steps/cleanup/debug_lexicon.sh --remove-stress true  --nj 200 --cmd "$train_cmd" data/train_clean_100 \
#    data/lang exp/tri6b data/local/dict/lexicon.txt exp/debug_lexicon_100h

# #Perform rescoring of tri6b be means of faster-rnnlm
# #Attention: with default settings requires 4 GB of memory per rescoring job, so commenting this out by default
# wait && local/run_rnnlm.sh \
#     --rnnlm-ver "faster-rnnlm" \
#     --rnnlm-options "-hidden 150 -direct 1000 -direct-order 5" \
#     --rnnlm-tag "h150-me5-1000" $data data/local/lm

# #Perform rescoring of tri6b be means of faster-rnnlm using Noise contrastive estimation
# #Note, that could be extremely slow without CUDA
# #We use smaller direct layer size so that it could be stored in GPU memory (~2Gb)
# #Suprisingly, bottleneck here is validation rather then learning
# #Therefore you can use smaller validation dataset to speed up training
# wait && local/run_rnnlm.sh \
#     --rnnlm-ver "faster-rnnlm" \
#     --rnnlm-options "-hidden 150 -direct 400 -direct-order 3 --nce 20" \
#     --rnnlm-tag "h150-me3-400-nce20" $data data/local/lm


if [ $stage -le 21 ]; then
  # train and test nnet3 tdnn models on the entire data with data-cleaning.
  local/chain/run_tdnn.sh # set "--stage 11" if you have already run local/nnet3/run_tdnn.sh
fi

# The nnet3 TDNN recipe:
# local/nnet3/run_tdnn.sh # set "--stage 11" if you have already run local/chain/run_tdnn.sh

# # train models on cleaned-up data
# # we've found that this isn't helpful-- see the comments in local/run_data_cleaning.sh
# local/run_data_cleaning.sh

# # The following is the current online-nnet2 recipe, with "multi-splice".
# local/online/run_nnet2_ms.sh

# # The following is the discriminative-training continuation of the above.
# local/online/run_nnet2_ms_disc.sh

# ## The following is an older version of the online-nnet2 recipe, without "multi-splice".  It's faster
# ## to train but slightly worse.
# # local/online/run_nnet2.sh
