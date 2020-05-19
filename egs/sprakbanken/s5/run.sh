#!/usr/bin/env bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh # so python3 is on the path if not on the system (we made a link to utils/).a

nj=12
stage=0
. utils/parse_options.sh

if [ $stage -le 0 ]; then
  # Download the corpus and prepare parallel lists of sound files and text files
  # Divide the corpus into train, dev and test sets
  local/sprak_data_prep.sh  || exit 1;
fi

if [ $stage -le 1 ]; then
  # Perform text normalisation, prepare dict folder and LM data transcriptions
  # This setup uses previsously prepared data. eSpeak must be installed and in PATH to use dict_prep.sh
  # local/dict_prep.sh || exit 1;
  local/copy_dict.sh || exit 1;
fi

if [ $stage -le 2 ]; then
  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang_tmp data/lang || exit 1;
fi

if [ $stage -le 3 ]; then
  # Extract mfccs 
  # p was added to the rspecifier (scp,p:$logdir/wav.JOB.scp) in make_mfcc.sh because some 
  # wave files are corrupt 
  # Will return a warning message because of the corrupt audio files, but compute them anyway
  # If this step fails and prints a partial diff, rerun from sprak_data_prep.sh
  for dataset in train test dev; do
    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/$dataset || exit 1;

    # Compute cepstral mean and variance normalisation
    steps/compute_cmvn_stats.sh data/$dataset || exit 1;

    # Repair data set (remove corrupt data points with corrupt audio)
    utils/fix_data_dir.sh data/$dataset || exit 1;

  done
  # Make a subset of the training data with the shortest 120k utterances. 
  utils/subset_data_dir.sh --shortest data/train 120000 data/train_120kshort || exit 1;
fi

if [ $stage -le 4 ]; then
  # Train LM with irstlm
  local/train_irstlm.sh data/local/transcript_lm/transcripts.uniq 3 "tg" data/lang data/local/train3_lm &> data/local/tg.log || exit 1;
  local/train_irstlm.sh data/local/transcript_lm/transcripts.uniq 4 "fg" data/lang data/local/train4_lm &> data/local/fg.log || exit 1;
fi

if [ $stage -le 5 ]; then
  # Train monophone model on short utterances
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
    data/train_120kshort data/lang exp/mono0a || exit 1;
  utils/mkgraph.sh --mono data/lang_test_tg exp/mono0a exp/mono0a/graph_tg || exit 1;
  steps/decode.sh --nj 12 --cmd "$decode_cmd" \
    exp/mono0a/graph_tg data/dev exp/mono0a/decode_tg_dev || exit 1;
fi

if [ $stage -le 6 ]; then
  # Train tri1 (delta+delta-delta)
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/mono0a exp/mono0a_ali || exit 1;
  steps/train_deltas.sh --cmd "$train_cmd" \
    3000 40000 data/train data/lang exp/mono0a_ali exp/tri1 || exit 1;

  # Decode dev set with both LMs
  utils/mkgraph.sh data/lang_test_tg exp/tri1 exp/tri1/graph_tg || exit 1;
  utils/mkgraph.sh data/lang_test_fg exp/tri1 exp/tri1/graph_fg || exit 1; 
  steps/decode.sh --nj 12 --cmd "$decode_cmd" \
    exp/tri1/graph_fg data/dev exp/tri1/decode_fg_dev || exit 1;
  steps/decode.sh --nj 12 --cmd "$decode_cmd" \
    exp/tri1/graph_tg data/dev exp/tri1/decode_tg_dev || exit 1;
fi

if [ $stage -le 7 ]; then
  # Train tri2a (delta + delta-delta)
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri1 exp/tri1_ali || exit 1;
  steps/train_deltas.sh --cmd "$train_cmd" \
    5000 60000 data/train data/lang exp/tri1_ali exp/tri2a || exit 1;
  utils/mkgraph.sh data/lang_test_tg exp/tri2a exp/tri2a/graph_tg || exit 1;
  steps/decode.sh --nj 12 --cmd "$decode_cmd" \
    exp/tri2a/graph_tg data/dev exp/tri2a/decode_tg_dev || exit 1;
fi

if [ $stage -le 8 ]; then
  # Train tri2b (LDA+MLLT)
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri2a exp/tri2a_ali || exit 1;
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=5 --right-context=5" \
    6500 75000 data/train data/lang exp/tri2a_ali exp/tri2b || exit 1;
  utils/mkgraph.sh data/lang_test_tg exp/tri2b exp/tri2b/graph_tg || exit 1;
  steps/decode.sh --nj 12 --cmd "$decode_cmd" \
    exp/tri2b/graph_tg data/dev exp/tri2b/decode_tg_dev || exit 1;
fi

if [ $stage -le 9 ]; then
  # From 2b system, train 3b which is LDA + MLLT + SAT.
  steps/align_si.sh  --nj $nj --cmd "$train_cmd" \
    --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali  || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" \
    7500 100000 data/train data/lang exp/tri2b_ali exp/tri3b || exit 1;

  # Decode dev with 4gram and 3gram LMs
  utils/mkgraph.sh data/lang_test_tg exp/tri3b exp/tri3b/graph_tg || exit 1;
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 12 \
    exp/tri3b/graph_tg data/dev exp/tri3b/decode_tg_dev || exit 1;
  utils/mkgraph.sh data/lang_test_fg exp/tri3b exp/tri3b/graph_fg || exit 1;
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 12 \
    exp/tri3b/graph_fg data/dev exp/tri3b/decode_fg_dev || exit 1;

  # Decode test with 4gram and 3gram LMs
  # there are fewer speaker (n=7) and decoding usually ends up waiting
  # for a single job so we use --num-threads 2 to speed up
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 7 --num-threads 2 \
    exp/tri3b/graph_tg data/test exp/tri3b/decode_tg_test || exit 1;
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 7 --num-threads 2 \
    exp/tri3b/graph_fg data/test exp/tri3b/decode_fg_test || exit 1;
fi

if [ $stage -le 10 ]; then
  # Alignment used to train nnets and sgmms
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri3b exp/tri3b_ali || exit 1;
fi


## Works
#local/sprak_run_nnet_cpu.sh tg dev 

## Works
#local/sprak_run_sgmm2.sh dev


# Run neural network setups based in the TEDLIUM recipe

# Running the nnet3-tdnn setup will train an ivector extractor that
# is used by the subsequent nnet3 and chain systems (why --stage is
# specified)
#local/nnet3/run_tdnn.sh --tdnn-affix "0" --nnet3-affix ""

# nnet3 LSTM
#local/nnet3/run_lstm.sh --stage 13 --affix "0"

# nnet3 bLSTM
#local/nnet3/run_blstm.sh --stage 12



# chain TDNN
# This setup creates a new lang directory that is also used by the
# TDNN-LSTM system
#local/chain/run_tdnn.sh --stage 14

# chain TDNN-LSTM
local/chain/run_tdnn_lstm.sh --stage 17


# Getting results [see RESULTS file]
local/generate_results_file.sh 2> /dev/null > RESULTS

