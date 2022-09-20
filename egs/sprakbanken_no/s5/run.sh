
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
. ./path.sh 


stage=0
nj=16

# Download the corpus and divide the corpus into train, dev and test sets
# train ~500h
# test ~12h
# dev ~12h
if [ $stage -le 0 ]; then
    local/sprak_data_prep.sh  || exit 1;
fi

# Prepare dict folder
if [ $stage -le 1 ]; then
    local/prep_dict.sh || exit 1;
    utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang_tmp data/lang || exit 1;
fi


# Extract mfccs and compute cepstral mean and variance normalisation
if [ $stage -le 2 ]; then
    for dataset in train test dev; do
        steps/make_mfcc.sh --nj $nj --cmd $train_cmd data/$dataset || exit 1;

        steps/compute_cmvn_stats.sh data/$dataset || exit 1;

        # Repair data set (remove corrupt data points with corrupt audio)
        utils/fix_data_dir.sh data/$dataset || exit 1;

    done
fi


# Train LM with irstlm
if [ $stage -le 3 ]; then
    local/train_irstlm.sh data/local/transcript_lm/transcripts.uniq 3 "tg" data/lang data/local/train3_lm &> data/local/tg.log || exit 1;
    local/train_irstlm.sh data/local/transcript_lm/transcripts.uniq 4 "fg" data/lang data/local/train4_lm &> data/local/fg.log || exit 1;
fi

# Train monophone model 
if [ $stage -le 4 ]; then
    steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
        data/train data/lang exp/mono0a || exit 1;
    utils/mkgraph.sh --mono data/lang_test_tg exp/mono0a exp/mono0a/graph_tg || exit 1;
    steps/decode.sh --nj 8 --cmd "$decode_cmd" \
    exp/mono0a/graph_tg data/dev exp/mono0a/decode_tg_dev || exit 1;
fi



if [ $stage -le 5 ]; then
  # Train tri1 (delta+delta-delta)
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/mono0a exp/mono0a_ali || exit 1;
  steps/train_deltas.sh --cmd "$train_cmd" \
    3000 40000 data/train data/lang exp/mono0a_ali exp/tri1 || exit 1;

  # Decode dev set with both LMs
  utils/mkgraph.sh data/lang_test_tg exp/tri1 exp/tri1/graph_tg || exit 1;
  utils/mkgraph.sh data/lang_test_fg exp/tri1 exp/tri1/graph_fg || exit 1; 
  steps/decode.sh --nj 8 --cmd "$decode_cmd" \
    exp/tri1/graph_fg data/dev exp/tri1/decode_fg_dev || exit 1;
  steps/decode.sh --nj 8 --cmd "$decode_cmd" \
    exp/tri1/graph_tg data/dev exp/tri1/decode_tg_dev || exit 1;
fi

if [ $stage -le 6 ]; then
  # Train tri2a (delta + delta-delta)
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri1 exp/tri1_ali || exit 1;
  steps/train_deltas.sh --cmd "$train_cmd" \
    5000 60000 data/train data/lang exp/tri1_ali exp/tri2a || exit 1;
  utils/mkgraph.sh data/lang_test_tg exp/tri2a exp/tri2a/graph_tg || exit 1;
  steps/decode.sh --nj 8 --cmd "$decode_cmd" \
    exp/tri2a/graph_tg data/dev exp/tri2a/decode_tg_dev || exit 1;
fi

if [ $stage -le 7 ]; then
  # Train tri2b (LDA+MLLT)
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri2a exp/tri2a_ali || exit 1;
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=5 --right-context=5" \
    6500 75000 data/train data/lang exp/tri2a_ali exp/tri2b || exit 1;
  utils/mkgraph.sh data/lang_test_tg exp/tri2b exp/tri2b/graph_tg || exit 1;
  steps/decode.sh --nj 8 --cmd "$decode_cmd" \
    exp/tri2b/graph_tg data/dev exp/tri2b/decode_tg_dev || exit 1;
fi

if [ $stage -le 8 ]; then
  # From 2b system, train 3b which is LDA + MLLT + SAT.
  steps/align_si.sh  --nj $nj --cmd "$train_cmd" \
    --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali  || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" \
    7500 100000 data/train data/lang exp/tri2b_ali exp/tri3b || exit 1;

  # Decode dev with 4gram and 3gram LMs
  utils/mkgraph.sh data/lang_test_tg exp/tri3b exp/tri3b/graph_tg || exit 1;
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 8 \
    exp/tri3b/graph_tg data/dev exp/tri3b/decode_tg_dev || exit 1;
  utils/mkgraph.sh data/lang_test_fg exp/tri3b exp/tri3b/graph_fg || exit 1;
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 8 \
    exp/tri3b/graph_fg data/dev exp/tri3b/decode_fg_dev || exit 1;

  # Decode test with 4gram and 3gram LMs
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 8 --num-threads 2 \
    exp/tri3b/graph_tg data/test exp/tri3b/decode_tg_test || exit 1;
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 8 --num-threads 2 \
    exp/tri3b/graph_fg data/test exp/tri3b/decode_fg_test || exit 1;
fi


if [ $stage -le 9 ]; then
  # Caution: this part needs a GPU.
  local/chain2/run_tdnn.sh
fi


# Generating results [see RESULTS file]
local/generate_results_file.sh 2> /dev/null > RESULTS

