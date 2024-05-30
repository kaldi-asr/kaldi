#!/usr/bin/env bash
#
# Johns Hopkins University (Author : Gaurav Kumar, Daniel Povey)
# Recipe for CallHome Egyptian Arabic
# Made to integrate KALDI with JOSHUA for end-to-end ASR and SMT

. ./cmd.sh
. ./path.sh
mfccdir=`pwd`/mfcc
set -e

# Specify the location of the speech files, the transcripts and the lexicon
# These are passed off to other scripts in including the one for data and lexicon prep

eca_speech=/export/corpora/LDC/LDC97S45
eca_transcripts=/export/corpora/LDC/LDC97T19
eca_lexicon=/export/corpora/LDC/LDC99L22
sup_speech=/export/corpora/LDC/LDC2002S37
sup_transcripts=/export/corpora/LDC/LDC2002T38
h5_speech=/export/corpora/LDC/LDC2002S22
h5_transcripts=/export/corpora/LDC/LDC2002T39
split=local/splits

local/callhome_data_prep.sh $eca_speech $eca_transcripts $sup_speech $sup_transcripts $h5_speech $h5_transcripts

local/callhome_prepare_dict.sh $eca_lexicon

# Added c,j, v to the non silences phones manually
utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang

# Make sure that you do not use your test and your dev sets to train the LM
# Some form of cross validation is possible where you decode your dev/set based on an
# LM that is trained on  everything but that that conversation
local/callhome_train_lms.sh $split
local/callhome_create_test_lang.sh

utils/fix_data_dir.sh data/local/data/train_all

steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/local/data/train_all exp/make_mfcc/train_all $mfccdir || exit 1;

utils/fix_data_dir.sh data/local/data/train_all
utils/validate_data_dir.sh data/local/data/train_all

cp -r data/local/data/train_all data/train_all

# Creating data partitions for the pipeline

local/create_splits $split

# Now compute CMVN stats for the train, dev and test subsets
steps/compute_cmvn_stats.sh data/dev exp/make_mfcc/dev $mfccdir
steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test $mfccdir
steps/compute_cmvn_stats.sh data/sup exp/make_mfcc/sup $mfccdir
steps/compute_cmvn_stats.sh data/h5 exp/make_mfcc/h5 $mfccdir

steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir

# Again from Dan's recipe : Reduced monophone training data
# Now-- there are 1.6 million utterances, and we want to start the monophone training
# on relatively short utterances (easier to align), but not only the very shortest
# ones (mostly uh-huh).  So take the 100k shortest ones, and then take 10k random
# utterances from those.

steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
  data/train data/lang exp/mono0a

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train data/lang exp/mono0a exp/mono0a_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    1000 10000 data/train data/lang exp/mono0a_ali exp/tri1 || exit 1;


(utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri1/graph data/dev exp/tri1/decode_dev)&

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    1400 15000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph || exit 1;
  steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri2/graph data/dev exp/tri2/decode_dev || exit 1;
)&

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri2 exp/tri2_ali || exit 1;

# Train tri3a, which is LDA+MLLT, on 100k data.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   1800 20000 data/train data/lang exp/tri2_ali exp/tri3a || exit 1;
(
  utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
  steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri3a/graph data/dev exp/tri3a/decode_dev || exit 1;
)&

# Next we'll use fMLLR and train with SAT (i.e. on
# fMLLR features)

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri3a exp/tri3a_ali || exit 1;

steps/train_sat.sh  --cmd "$train_cmd" \
  2200 25000 data/train data/lang exp/tri3a_ali  exp/tri4a || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri4a/graph data/dev exp/tri4a/decode_dev
)&


steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri4a exp/tri4a_ali || exit 1;

# Reduce the number of gaussians
steps/train_sat.sh  --cmd "$train_cmd" \
  2600 30000 data/train data/lang exp/tri4a_ali  exp/tri5a || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri5a/graph data/dev exp/tri5a/decode_dev
)&

(
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
  exp/tri5a/graph data/test exp/tri5a/decode_test
  # Decode Supplement and H5
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
  exp/tri5a/graph data/sup exp/tri5a/decode_sup
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
  exp/tri5a/graph data/h5 exp/tri5a/decode_h5
)&

dnn_cpu_parallel_opts=(--minibatch-size 128 --max-change 10 --num-jobs-nnet 8 --num-threads 16 \
                       --parallel-opts "--num-threads 16" --cmd "queue.pl  --mem 1G")
dnn_gpu_parallel_opts=(--minibatch-size 512 --max-change 40 --num-jobs-nnet 4 --num-threads 1 \
                       --parallel-opts "--gpu 1" --cmd "queue.pl  --mem 1G")

steps/nnet2/train_pnorm_ensemble.sh \
  --mix-up 5000  --initial-learning-rate 0.008 --final-learning-rate 0.0008\
  --num-hidden-layers 4 --pnorm-input-dim 2000 --pnorm-output-dim 200\
  --cmd "$train_cmd" \
  "${dnn_gpu_parallel_opts[@]}" \
  --ensemble-size 4 --initial-beta 0.1 --final-beta 5 \
  data/train data/lang exp/tri5a_ali exp/tri6a_dnn

(
  steps/nnet2/decode.sh --nj 13 --cmd "$decode_cmd" --num-threads 4 --parallel-opts " --num-threads 4"   \
    --scoring-opts "--min-lmwt 8 --max-lmwt 16" --transform-dir exp/tri5a/decode_dev exp/tri5a/graph data/dev exp/tri6a_dnn/decode_dev
) &

# Decode test sets
(
  steps/nnet2/decode.sh --nj 13 --cmd "$decode_cmd" --num-threads 4 --parallel-opts " --num-threads 4"   \
    --scoring-opts "--min-lmwt 8 --max-lmwt 16" --transform-dir exp/tri5a/decode_test exp/tri5a/graph data/test exp/tri6a_dnn/decode_test
  steps/nnet2/decode.sh --nj 13 --cmd "$decode_cmd" --num-threads 4 --parallel-opts " --num-threads 4"   \
    --scoring-opts "--min-lmwt 8 --max-lmwt 16" --transform-dir exp/tri5a/decode_sup exp/tri5a/graph data/sup exp/tri6a_dnn/decode_sup
  steps/nnet2/decode.sh --nj 13 --cmd "$decode_cmd" --num-threads 4 --parallel-opts " --num-threads 4"   \
    --scoring-opts "--min-lmwt 8 --max-lmwt 16" --transform-dir exp/tri5a/decode_h5 exp/tri5a/graph data/h5 exp/tri6a_dnn/decode_h5
) &

wait

# (TDNN + iVectors) training
# Note that the alignments used by run_tdnn.sh come from the pnorm-ensemble model
# If you choose to skip ensemble training (which is slow), use the best
# fmllr alignments available (tri4a)
# You can modify this in local/nnet/run_tdnn.sh
local/nnet3/run_tdnn.sh

exit 0;
