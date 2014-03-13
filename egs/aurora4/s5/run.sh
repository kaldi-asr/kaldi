#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

case 0 in    #goto here
    1)


aurora4=/mnt/spdb/aurora4
#we need lm, trans, from WSJ0 CORPUS
wsj0=/mnt/spdb/wall_street_journal

local/aurora4_data_prep.sh $aurora4 $wsj0

local/wsj_prepare_dict.sh || exit 1;

utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;

local/aurora4_format_data.sh || exit 1;

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc
for x in train_si84_clean train_si84_multi test_eval92 test_0166 dev_0330 dev_1206; do 
 steps/make_mfcc.sh  --nj 10 \
   data/$x exp/make_mfcc/$x $mfccdir || exit 1;
 steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

# make fbank features
fbankdir=fbank
mkdir -p data-fbank
for x in train_si84_clean train_si84_multi dev_0330 dev_1206 test_eval92 test_0166; do
  cp -r data/$x data-fbank/$x
  steps/make_fbank.sh --nj 10 \
    data-fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
done

# Note: the --boost-silence option should probably be omitted by default
# for normal setups.  It doesn't always help. [it's to discourage non-silence
# models from modeling silence.]
#steps/train_mono.sh --boost-silence 1.25 --nj 10  \
#  data/train_si84_clean data/lang exp/mono0a || exit 1;

steps/train_mono.sh --boost-silence 1.25 --nj 10  \
  data/train_si84_multi data/lang exp/mono0a_multi || exit 1;
#(
# utils/mkgraph.sh --mono data/lang_test_tgpr exp/mono0a exp/mono0a/graph_tgpr && \
# steps/decode.sh --nj 8  \
#   exp/mono0a/graph_tgpr data/test_eval92 exp/mono0a/decode_tgpr_eval92 
#) &

#steps/align_si.sh --boost-silence 1.25 --nj 10  \
#   data/train_si84_clean data/lang exp/mono0a exp/mono0a_ali || exit 1;

steps/align_si.sh --boost-silence 1.25 --nj 10  \
   data/train_si84_multi data/lang exp/mono0a_multi exp/mono0a_multi_ali || exit 1;

#steps/train_deltas.sh --boost-silence 1.25 \
#    2000 10000 data/train_si84_clean data/lang exp/mono0a_ali exp/tri1 || exit 1;

steps/train_deltas.sh --boost-silence 1.25 \
    2000 10000 data/train_si84_multi data/lang exp/mono0a_multi_ali exp/tri1_multi || exit 1;


while [ ! -f data/lang_test_tgpr/tmp/LG.fst ] || \
   [ -z data/lang_test_tgpr/tmp/LG.fst ]; do
  sleep 20;
done
sleep 30;
# or the mono mkgraph.sh might be writing 
# data/lang_test_tgpr/tmp/LG.fst which will cause this to fail.

steps/align_si.sh --nj 10 \
  data/train_si84_multi data/lang exp/tri1_multi exp/tri1_multi_ali_si84 || exit 1;

steps/train_deltas.sh  \
  2500 15000 data/train_si84_multi data/lang exp/tri1_multi_ali_si84 exp/tri2a_multi || exit 1;


steps/train_lda_mllt.sh \
   --splice-opts "--left-context=3 --right-context=3" \
   2500 15000 data/train_si84_multi data/lang exp/tri1_multi_ali_si84 exp/tri2b_multi || exit 1;


utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri2b_multi exp/tri2b_multi/graph_tgpr_5k || exit 1;
steps/decode.sh --nj 8 \
  exp/tri2b_multi/graph_tgpr_5k data/test_eval92 exp/tri2b_multi/decode_tgpr_5k_eval92 || exit 1;

# Align tri2b system with si84 multi-condition data.
steps/align_si.sh  --nj 10 \
  --use-graphs true data/train_si84_multi data/lang exp/tri2b_multi exp/tri2b_multi_ali_si84  || exit 1;

steps/align_si.sh  --nj 10 \
  data/dev_0330 data/lang exp/tri2b_multi exp/tri2b_multi_ali_dev_0330 || exit 1;

steps/align_si.sh  --nj 10 \
  data/dev_1206 data/lang exp/tri2b_multi exp/tri2b_multi_ali_dev_1206 || exit 1;

#Now begin train DNN systems on multi data
. ./path.sh
#RBM pretrain
dir=exp/tri3a_dnn_pretrain
$cuda_cmd $dir/_pretrain_dbn.log \
  steps/nnet/pretrain_dbn.sh --use-gpu-id 0 --nn-depth 7 --rbm-iter 3 data-fbank/train_si84_multi $dir

dir=exp/tri3a_dnn
ali=exp/tri2b_multi_ali_si84
ali_dev=exp/tri2b_multi_ali_dev_0330
feature_transform=exp/tri3a_dnn_pretrain/final.feature_transform
dbn=exp/tri3a_dnn_pretrain/7.dbn
$cuda_cmd $dir/_train_nnet.log \
  steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 --use-gpu-id 0 \
  data-fbank/train_si84_multi data-fbank/dev_0330 data/lang $ali $ali_dev $dir || exit 1;

utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri3a_dnn exp/tri3a_dnn/graph_tgpr_5k || exit 1;
dir=exp/tri3a_dnn
steps/nnet/decode.sh --nj 8 --acwt 0.10 --config conf/decode_dnn.config \
  exp/tri3a_dnn/graph_tgpr_5k data-fbank/test_eval92 $dir/decode_tgpr_5k_eval92 || exit 1;


#realignments
srcdir=exp/tri3a_dnn
steps/nnet/align.sh --nj 10 \
  data-fbank/train_si84_multi data/lang $srcdir ${srcdir}_ali_si84_multi || exit 1;
steps/nnet/align.sh --nj 10 \
  data-fbank/dev_0330 data/lang $srcdir ${srcdir}_ali_dev_0330 || exit 1;

#train system again 

dir=exp/tri4a_dnn
ali=exp/tri3a_dnn_ali_si84_multi
ali_dev=exp/tri3a_dnn_ali_dev_0330
feature_transform=exp/tri3a_dnn_pretrain/final.feature_transform
dbn=exp/tri3a_dnn_pretrain/7.dbn
$cuda_cmd $dir/_train_nnet.log \
  steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 --use-gpu-id 0 \
  data-fbank/train_si84_multi data-fbank/dev_0330 data/lang $ali $ali_dev $dir || exit 1;

utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri4a_dnn exp/tri4a_dnn/graph_tgpr_5k || exit 1;
dir=exp/tri4a_dnn
steps/nnet/decode.sh --nj 8 --acwt 0.10 --config conf/decode_dnn.config \
  exp/tri4a_dnn/graph_tgpr_5k data-fbank/test_eval92 $dir/decode_tgpr_5k_eval92 || exit 1;


# DNN Sequential DT training
#......
