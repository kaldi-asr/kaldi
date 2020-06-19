#!/usr/bin/env bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

case 0 in    #goto here
    1)
;;           #here:
esac

#exit 1;
#need wsj0 for the clean version and LMs
#wsj0=/mnt/spdb/wall_street_journal
wsj0=/export/corpora5/LDC/LDC93S6B
local/clean_wsj0_data_prep.sh $wsj0

#reverb=/mnt/spdb/CHiME/chime2-wsj0/reverberated 
reverb=/export/corpora5/ChiME/chime2-wsj0/reverberated
local/reverb_wsj0_data_prep.sh $reverb 

#noisy=/mnt/spdb/CHiME/chime2-wsj0/isolated
noisy=/export/corpora5/ChiME/chime2-wsj0/isolated
local/noisy_wsj0_data_prep.sh $noisy 

local/wsj_prepare_dict.sh || exit 1;

utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;

local/chime_format_data.sh || exit 1;

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.

mfccdir=mfcc
for x in test_eval92_clean test_eval92_5k_clean dev_dt_05_clean dev_dt_20_clean train_si84_clean; do 
 steps/make_mfcc.sh --nj 10 --cmd "$train_cmd" \
   data/$x exp/make_mfcc/$x $mfccdir || exit 1;
 steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

# Note: the --boost-silence option should probably be omitted by default
# for normal setups.  It doesn't always help. [it's to discourage non-silence
# models from modeling silence.]
mfccdir=mfcc
for x in test_eval92_5k_noisy dev_dt_05_noisy train_si84_noisy; do 
 steps/make_mfcc.sh --nj 10 --cmd "$train_cmd" \
   data/$x exp/make_mfcc/$x $mfccdir || exit 1;
 steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

mfccdir=mfcc
for x in dev_dt_05_reverb train_si84_reverb; do 
 steps/make_mfcc.sh --nj 10 --cmd "$train_cmd" \
   data/$x exp/make_mfcc/$x $mfccdir || exit 1;
 steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

# make fbank features
mkdir -p data-fbank
fbankdir=fbank
for x in test_eval92_clean test_eval92_5k_clean dev_dt_05_clean dev_dt_20_clean train_si84_clean; do 
 cp -r data/$x data-fbank/$x
 steps/make_fbank.sh --nj 10 --cmd "$train_cmd" \
   data-fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
done

fbankdir=fbank
for x in test_eval92_5k_noisy dev_dt_05_noisy train_si84_noisy; do 
 cp -r data/$x data-fbank/$x
 steps/make_fbank.sh --nj 10 --cmd "$train_cmd" \
   data-fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
done

fbankdir=fbank
for x in dev_dt_05_reverb train_si84_reverb; do 
 cp -r data/$x data-fbank/$x
 steps/make_fbank.sh --nj 10 --cmd "$train_cmd" \
   data-fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
done

#begin train gmm systems using multi condition data
#train_si84 = clean+reverb+noisy, 
for s in train_si84 ; do 
  mkdir -p data/$s
  cp data/${s}_clean/spk2gender data/$s/ 
  for x in text wav.scp; do
    cat data/${s}_clean/$x data/${s}_reverb/$x data/${s}_noisy/$x | sort -k1 > data/$s/$x 
  done
  cat data/$s/wav.scp | awk '{print $1}' | perl -ane 'chop; m:^...:; print "$_ $&\n";' > data/$s/utt2spk 
  cat data/$s/utt2spk | utils/utt2spk_to_spk2utt.pl > data/$s/spk2utt 
done

mfccdir=mfcc
for x in train_si84; do 
 steps/make_mfcc.sh --nj 10 --cmd "$train_cmd" \
   data/$x exp/make_mfcc/$x $mfccdir || exit 1;
 steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

fbankdir=fbank
for x in train_si84; do 
 cp -r data/$x data-fbank/$x
 steps/make_fbank.sh --nj 10 --cmd "$train_cmd" \
   data-fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
done


steps/train_mono.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/mono0a || exit 1;



utils/mkgraph.sh data/lang_test_tgpr_5k exp/mono0a exp/mono0a/graph_tgpr_5k
#steps/decode.sh --nj 8  \
#  exp/mono0a/graph_tgpr_5k data/test_eval92_5k_clean exp/mono0a/decode_tgpr_eval92_5k_clean
steps/decode.sh --nj 8  --cmd "$train_cmd" \
  exp/mono0a/graph_tgpr_5k data/test_eval92_5k_noisy exp/mono0a/decode_tgpr_eval92_5k_noisy
 

steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
   data/train_si84 data/lang exp/mono0a exp/mono0a_ali || exit 1;

steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train_si84 data/lang exp/mono0a_ali exp/tri1 || exit 1;
utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri1 exp/tri1/graph_tgpr_5k || exit 1;

#steps/decode.sh --nj 8 \
#  exp/tri1/graph_tgpr data/test_eval92_5k_clean exp/tri1/decode_tgpr_eval92_5k_clean || exit 1;
steps/decode.sh --nj 8 --cmd "$train_cmd" \
  exp/tri1/graph_tgpr_5k data/test_eval92_5k_noisy exp/tri1/decode_tgpr_eval92_5k_noisy || exit 1;


# test various modes of LM rescoring (4 is the default one).
# This is just confirming they're equivalent.
#for mode in 1 2 3 4; do
#steps/lmrescore.sh --mode $mode --cmd "$decode_cmd" data/lang_test_{tgpr,tg} \
#  data/test_dev93 exp/tri1/decode_tgpr_dev93 exp/tri1/decode_tgpr_dev93_tg$mode  || exit 1;
#done

# demonstrate how to get lattices that are "word-aligned" (arcs coincide with
# words, with boundaries in the right place).
#sil_label=`grep '!SIL' data/lang_test_tgpr/words.txt | awk '{print $2}'`
#steps/word_align_lattices.sh --cmd "$train_cmd" --silence-label $sil_label \
#  data/lang_test_tgpr exp/tri1/decode_tgpr_dev93 exp/tri1/decode_tgpr_dev93_aligned || exit 1;

steps/align_si.sh --nj 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri1 exp/tri1_ali_si84 || exit 1;

# Train tri2a, which is deltas + delta-deltas, on si84 data.
steps/train_deltas.sh --cmd "$train_cmd" \
  2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2a || exit 1;

utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri2a exp/tri2a/graph_tgpr_5k || exit 1;

#steps/decode.sh --nj 8  \
#  exp/tri2a/graph_tgpr_5k data/test_eval92_5k_clean exp/tri2a/decode_tgpr_eval92_5k_clean || exit 1;
steps/decode.sh --nj 8 --cmd "$train_cmd" \
  exp/tri2a/graph_tgpr_5k data/test_eval92_5k_noisy exp/tri2a/decode_tgpr_eval92_5k_noisy|| exit 1;

#utils/mkgraph.sh data/lang_test_bg_5k exp/tri2a exp/tri2a/graph_bg5k
#steps/decode.sh --nj 8 \
#  exp/tri2a/graph_bg5k data/test_eval92_5k_clean exp/tri2a/decode_bg_eval92_5k_clean || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2b || exit 1;

utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri2b exp/tri2b/graph_tgpr_5k || exit 1;
steps/decode.sh --nj 8 --cmd "$train_cmd" \
  exp/tri2b/graph_tgpr_5k data/test_eval92_5k_noisy exp/tri2b/decode_tgpr_eval92_5k_noisy || exit 1;
#steps/decode.sh --nj 8 \
#  exp/tri2b/graph_tgpr data/test_eval92_clean exp/tri2b/decode_tgpr_eval92_clean || exit 1;


# Align tri2b system with si84 data.
steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
  --use-graphs true data/train_si84 data/lang exp/tri2b exp/tri2b_ali_si84  || exit 1;


# From 2b system, train 3b which is LDA + MLLT + SAT.
steps/train_sat.sh --cmd "$train_cmd" \
  2500 15000 data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri3b || exit 1;
utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri3b exp/tri3b/graph_tgpr_5k || exit 1;
steps/decode_fmllr.sh --nj 8 --cmd "$train_cmd" \
  exp/tri3b/graph_tgpr_5k data/test_eval92_5k_noisy exp/tri3b/decode_tgpr_eval92_5k_noisy || exit 1;


# From 3b multi-condition system, align noisy si84 data.
steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" \
  data/train_si84_noisy data/lang exp/tri3b exp/tri3b_ali_si84_noisy || exit 1;

steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" \
  data/dev_dt_05_noisy data/lang exp/tri3b exp/tri3b_ali_dev_dt_05 || exit 1;

#begin training DNN-HMM system
#only on noisy si84 

. ./path.sh
#RBM pretraining
dir=exp/tri4a_dnn_pretrain
$cuda_cmd $dir/_pretrain_dbn.log \
  steps/nnet/pretrain_dbn.sh --nn-depth 7 --rbm-iter 3 data-fbank/train_si84_noisy $dir
#BP 
dir=exp/tri4a_dnn
ali=exp/tri3b_ali_si84_noisy
ali_dev=exp/tri3b_ali_dev_dt_05
feature_transform=exp/tri4a_dnn_pretrain/final.feature_transform
dbn=exp/tri4a_dnn_pretrain/7.dbn
$cuda_cmd $dir/_train_nnet.log \
  steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
  data-fbank/train_si84_noisy data-fbank/dev_dt_05_noisy data/lang $ali $ali_dev $dir || exit 1;

utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri4a_dnn exp/tri4a_dnn/graph_tgpr_5k || exit 1;
steps/nnet/decode.sh --nj 8 --acwt 0.10 --config conf/decode_dnn.config \
  exp/tri4a_dnn/graph_tgpr_5k data-fbank/test_eval92_5k_noisy $dir/decode_tgpr_5k_eval92_5k_noisy || exit 1;

#Retrain system using new ali,
#this is essential 
#repeat this process for 3 times 
srcdir=exp/tri4a_dnn
steps/nnet/align.sh --nj 10 \
  data-fbank/train_si84_noisy data/lang $srcdir ${srcdir}_ali_si84_noisy || exit 1;
steps/nnet/align.sh --nj 10 \
  data-fbank/dev_dt_05_noisy data/lang $srcdir ${srcdir}_ali_dt_05_noisy || exit 1;

#no need to do pretraining again
dir=exp/tri5a_dnn
ali=exp/tri4a_dnn_ali_si84_noisy
ali_dev=exp/tri4a_dnn_ali_dt_05_noisy
feature_transform=exp/tri4a_dnn_pretrain/final.feature_transform
dbn=exp/tri4a_dnn_pretrain/7.dbn
$cuda_cmd $dir/_train_nnet.log \
  steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
  data-fbank/train_si84_noisy data-fbank/dev_dt_05_noisy data/lang $ali $ali_dev $dir || exit 1;

utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri5a_dnn exp/tri5a_dnn/graph_tgpr_5k || exit 1;
steps/nnet/decode.sh --nj 8 --acwt 0.10 --config conf/decode_dnn.config \
  exp/tri5a_dnn/graph_tgpr_5k data-fbank/test_eval92_5k_noisy $dir/decode_tgpr_5k_eval92_5k_noisy || exit 1;


srcdir=exp/tri5a_dnn
steps/nnet/align.sh --nj 10 \
  data-fbank/train_si84_noisy data/lang $srcdir ${srcdir}_ali_si84_noisy || exit 1;
steps/nnet/align.sh --nj 10 \
  data-fbank/dev_dt_05_noisy data/lang $srcdir ${srcdir}_ali_dt_05_noisy || exit 1;

. ./path.sh
dir=exp/tri6a_dnn
ali=exp/tri5a_dnn_ali_si84_noisy
ali_dev=exp/tri5a_dnn_ali_dt_05_noisy
feature_transform=exp/tri4a_dnn_pretrain/final.feature_transform
dbn=exp/tri4a_dnn_pretrain/7.dbn
$cuda_cmd $dir/_train_nnet.log \
  steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
  data-fbank/train_si84_noisy data-fbank/dev_dt_05_noisy data/lang $ali $ali_dev $dir || exit 1;

utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri6a_dnn exp/tri6a_dnn/graph_tgpr_5k || exit 1;
steps/nnet/decode.sh --nj 8 --acwt 0.10 --config conf/decode_dnn.config \
  exp/tri6a_dnn/graph_tgpr_5k data-fbank/test_eval92_5k_noisy $dir/decode_tgpr_5k_eval92_5k_noisy || exit 1;

srcdir=exp/tri6a_dnn
steps/nnet/align.sh --nj 10 \
  data-fbank/train_si84_noisy data/lang $srcdir ${srcdir}_ali_si84_noisy || exit 1;
steps/nnet/align.sh --nj 10 \
  data-fbank/dev_dt_05_noisy data/lang $srcdir ${srcdir}_ali_dt_05_noisy || exit 1;

. ./path.sh
dir=exp/tri7a_dnn
ali=exp/tri6a_dnn_ali_si84_noisy
ali_dev=exp/tri6a_dnn_ali_dt_05_noisy
feature_transform=exp/tri4a_dnn_pretrain/final.feature_transform
dbn=exp/tri4a_dnn_pretrain/7.dbn
$cuda_cmd $dir/_train_nnet.log \
  steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
  data-fbank/train_si84_noisy data-fbank/dev_dt_05_noisy data/lang $ali $ali_dev $dir || exit 1;

utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri7a_dnn exp/tri7a_dnn/graph_tgpr_5k || exit 1;
steps/nnet/decode.sh --nj 8 --acwt 0.10 --config conf/decode_dnn.config \
  exp/tri7a_dnn/graph_tgpr_5k data-fbank/test_eval92_5k_noisy $dir/decode_tgpr_5k_eval92_5k_noisy || exit 1;

# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. We use usually good acwt 0.1
# Lattices are re-generated after 1st epoch, to get faster convergence.
dir=exp/tri7a_dnn_smbr
srcdir=exp/tri7a_dnn
acwt=0.1

# First we generate lattices and alignments:
# awk -v FS="/" '{ NF_nosuffix=$NF; gsub(".gz","",NF_nosuffix); print NF_nosuffix gunzip -c "$0" |"; }' in 
# steps/nnet/make_denlats.sh
steps/nnet/align.sh --nj 10 --cmd "$train_cmd" \
    data-fbank/train_si84_noisy data/lang $srcdir ${srcdir}_ali || exit 1;
steps/nnet/make_denlats.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    data-fbank/train_si84_noisy data/lang $srcdir ${srcdir}_denlats || exit 1;

# Re-train the DNN by 1 iteration of sMBR 
steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt --do-smbr true \
    data-fbank/train_si84_noisy data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
# Decode (reuse HCLG graph)
for ITER in 1; do
    steps/nnet/decode.sh --nj 8 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri7a_dnn/graph_tgpr_5k data-fbank/dev_dt_05_noisy $dir/decode_tgpr_5k_dt_05_noisy_it${ITER} || exit 1;
    steps/nnet/decode.sh --nj 8 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri7a_dnn/graph_tgpr_5k data-fbank/test_eval92_5k_noisy $dir/decode_tgpr_5k_eval92_5k_noisy_it${ITER} || exit 1;
done 

# Re-generate lattices, run 4 more sMBR iterations
dir=exp/tri7a_dnn_smbr_i1lats
srcdir=exp/tri7a_dnn_smbr
acwt=0.1

# Generate lattices and alignments:
steps/nnet/align.sh --nj 10 --cmd "$train_cmd" \
    data-fbank/train_si84_noisy data/lang $srcdir ${srcdir}_ali || exit 1;
steps/nnet/make_denlats.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    data-fbank/train_si84_noisy data/lang $srcdir ${srcdir}_denlats || exit 1;

# Re-train the DNN by 1 iteration of sMBR 
steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt --do-smbr true \
    data-fbank/train_si84_noisy data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1

    # Decode (reuse HCLG graph)
for ITER in 1 2 3 4; do
    steps/nnet/decode.sh --nj 8 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri7a_dnn/graph_tgpr_5k data-fbank/dev_dt_05_noisy $dir/decode_tgpr_5k_dt_05_noisy_it${ITER} || exit 1;
    steps/nnet/decode.sh --nj 8 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri7a_dnn/graph_tgpr_5k data-fbank/test_eval92_5k_noisy $dir/decode_tgpr_5k_eval92_5k_noisy_it${ITER} || exit 1;
done 
