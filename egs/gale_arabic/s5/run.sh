#!/bin/bash 

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0

. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
nJobs=120
nDecodeJobs=40

LDC2013S02_1=/alt/data/speech/LDC/LDC2013S02/gale_p2_arb_bc_speech_p1_d1
LDC2013S02_2=/alt/data/speech/LDC/LDC2013S02/gale_p2_arb_bc_speech_p1_d2
LDC2013S02_3=/alt/data/speech/LDC/LDC2013S02/gale_p2_arb_bc_speech_p1_d3
LDC2013S02_4=/alt/data/speech/LDC/LDC2013S02/gale_p2_arb_bc_speech_p1_d4
LDC2013S07_1=/alt/data/speech/LDC/LDC2013S07/gale_p2_arb_bc_spch_p2_d1
LDC2013S07_2=/alt/data/speech/LDC/LDC2013S07/gale_p2_arb_bc_spch_p2_d2
LDC2013T17=/alt/data/speech/LDC/LDC2013T17.tgz
LDC2013T04=/alt/data/speech/LDC/LDC2013T04.tgz

galeData=GALE
#prepare the data
#split train dev test 
#prepare lexicon and LM 

# You can run the script from here automatically, but it is recommended to run the data preparation,
# and features extraction manually and and only once.
# By copying and pasting into the shell.

#copy the audio files to local folder wav and convet flac files to wav
local/gale_data_prep_audio.sh $galeData $LDC2013S02_1 $LDC2013S02_2 \
  $LDC2013S02_3 $LDC2013S02_4 $LDC2013S07_1 $LDC2013S07_2
  
#get the transcription and remove empty prompts and all noise markers  
local/gale_data_prep_txt.sh  $galeData $LDC2013T17 $LDC2013T04


# split the data to reports and conversational and for each class will have rain/dev and test
local/gale_data_prep_split.sh $galeData 

# get QCRI dictionary and add silence and UN
local/gale_prep_dict.sh 


#prepare the langauge resources
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang   

# LM training
local/gale_train_lms.sh

# G compilation, check LG composition
local/gale_format_data.sh 


# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc

for x in test train; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nJobs \
    data/$x exp/make_mfcc/$x $mfccdir
  utils/fix_data_dir.sh data/$x # some files fail to get mfcc for many reasons
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
done


# Here we start the AM

# Let's create a subset with 10k segments to make quick flat-start training:
utils/subset_data_dir.sh data/train 10000 data/train.10K || exit 1;

# Train monophone models on a subset of the data, 10K segment
# Note: the --boost-silence option should probably be omitted by default
steps/train_mono.sh --nj 40 --cmd "$train_cmd" \
  data/train.10K data/lang exp/mono || exit 1;

# Get alignments from monophone system.
steps/align_si.sh --nj $nJobs --cmd "$train_cmd" \
  data/train data/lang exp/mono exp/mono_ali || exit 1;

# train tri1 [first triphone pass]
steps/train_deltas.sh --cmd "$train_cmd" \
  2500 30000 data/train data/lang exp/mono_ali exp/tri1 || exit 1;

# First triphone decoding
time utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
time steps/decode.sh  --nj $nDecodeJobs --cmd "$decode_cmd" \
  exp/tri1/graph data/test exp/tri1/decode

steps/align_si.sh --nj $nJobs --cmd "$train_cmd" \
  data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

# Train tri2a, which is deltas+delta+deltas
steps/train_deltas.sh --cmd "$train_cmd" \
  3000 40000 data/train data/lang exp/tri1_ali exp/tri2a || exit 1;

# tri2a decoding
time utils/mkgraph.sh data/lang_test exp/tri2a exp/tri2a/graph || exit 1;
time steps/decode.sh --nj $nDecodeJobs --cmd "$decode_cmd" \
  exp/tri2a/graph data/test exp/tri2a/decode

# train and decode tri2b [LDA+MLLT]
steps/train_lda_mllt.sh --cmd "$train_cmd" 4000 50000 \
  data/train data/lang exp/tri1_ali exp/tri2b || exit 1;
time utils/mkgraph.sh data/lang_test exp/tri2b exp/tri2b/graph || exit 1;
time steps/decode.sh --nj $nDecodeJobs --cmd "$decode_cmd" exp/tri2b/graph data/test exp/tri2b/decode

# Align all data with LDA+MLLT system (tri2b)
steps/align_si.sh --nj $nJobs --cmd "$train_cmd" \
  --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali  || exit 1;

#  Do MMI on top of LDA+MLLT.
steps/make_denlats.sh --nj $nJobs --cmd "$train_cmd" \
 data/train data/lang exp/tri2b exp/tri2b_denlats || exit 1;
 
steps/train_mmi.sh data/train data/lang exp/tri2b_ali \
 exp/tri2b_denlats exp/tri2b_mmi || exit 1;

steps/decode.sh  --iter 4 --nj $nJobs --cmd "$decode_cmd"  exp/tri2b/graph \
 data/test exp/tri2b_mmi/decode_it4
steps/decode.sh  --iter 3 --nj $nJobs --cmd "$decode_cmd" exp/tri2b/graph \
 data/test exp/tri2b_mmi/decode_it3 # Do the same with boosting.

steps/train_mmi.sh --boost 0.1 data/train data/lang exp/tri2b_ali \
exp/tri2b_denlats exp/tri2b_mmi_b0.1 || exit 1;

steps/decode.sh  --iter 4 --nj $nJobs --cmd "$decode_cmd" exp/tri2b/graph \
 data/test exp/tri2b_mmi_b0.1/decode_it4 || exit 1;
steps/decode.sh  --iter 3 --nj $nJobs --cmd "$decode_cmd" exp/tri2b/graph \
 data/test exp/tri2b_mmi_b0.1/decode_it3 || exit 1;

# Do MPE.
steps/train_mpe.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mpe || exit 1;

steps/decode.sh  --iter 4 --nj $nDecodeJobs --cmd "$decode_cmd" exp/tri2b/graph \
 data/test exp/tri2b_mpe/decode_it4 || exit 1;

steps/decode.sh  --iter 3 --nj $nDecodeJobs --cmd "$decode_cmd" exp/tri2b/graph \
 data/test exp/tri2b_mpe/decode_it3 || exit 1;


# From 2b system, train 3b which is LDA + MLLT + SAT.
steps/train_sat.sh --cmd "$train_cmd" \
  5000 100000 data/train data/lang exp/tri2b_ali exp/tri3b || exit 1;
utils/mkgraph.sh data/lang_test exp/tri3b exp/tri3b/graph|| exit 1;
steps/decode_fmllr.sh --nj $nDecodeJobs --cmd "$decode_cmd" \
  exp/tri3b/graph data/test exp/tri3b/decode || exit 1;

# From 3b system, align all data.
steps/align_fmllr.sh --nj $nJobs --cmd "$train_cmd" \
  data/train data/lang exp/tri3b exp/tri3b_ali || exit 1;
  

## SGMM (subspace gaussian mixture model), excluding the "speaker-dependent weights"
steps/train_ubm.sh --cmd "$train_cmd" 700 \
 data/train data/lang exp/tri3b_ali exp/ubm5a || exit 1;
 
steps/train_sgmm2.sh --cmd "$train_cmd" 5000 20000 data/train data/lang exp/tri3b_ali \
  exp/ubm5a/final.ubm exp/sgmm_5a || exit 1;

utils/mkgraph.sh data/lang_test exp/sgmm_5a exp/sgmm_5a/graph || exit 1;

steps/decode_sgmm2.sh --nj $nDecodeJobs --cmd "$decode_cmd" --config conf/decode.config \
  --transform-dir exp/tri3b/decode exp/sgmm_5a/graph data/test exp/sgmm_5a/decode

steps/align_sgmm2.sh --nj $nJobs --cmd "$train_cmd" --transform-dir exp/tri3b_ali \
  --use-graphs true --use-gselect true data/train data/lang exp/sgmm_5a exp/sgmm_5a_ali || exit 1;

## boosted MMI on SGMM
steps/make_denlats_sgmm2.sh --nj $nJobs --sub-split 30 --beam 9.0 --lattice-beam 6 \
  --cmd "$decode_cmd" --transform-dir \
  exp/tri3b_ali data/train data/lang exp/sgmm_5a_ali exp/sgmm_5a_denlats || exit 1;
  
steps/train_mmi_sgmm2.sh --cmd "$train_cmd" --num-iters 8 --transform-dir exp/tri3b_ali --boost 0.1 \
  data/train data/lang exp/sgmm_5a exp/sgmm_5a_denlats exp/sgmm_5a_mmi_b0.1
 
#decode GMM MMI
utils/mkgraph.sh data/lang_test exp/sgmm_5a_mmi_b0.1 exp/sgmm_5a_mmi_b0.1/graph || exit 1;

steps/decode_sgmm2.sh --nj $nDecodeJobs --cmd "$decode_cmd" --config conf/decode.config \
  --transform-dir exp/tri3b/decode exp/sgmm_5a_mmi_b0.1/graph data/test exp/sgmm_5a_mmi_b0.1/decode
  
for n in 1 2 3 4; do
  steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $n --transform-dir exp/tri3b/decode data/lang_test \
    data/test exp/sgmm_5a_mmi_b0.1/decode exp/sgmm_5a_mmi_b0.1/decode$n
  
  steps/decode_sgmm_rescore.sh --cmd "$decode_cmd" --iter $n --transform-dir exp/tri3b/decode data/lang_test \
    data/test exp/sgmm_5a/decode exp/sgmm_5a_mmi_onlyRescoreb0.1/decode$n
done


#train DNN
mfcc_fmllr_dir=mfcc_fmllr
baseDir=exp/tri3b
alignDir=exp/tri3b_ali
dnnDir=exp/tri3b_dnn_2048x5
align_dnnDir=exp/tri3b_dnn_2048x5_ali
dnnLatDir=exp/tri3b_dnn_2048x5_denlats
dnnMPEDir=exp/tri3b_dnn_2048x5_smb

trainTr90=data/train_tr90
trainCV=data/train_cv10 

steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$cuda_cmd" \
  --transform-dir $baseDir/decode data/test_fmllr data/test \
  $baseDir $mfcc_fmllr_dir/log_test $mfcc_fmllr_dir || exit 1;

steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$cuda_cmd" \
  --transform-dir $alignDir data/train_fmllr data/train \
  $baseDir $mfcc_fmllr_dir/log_train $mfcc_fmllr_dir || exit 1;
                            
utils/subset_data_dir_tr_cv.sh  data/train_fmllr $trainTr90 $trainCV || exit 1;

(tail --pid=$$ -F $dnnDir/train_nnet.log 2>/dev/null)& 
$cuda_cmd $dnnDir/train_nnet.log \
steps/train_nnet.sh --use-gpu-id 0 --hid-dim 2048 --hid-layers 5 --learn-rate 0.008 \
$trainTr90 $trainCV data/lang $alignDir $alignDir $dnnDir || exit 1;

steps/decode_nnet.sh --nj $nDecodeJobs --cmd $decode_cmd --config conf/decode_dnn.config \
  --nnet $dnnDir/final.nnet --acwt 0.08 $baseDir/graph data/test_fmllr $dnnDir/decode || exit 1;

#
steps/nnet/align.sh --nj $nJobs --cmd $train_cmd data/train_fmllr data/lang \
  $dnnDir $align_dnnDir || exit 1;

steps/nnet/make_denlats.sh --nj $nJobs --cmd $train_cmd --config conf/decode_dnn.config --acwt 0.1 \
  data/train_fmllr data/lang $dnnDir $dnnLatDir || exit 1;

steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt 0.1 --do-smbr true \
  data/train_fmllr data/lang $dnnDir $align_dnnDir $dnnLatDir $dnnMPEDir || exit 1;
  
#decode
for n in 1 2 3 4 5 6; do
  steps/decode_nnet.sh --nj $nDecodeJobs --cmd "$train_cmd" --config conf/decode_dnn.config \
  --nnet $dnnMPEDir/$n.nnet --acwt 0.08 \
  $baseDir/graph data/test_fmllr $dnnMPEDir/decode_test_it$n || exit 1;

# End of DNN

time=$(date +"%Y-%m-%d-%H-%M-%S")
#get WER
for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; \
done | sort -n -r -k2 > RESULTS.$USER.$time # to make sure you keep the results timed and owned

#get detailed WER; reports, conversational and combined
local/split_wer.sh $galeData > RESULTS.details.$USER.$time
 

echo training succedded
exit 0





