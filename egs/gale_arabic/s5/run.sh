#!/bin/bash 

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0

. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
nJobs=120

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

# Now make subset with half of the data from si-84.
utils/subset_data_dir.sh data/train 10000 data/train.10K || exit 1;

# Train monophone models on a subset of the data, 10K segment
# Note: the --boost-silence option should probably be omitted by default
steps/train_mono.sh --boost-silence 1.25 --nj $nJobs --cmd "$train_cmd" \
  data/train.10K data/lang exp/mono || exit 1;

# Monophone decoding
time utils/mkgraph.sh --mono data/lang_test exp/mono exp/mono/graph || exit 1;
time steps/decode.sh --nj $nJobs --cmd "$decode_cmd" exp/mono/graph \
 data/test exp/mono/decode || exit 1;
  
# Get alignments from monophone system.
steps/align_si.sh --boost-silence 1.25 --nj $nJobs --cmd "$train_cmd" \
   data/train data/lang exp/mono exp/mono_ali || exit 1;

# train tri1 [first triphone pass]
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
      1800 9000 data/train data/lang exp/mono_ali exp/tri1 || exit 1;

# First triphone decoding
time utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
time steps/decode.sh  --nj $nJobs --cmd "$decode_cmd" \
 exp/tri1/graph data/test exp/tri1/decode

steps/align_si.sh --nj $nJobs --cmd "$train_cmd" \
  data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

# Train tri2a, which is deltas+delta+deltas
steps/train_deltas.sh --cmd "$train_cmd" \
  1800 9000 data/train data/lang exp/tri1_ali exp/tri2a || exit 1;

# tri2a decoding
time utils/mkgraph.sh data/lang_test exp/tri2a exp/tri2a/graph || exit 1;
time steps/decode.sh  --nj $nJobs --cmd "$decode_cmd" \
 exp/tri2a/graph data/test exp/tri2a/decode

# ok1 

# train and decode tri2b [LDA+MLLT]
steps/train_lda_mllt.sh --cmd "$train_cmd" 1800 9000 \
  data/train data/lang exp/tri1_ali exp/tri2b || exit 1;
time utils/mkgraph.sh data/lang_test exp/tri2b exp/tri2b/graph
time steps/decode.sh  --nj $nJobs --cmd "$decode_cmd"   exp/tri2b/graph data/test exp/tri2b/decode


#removed the part for confusion network

# Align all data with LDA+MLLT system (tri2b)
steps/align_si.sh  --nj $nJobs --cmd "$train_cmd" \
  --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali  || exit 1;

# the part may not be useful:
#  Do MMI on top of LDA+MLLT.
steps/make_denlats.sh --nj $nJobs --cmd "$train_cmd" \
 data/train data/lang exp/tri2b exp/tri2b_denlats || exit 1;
steps/train_mmi.sh data/train data/lang exp/tri2b_ali \
 exp/tri2b_denlats exp/tri2b_mmi || exit 1;

steps/decode.sh  --iter 4 --nj $nJobs --cmd "$decode_cmd"  exp/tri2b/graph \
 data/test exp/tri2b_mmi/decode_it4
steps/decode.sh  --iter 3 --nj $nJobs --cmd "$decode_cmd" exp/tri2b/graph \
 data/test exp/tri2b_mmi/decode_it3 # Do the same with boosting.

steps/train_mmi.sh --boost 0.05 data/train data/lang exp/tri2b_ali \
exp/tri2b_denlats exp/tri2b_mmi_b0.05 || exit 1;

steps/decode.sh  --iter 4 --nj $nJobs --cmd "$decode_cmd" exp/tri2b/graph \
 data/test exp/tri2b_mmi_b0.05/decode_it4 || exit 1;
steps/decode.sh  --iter 3 --nj $nJobs --cmd "$decode_cmd" exp/tri2b/graph \
 data/test exp/tri2b_mmi_b0.05/decode_it3 || exit 1;

# Do MPE.
steps/train_mpe.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mpe || exit 1;

steps/decode.sh  --iter 4 --nj $nJobs --cmd "$decode_cmd" exp/tri2b/graph \
 data/test exp/tri2b_mpe/decode_it4 || exit 1;

steps/decode.sh  --iter 3 --nj $nJobs --cmd "$decode_cmd" exp/tri2b/graph \
 data/test exp/tri2b_mpe/decode_it3 || exit 1;

# the part may not be useful: END of not very useful part  


# From 2b system, train 3b which is LDA + MLLT + SAT.
steps/train_sat.sh --cmd "$train_cmd" \
  1800 9000 data/train data/lang exp/tri2b_ali exp/tri3b || exit 1;
utils/mkgraph.sh data/lang_test exp/tri3b exp/tri3b/graph|| exit 1;
steps/decode_fmllr.sh --nj $nJobs --cmd "$decode_cmd" \
  exp/tri3b/graph data/test exp/tri3b/decode || exit 1;


# From 3b system, align all si284 data.
steps/align_fmllr.sh --nj $nJobs --cmd "$train_cmd" \
  data/train data/lang exp/tri3b exp/tri3b_ali || exit 1;
  
  
## SGMM (subspace gaussian mixture model), excluding the "speaker-dependent weights"
steps/train_ubm.sh --silence-weight 0.5 --cmd "$train_cmd" 800 \
 data/train data/lang exp/tri3b_ali exp/ubm5a || exit 1;

steps/train_sgmm.sh  --cmd "$train_cmd" 4500 40000 data/train data/lang exp/tri3b_ali \
 exp/ubm5a/final.ubm exp/sgmm_5a || exit 1;

utils/mkgraph.sh data/lang_test exp/sgmm_5a exp/sgmm_5a/graph || exit 1;
steps/decode_sgmm.sh --nj $nJobs --cmd "$decode_cmd" --transform-dir exp/tri3b/decode \
 exp/sgmm_5a/graph data/test exp/sgmm_5a/decode

## boosted MMI on SGMM
steps/align_sgmm.sh --nj $nJobs --cmd "$train_cmd" --transform-dir exp/tri3b_ali  \
 --use-graphs true --use-gselect true data/train data/lang exp/sgmm_5a exp/sgmm_5a_ali
steps/make_denlats_sgmm.sh --nj $nJobs --sub-split $nJobs --cmd "$decode_cmd" --transform-dir \
 exp/tri3b_ali data/train data/lang exp/sgmm_5a_ali exp/sgmm_5a_denlats
steps/train_mmi_sgmm.sh --cmd "$decode_cmd" --transform-dir exp/tri3b_ali --boost 0.1 \
 data/train data/lang exp/sgmm_5a_ali exp/sgmm_5a_denlats exp/sgmm_5a_mmi_b0.1

#decode sgmm
utils/mkgraph.sh data/lang_test exp/sgmm_5a_mmi_b0.1 exp/sgmm_5a_mmi_b0.1/graph
steps/decode_sgmm.sh --nj $nJobs --cmd "$decode_cmd" --transform-dir exp/tri3b/decode \
 exp/sgmm_5a_mmi_b0.1/graph data/test exp/sgmm_5a_mmi_b0.1/decode

for n in 1 2 3 4; do
  steps/decode_sgmm_rescore.sh --cmd "$decode_cmd" --iter $n --transform-dir exp/tri3b/decode data/lang_test \
  data/test exp/sgmm_5a_mmi_b0.1/decode exp/sgmm_5a_mmi_b0.1/decode$n
done


## Nerual Network on top of LDA+MLLT+SAT
### Half parameters compare to previous  =>  3 hidden layers, 4 million parameters
steps/train_nnet_cpu.sh --mix-up 8000 --initial-learning-rate 0.01 --final-learning-rate 0.001 \
 --num-jobs-nnet $nJobs --num-hidden-layers 3 --num-parameters 4000000 --cmd "$decode_cmd" \
 data/train data/lang exp/tri3b exp/nnet_4m_3l

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj $nJobs --transform-dir exp/tri3b/decode \
 exp/tri3b/graph data/test exp/nnet_4m_3l/decode 

time=$(date +"%Y-%m-%d-%H-%M-%S")
#get WER
#for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; \
# done | sort -n -r -k2 > RESULTS.$USER.$time # to make sure you keep the results timed and owned

#get detailed WER; reports, conversational and combined
local/split_wer.sh $galeData > RESULTS.$USER.$time
 
 
echo training succedded
exit 0





