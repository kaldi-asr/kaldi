#!/bin/bash
### [men at work sign] ###
### WORK IN PROGRESS###
# Copyright 2010-2011 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

exit 1 # Don't run this... it's to be run line by line from the shell.

# call the next line with the directory where the RM data is
# (the argument below is just an example).  This should contain
# subdirectories named as follows:
#    rm1_audio1  rm1_audio2	rm2_audio

local/RM_data_prep.sh /mnt/matylda2/data/RM/

local/RM_format_data.sh

# mfccdir should be some place with a largish disk where you
# want to store MFCC features. 
mfccdir=/mnt/matylda6/jhu09/qpovey/kaldi_rm_mfcc

steps/make_mfcc.sh data/train exp/make_mfcc/train $mfccdir 4
for test in mar87 oct87 feb89 oct89 feb91 sep92; do
  steps/make_mfcc.sh data/test_$test exp/make_mfcc/test_$test $mfccdir 4
done

scripts/subset_data_dir.sh data/train 1000 data/train.1k

# train monophone system.
steps/train_mono.sh data/train.1k data/lang exp/mono


local/decode.sh --mono steps/decode_deltas.sh exp/mono/decode


# Get alignments from monophone system.
steps/align_deltas.sh data/train data/lang exp/mono exp/mono_ali

# train tri1 [first triphone pass]
steps/train_deltas.sh data/train data/lang exp/mono_ali exp/tri1
# decode tri1
local/decode.sh steps/decode_deltas.sh exp/tri1/decode

# align tri1
steps/align_deltas.sh --graphs "ark,s,cs:gunzip -c exp/tri1/graphs.fsts.gz|" \
    $data/train data/lang exp/tri1 exp/tri1_ali

# train tri2a [delta+delta-deltas]
steps/train_deltas.sh data/train data/lang exp/tri1_ali exp/tri2a
# decode tri2a
local/decode.sh steps/decode_deltas.sh exp/tri2a/decode

# train tri2b [LDA+MLLT]
steps/train_lda_mllt.sh data/train data/lang exp/tri1_ali exp/tri2b
# decode tri2b
local/decode.sh steps/decode_lda_mllt.sh exp/tri2b/decode

# Train and test ET.
steps/train_lda_et.sh data/train data/lang exp/tri1_ali exp/tri2c
scripts/mkgraph.sh data/lang_test exp/tri2c exp/tri2c/graph
local/decode.sh steps/decode_lda_et.sh exp/tri2c/decode

# Align all data with LDA+MLLT system (tri2b) and do LDA+MLLT+SAT
steps/align_lda_mllt.sh --graphs "ark,s,cs:gunzip -c exp/tri2b/graphs.fsts.gz|" \
   data/train data/lang exp/tri2b exp/tri2b_ali
steps/train_lda_mllt_sat.sh data/train data/lang exp/tri2b_ali exp/tri3d
scripts/mkgraph.sh data/lang_test exp/tri3d exp/tri3d/graph
local/decode.sh steps/decode_lda_mllt_sat.sh exp/tri3d/decode

# Align all data with LDA+MLLT+SAT system (tri3d)
steps/align_lda_mllt_sat.sh --graphs "ark,s,cs:gunzip -c exp/tri3d/graphs.fsts.gz|" \
    data/train data/lang exp/tri3d exp/tri3d_ali

# Try another pass on top of that.
steps/train_lda_mllt_sat.sh data/train data/lang exp/tri3d_ali exp/tri4d
scripts/mkgraph.sh data/lang_test exp/tri4d exp/tri4d/graph
local/decode.sh steps/decode_lda_mllt_sat.sh exp/tri4d/decode

# Next, SGMM system-- train SGMM system with speaker vectors, on top 
# of LDA+MLLT features.

steps/train_ubm_lda_etc.sh data/train data/lang exp/tri2b_ali exp/ubm3d
steps/train_sgmm_lda_etc.sh data/train data/lang exp/tri2b_ali exp/ubm3d/final.ubm exp/sgmm3d

scripts/mkgraph.sh data/lang_test exp/sgmm3d exp/sgmm3d/graph
local/decode.sh steps/decode_sgmm_lda_etc.sh exp/sgmm3d/decode

# Align LDA+ET system prior to training corresponding SGMM system.
steps/align_lda_et.sh --graphs "ark,s,cs:gunzip -c exp/tri2c/graphs.fsts.gz|" \
  data/train data/lang exp/tri2c exp/tri2c_ali 

# Train SGMM system on top of LDA+ET.
steps/train_ubm_lda_etc.sh data/train data/lang exp/tri2c_ali exp/ubm3e
steps/train_sgmm_lda_etc.sh data/train data/lang exp/tri2c_ali exp/ubm3e/final.ubm exp/sgmm3e

local/decode.sh steps/decode_sgmm_lda_etc.sh exp/sgmm3e/decode exp/tri2c/decode

# Now train SGMM system on top of LDA+MLLT+SAT
steps/train_ubm_lda_etc.sh data/train data/lang exp/tri3d_ali exp/ubm4f
steps/train_sgmm_lda_etc.sh data/train data/lang exp/tri3d_ali exp/ubm4f/final.ubm exp/sgmm4f

local/decode.sh steps/decode_sgmm_lda_etc.sh exp/sgmm4f/decode exp/tri3d/decode

for x in exp/*/decode; do grep WER $x/wer_* | scripts/best_wer.sh; done

exp/mono/decode/wer_6:%WER 10.340701 [ 1296 / 12533, 95 ins, 391 del, 810 sub ]
exp/sgmm3d/decode/wer_5:%WER 2.267284 [ 284 / 12526, 38 ins, 51 del, 195 sub ]
exp/sgmm3e/decode/wer_6:%WER 2.122397 [ 266 / 12533, 37 ins, 51 del, 178 sub ]
exp/sgmm4f/decode/wer_4:%WER 1.795261 [ 225 / 12533, 45 ins, 37 del, 143 sub ]
exp/tri1/decode/wer_6:%WER 3.566584 [ 447 / 12533, 74 ins, 88 del, 285 sub ]
exp/tri2a/decode/wer_7:%WER 3.518711 [ 441 / 12533, 57 ins, 91 del, 293 sub ]
exp/tri2b/decode/wer_9:%WER 3.614458 [ 453 / 12533, 59 ins, 111 del, 283 sub ]
exp/tri2c/decode/wer_6:%WER 2.833653 [ 355 / 12528, 54 ins, 71 del, 230 sub ]
exp/tri3d/decode/wer_7:%WER 2.489428 [ 312 / 12533, 43 ins, 63 del, 206 sub ]
exp/tri4d/decode/wer_7:%WER 2.649007 [ 332 / 12533, 53 ins, 67 del, 212 sub ]


##### Below here is trash. ######

#steps/train_lda_mllt.sh.bak data/train data/train.1k data/lang exp/tri1 exp/tri2b_tmp

#scripts/subset_data_dir.sh data/train 800 data/train.800
#steps/train_lda_mllt.sh data/train data/train.800 data/lang exp/tri1_ali exp/tri2b_tmp2


scripts/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph
for test in mar87 oct87 feb89 oct89 feb91 sep92; do
  steps/decode_deltas.sh exp/tri1 data/test_$test data/lang exp/tri1/decode_$test &
done
wait
scripts/average_wer.sh exp/mono/decode_?????/wer > exp/mono/wer



scripts/mkgraph.sh --mono exp/mono/tree exp/mono/final.mdl exp/mono/graph


\
   > $dir/wer

notes on structure...




scripts/ contains generic scripts
local/ contains more corpus-specific scripts
steps/ contains system-building steps...


data/local  contains temp., local stuff
data/train
data/train.1k
data/lang  [note: could have separate dirs like this for different test sets]
data/test_feb89
data/test_feb89


local/RM_data_prep.sh


steps/train_mono.sh




exp/ contains experiments.
  [ Decode_dirs in subdir of exp. dir? ]


ocal_scripts/ contains the most RM-specific scripts. [used to create data_prep/]

scripts/ will contain generic scipts.

Stuff that's about the language:

lang/
  words.txt phones.txt silphones.csl nonsilphones.csl topo
  L.fst

maybe also, later:
  phonesets.txt [ phonesets used in building questions... if not supplied, use the "base phones" ]
  extra_questions.txt [ extra questions appended to automatically generated questions.  Should ask 
        questions that elicit information that's lost when we go to "phonesets.txt", e.g. about stress
        and position ]
  questions.txt [ if you supply the questions, this file should exist. ]


lang_test/
 words.txt phones.txt silphones.csl nonsilphones.csl topo  
 phones_dismbig.txt L_disambig.txt G.fst


[for training:]
 phones.txt [for testing too?]
 phonesets.txt [ phonesets used in building questions... if not supplied, use the "base phones" ]
 extra_questions.txt [ extra questions appended to automatically generated questions.  Should ask 
        questions that elicit information that's lost when we go to "phonesets.txt", e.g. about stress
        and position ]
 questions.txt [ if you supply the questions, this file should exist. ]
 L.fst

 [for testing:]
 phones_disambig.txt
 L_disambig.fst
 G.fst

data/
 spk2utt
 utt2spk
 txt
 scp
 spk2gender


steps/train_mono.sh data.1h/ lang/ exp/mono

steps/train_tri1.sh exp/mono data.1h/ lang/ exp/mono



# This script file cannot be run as-is; some paths in it need to be changed
# before you can run it.
# Search for /path/to.
# It is recommended that you do not invoke this file from the shell, but
# run the paths one by one, by hand.

# the step in data_prep/ will need to be modified for your system.

# First step is to do data preparation:
# This just creates some text files, it is fast.
# If not on the BUT system, you would have to change run.sh to reflect
# your own paths.
#

#Example arguments to run.sh: /mnt/matylda2/data/RM, /ais/gobi2/speech/RM, /cygdrive/e/data/RM
# RM is a directory with subdirectories rm1_audio1, rm1_audio2, rm2_audio
cd data_prep
#*** You have to change the pathname below.***
./run.sh /path/to/RM
cd ..

mkdir -p data
( cd data; cp ../data_prep/{train,test*}.{spk2utt,utt2spk} . ; cp ../data_prep/spk2gender.map . )

# This next step converts the lexicon, grammar, etc., into FST format.
steps/prepare_graphs.sh

# Next, make sure that "exp/" is someplace you can write a significant amount of
# data to (e.g. make it a link to a file on some reasonably large file system).
# If it doesn't exist, the scripts below will make the directory "exp".

# mfcc should be set to some place to put training mfcc's
# where you have space.  Make sure you create the directory.
#e.g.: mfccdir=/mnt/matylda6/jhu09/qpovey/kaldi_rm_mfccb
# Note: mfccdir should be an absolute pathname
mfccdir=/path/to/mfccdir
steps/make_mfcc_train.sh $mfccdir
steps/make_mfcc_test.sh $mfccdir

steps/train_mono.sh
steps/decode_mono.sh  &
steps/train_tri1.sh
(steps/decode_tri1.sh ; steps/decode_tri1_fmllr.sh; steps/decode_tri1_regtree_fmllr.sh ; steps/decode_tri1_latgen.sh) &

steps/train_tri2a.sh
(steps/decode_tri2a.sh ; steps/decode_tri2a_fmllr.sh; steps/decode_tri2a_fmllr_utt.sh ;
 steps/decode_tri2a_dfmllr.sh;  steps/decode_tri2a_dfmllr_fmllr.sh;  
 steps/decode_tri2a_dfmllr_utt.sh; 
)&


# Then do the same for 2b, 2c, and so on
# 2a = basic triphone (all features double-deltas unless stated).
# 2b = exponential transform
# 2c = mean normalization (cmn)
# 2d = MLLT
# 2e = splice-9-frames + LDA
# 2f = splice-9-frames + LDA + MLLT
# 2g = linear VTLN (+ regular VTLN); various decode scripts available.
# 2h = splice-9-frames + HLDA
# 2i = triple-deltas + HLDA
# 2j = triple-deltas + LDA + MLLT
# 2k = LDA + ET (equiv to LDA+MLLT+ET)
# 2l = splice-9-frames + LDA + MLLT + SAT (i.e. train with CMLLR)
# 2m = splice-9-frames + LDA + MLLT + LVTLN [depends on 2f]

for group in "b c d e" "f g h i" "j k l m"; do 
  for x in $group; do
    steps/train_tri2$x.sh &
  done
  wait;
  for x in $group; do
    for y in steps/decode_tri2$x*.sh; do
     $y
    done
  done
done


# To train and test SGMM systems:



# note: if the SGMM decoding is too slow, aside from playing
# with decoder beams and max-leaves, you can set e.g.
# --full-gmm-nbest=5 to the sgmm-gselect program (default is 15, 
# so max possible speedup with this setting is 3x).  For best WER,
# this should have the
# same value in training and test ("matched training"), but
# you can get the speed improvements by just doing it in test.
# You can take this all the way down to 1 for fastest speed, although
# this will degrade results.


steps/train_ubma.sh

(steps/train_sgmma.sh; steps/decode_sgmma.sh; steps/decode_sgmma_fmllr.sh;
 steps/decode_sgmma_fmllr_utt.sh; steps/train_sgmma_fmllrbasis.sh; 
 steps/decode_sgmma_fmllrbasis_utt.sh )&

# train and test system with speaker vectors.
(steps/train_sgmmb.sh; steps/decode_sgmmb.sh; steps/decode_sgmmb_fmllr.sh; steps/decode_sgmmb_utt.sh )&

# + gender dependency.
(steps/train_ubmb.sh; steps/train_sgmmc.sh; steps/decode_sgmmc.sh; steps/decode_sgmmc_fmllr.sh )&

# as sgmmb but with LDA+STC features.
(steps/train_ubmc.sh; steps/train_sgmmd.sh; steps/decode_sgmmd.sh; steps/decode_sgmmd_fmllr.sh )&

(steps/train_ubmd.sh; steps/train_sgmme.sh; steps/decode_sgmme.sh; steps/decode_sgmme_fmllr.sh;
  steps/decode_sgmme_latgen.sh )&





