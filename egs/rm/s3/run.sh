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

local/rm_data_prep.sh /mnt/matylda2/data/RM/

local/rm_format_data.sh

# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=/mnt/matylda6/jhu09/qpovey/kaldi_rm_mfcc
for x in train test_mar87 test_oct87 test_feb89 test_oct89 test_feb89 test_sep92; do
  steps/make_mfcc.sh data/$x exp/make_mfcc/$x $mfccdir 4
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
    data/train data/lang exp/tri1 exp/tri1_ali

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

# Decode with fMLLR
sgmm-comp-prexform exp/sgmm4f/final.{mdl,occs,fmllr_mdl}
local/decode.sh steps/decode_sgmm_lda_etc_fmllr.sh exp/sgmm4f/decode_fmllr exp/sgmm4f/decode exp/tri3d/decode


# Some system combination experiments (just compose lattices).
local/decode_combine.sh steps/decode_combine.sh exp/tri1/decode exp/tri2a/decode exp/combine_1_2a/decode
local/decode_combine.sh steps/decode_combine.sh exp/sgmm4f/decode/ exp/tri3d/decode exp/combine_sgmm4f_tri3d/decode

for x in exp/*/decode*; do grep WER $x/wer_* | scripts/best_wer.sh; done

exp/combine_1_2a/decode/wer_7:%WER 3.399027 [ 426 / 12533, 55 ins, 94 del, 277 sub ]
exp/combine_sgmm4f_tri3d/decode/wer_5:%WER 1.731429 [ 217 / 12533, 30 ins, 43 del, 144 sub ]
exp/mono/decode/wer_6:%WER 10.340701 [ 1296 / 12533, 95 ins, 391 del, 810 sub ]
exp/sgmm3d/decode/wer_5:%WER 2.267284 [ 284 / 12526, 38 ins, 51 del, 195 sub ]
exp/sgmm3e/decode/wer_6:%WER 2.122397 [ 266 / 12533, 37 ins, 51 del, 178 sub ]
exp/sgmm4f/decode/wer_4:%WER 1.795261 [ 225 / 12533, 45 ins, 37 del, 143 sub ]
exp/sgmm4f/decode_fmllr/wer_5:%WER 1.771324 [ 222 / 12533, 38 ins, 42 del, 142 sub ]
exp/tri1/decode/wer_6:%WER 3.566584 [ 447 / 12533, 74 ins, 88 del, 285 sub ]
exp/tri2a/decode/wer_7:%WER 3.518711 [ 441 / 12533, 57 ins, 91 del, 293 sub ]
exp/tri2b/decode/wer_9:%WER 3.614458 [ 453 / 12533, 59 ins, 111 del, 283 sub ]
exp/tri2c/decode/wer_6:%WER 2.833653 [ 355 / 12528, 54 ins, 71 del, 230 sub ]
exp/tri3d/decode/wer_7:%WER 2.489428 [ 312 / 12533, 43 ins, 63 del, 206 sub ]
exp/tri4d/decode/wer_7:%WER 2.649007 [ 332 / 12533, 53 ins, 67 del, 212 sub ]

local/decode_combine.sh steps/decode_combine.sh exp/tri1/decode exp/tri2a/decode exp/combine_tri3d_sgmm4f

