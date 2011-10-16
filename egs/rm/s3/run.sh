#!/bin/bash

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
for x in train test_mar87 test_oct87 test_feb89 test_oct89 test_feb91 test_sep92; do
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
local/decode.sh steps/decode_lda_et.sh exp/tri2c/decode

# Align all data with LDA+MLLT system (tri2b)
steps/align_lda_mllt.sh --graphs "ark,s,cs:gunzip -c exp/tri2b/graphs.fsts.gz|" \
   data/train data/lang exp/tri2b exp/tri2b_ali

#  Do MMI on top of LDA+MLLT.
steps/train_lda_etc_mmi.sh data/train data/lang exp/tri2b_ali exp/tri3a
local/decode.sh steps/decode_lda_mllt.sh exp/tri3a/decode

# Do the same with boosting.
steps/train_lda_etc_mmi.sh --boost 0.05 data/train data/lang exp/tri2b_ali exp/tri3b 
local/decode.sh steps/decode_lda_mllt.sh exp/tri3b/decode


# Do LDA+MLLT+SAT
steps/train_lda_mllt_sat.sh data/train data/lang exp/tri2b_ali exp/tri3d
local/decode.sh steps/decode_lda_mllt_sat.sh exp/tri3d/decode


# Align all data with LDA+MLLT+SAT system (tri3d)
steps/align_lda_mllt_sat.sh --graphs "ark,s,cs:gunzip -c exp/tri3d/graphs.fsts.gz|" \
    data/train data/lang exp/tri3d exp/tri3d_ali

# MMI on top of tri3d
steps/train_lda_etc_mmi.sh data/train data/lang exp/tri3d_ali exp/tri4a 
local/decode.sh steps/decode_lda_mllt_sat.sh exp/tri4a/decode

# Try another pass of training on top of 3d
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
. path.sh
sgmm-comp-prexform exp/sgmm4f/final.{mdl,occs,fmllr_mdl}
local/decode.sh steps/decode_sgmm_lda_etc_fmllr.sh exp/sgmm4f/decode_fmllr exp/sgmm4f/decode exp/tri3d/decode


# Some system combination experiments (just compose lattices).
local/decode_combine.sh steps/decode_combine.sh exp/tri1/decode exp/tri2a/decode exp/combine_1_2a/decode
local/decode_combine.sh steps/decode_combine.sh exp/sgmm4f/decode/ exp/tri3d/decode exp/combine_sgmm4f_tri3d/decode
local/decode_combine.sh steps/decode_combine.sh exp/sgmm4f/decode/ exp/tri4a/decode exp/combine_sgmm4f_tri4a/decode


### From here is semi-continuous experiments. ###
### Note: this is not yet working.***

# Train a classic semi-continuous model using {diag,full} densities
# the numeric parameters following exp/tri1-semi are: 
#   number of gaussians, something like 4096 for diag, 2048 for full
#   number of tree leaves 
#   type of suff-stats interpolation (0 regular, 1 preserves counts)
#   rho-stats, rho value for the smoothing of the statistics (0 for no smoothing)
#   rho-iters, rho value to interpolate the parameters with the last iteration (0 for no interpolation)
steps/train_semi_full.sh data/train data/lang exp/tri1_ali exp/tri1_semi 1024 2500 1 35 0.2
local/decode.sh steps/decode_tied_full.sh exp/tri1_semi/decode

# 2level full-cov training...
steps/train_2lvl.sh data/train data/lang exp/tri1_ali exp/tri1_2lvl 100 1024 1800 0 0 0

# Train a 2-lvl semi-continuous model using {diag,full} densities
# the numeric parameters following exp/tri1_2lvl are:
#   number of codebooks, typically 1-3 times number of phones, the more, the faster
#   total number of gaussians, something like 2048 for full, 4096 for diag
#   number of tree leaves
#   type of suff-stats interpolation (0 regular, 1 preserves counts)
#   rho-stats, rho value for the smoothing of the statistics (0 for no smoothing)
#   rho-iters, rho value to interpolate the parameters with the last iteration (0 for no interpolation)
steps/train_2lvl_full.sh data/train data/lang exp/tri1_ali exp/tri1_2lvl 104 2048 2500 0 1 10 0
local/decode.sh steps/decode_tied_full.sh exp/tri1_2lvl/decode



