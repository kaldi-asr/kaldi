#!/bin/bash


# call the next line with the directory where the RM data is
# (the argument below is just an example).  This should contain
# subdirectories named as follows:
#    rm1_audio1  rm1_audio2	rm2_audio

local/rm_data_prep.sh /mnt/matylda2/data/RM || exit 1;

local/rm_format_data.sh || exit 1;

# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
featdir=/mnt/matylda6/jhu09/qpovey/kaldi_rm_feats

for x in train test_mar87 test_oct87 test_feb89 test_oct89 test_feb91 test_sep92; do
  steps/make_mfcc.sh data/$x exp/make_mfcc/$x $featdir 4  || exit 1;
  #steps/make_plp.sh data/$x exp/make_plp/$x $featdir 4
done

scripts/subset_data_dir.sh data/train 1000 data/train.1k  || exit 1;

# train monophone system.
steps/train_mono.sh data/train.1k data/lang exp/mono  || exit 1;


local/decode.sh --mono steps/decode_deltas.sh exp/mono/decode || exit 1;


# Get alignments from monophone system.
steps/align_deltas.sh data/train data/lang exp/mono exp/mono_ali || exit 1;

# train tri1 [first triphone pass]
steps/train_deltas.sh data/train data/lang exp/mono_ali exp/tri1 || exit 1;
# decode tri1
local/decode.sh steps/decode_deltas.sh exp/tri1/decode || exit 1;

# align tri1
steps/align_deltas.sh --graphs "ark,s,cs:gunzip -c exp/tri1/graphs.fsts.gz|" \
    data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

# train tri2a [delta+delta-deltas]
steps/train_deltas.sh data/train data/lang exp/tri1_ali exp/tri2a || exit 1;
# decode tri2a
local/decode.sh steps/decode_deltas.sh exp/tri2a/decode || exit 1;

# train tri2b [LDA+MLLT]
steps/train_lda_mllt.sh data/train data/lang exp/tri1_ali exp/tri2b || exit 1;
# decode tri2b
local/decode.sh steps/decode_lda_mllt.sh exp/tri2b/decode || exit 1;

# Train and test ET.
steps/train_lda_et.sh data/train data/lang exp/tri1_ali exp/tri2c || exit 1;
local/decode.sh steps/decode_lda_et.sh exp/tri2c/decode || exit 1;

# Align all data with LDA+MLLT system (tri2b)
steps/align_lda_mllt.sh --graphs "ark,s,cs:gunzip -c exp/tri2b/graphs.fsts.gz|" \
   data/train data/lang exp/tri2b exp/tri2b_ali || exit 1;

#  Do MMI on top of LDA+MLLT.
steps/train_lda_etc_mmi.sh data/train data/lang exp/tri2b_ali exp/tri3a || exit 1;
local/decode.sh steps/decode_lda_mllt.sh exp/tri3a/decode || exit 1;

# Do the same with boosting.
steps/train_lda_etc_mmi.sh --boost 0.05 data/train data/lang exp/tri2b_ali exp/tri3b  || exit 1;
local/decode.sh steps/decode_lda_mllt.sh exp/tri3b/decode || exit 1;

# An experiment with MCE.
steps/train_lda_etc_mce.sh data/train data/lang exp/tri2b_ali exp/tri3c || exit 1;
local/decode.sh steps/decode_lda_mllt.sh exp/tri3c/decode || exit 1;


# Do LDA+MLLT+SAT
steps/train_lda_mllt_sat.sh data/train data/lang exp/tri2b_ali exp/tri3d || exit 1;
local/decode.sh steps/decode_lda_mllt_sat.sh exp/tri3d/decode || exit 1;


# Align all data with LDA+MLLT+SAT system (tri3d)
steps/align_lda_mllt_sat.sh --graphs "ark,s,cs:gunzip -c exp/tri3d/graphs.fsts.gz|" \
    data/train data/lang exp/tri3d exp/tri3d_ali || exit 1;

# MMI on top of tri3d
steps/train_lda_etc_mmi.sh data/train data/lang exp/tri3d_ali exp/tri4a  || exit 1;
local/decode.sh steps/decode_lda_mllt_sat.sh exp/tri4a/decode || exit 1;

# Try another pass of training on top of 3d
steps/train_lda_mllt_sat.sh data/train data/lang exp/tri3d_ali exp/tri4d || exit 1;
scripts/mkgraph.sh data/lang_test exp/tri4d exp/tri4d/graph || exit 1;
local/decode.sh steps/decode_lda_mllt_sat.sh exp/tri4d/decode || exit 1;

# Next, SGMM system-- train SGMM system with speaker vectors, on top 
# of LDA+MLLT features.
steps/train_ubm_lda_etc.sh 400 data/train data/lang exp/tri2b_ali exp/ubm3d || exit 1;
steps/train_sgmm_lda_etc.sh data/train data/lang exp/tri2b_ali exp/ubm3d/final.ubm exp/sgmm3d || exit 1;

scripts/mkgraph.sh data/lang_test exp/sgmm3d exp/sgmm3d/graph || exit 1;
local/decode.sh steps/decode_sgmm_lda_etc.sh exp/sgmm3d/decode || exit 1;

# Align LDA+ET system prior to training corresponding SGMM system.
steps/align_lda_et.sh --graphs "ark,s,cs:gunzip -c exp/tri2c/graphs.fsts.gz|" \
  data/train data/lang exp/tri2c exp/tri2c_ali  || exit 1;

# Train SGMM system on top of LDA+ET.
steps/train_ubm_lda_etc.sh 400 data/train data/lang exp/tri2c_ali exp/ubm3e || exit 1;
steps/train_sgmm_lda_etc.sh data/train data/lang exp/tri2c_ali exp/ubm3e/final.ubm exp/sgmm3e || exit 1;

local/decode.sh steps/decode_sgmm_lda_etc.sh exp/sgmm3e/decode exp/tri2c/decode || exit 1;

# Now train SGMM system on top of LDA+MLLT+SAT
steps/train_ubm_lda_etc.sh 400 data/train data/lang exp/tri3d_ali exp/ubm4f || exit 1;
steps/train_sgmm_lda_etc.sh data/train data/lang exp/tri3d_ali exp/ubm4f/final.ubm exp/sgmm4f || exit 1;
local/decode.sh steps/decode_sgmm_lda_etc.sh exp/sgmm4f/decode exp/tri3d/decode || exit 1;


# Decode with fMLLR
. ./path.sh
sgmm-comp-prexform exp/sgmm4f/final.{mdl,occs,fmllr_mdl} || exit 1;
local/decode.sh steps/decode_sgmm_lda_etc_fmllr.sh exp/sgmm4f/decode_fmllr exp/sgmm4f/decode exp/tri3d/decode || exit 1;


# Some system combination experiments (just compose lattices).
local/decode_combine.sh steps/decode_combine.sh exp/tri1/decode exp/tri2a/decode exp/combine_1_2a/decode || exit 1;
local/decode_combine.sh steps/decode_combine.sh exp/sgmm4f/decode/ exp/tri3d/decode exp/combine_sgmm4f_tri3d/decode || exit 1;
local/decode_combine.sh steps/decode_combine.sh exp/sgmm4f/decode/ exp/tri4a/decode exp/combine_sgmm4f_tri4a/decode || exit 1;

### From here is semi-continuous experiments. ###
### Note: this is not yet working.  Do not run this.  ***
echo "semi-continuous code not finalized" && exit 1;

# Train a classic semi-continuous model using {diag,full} densities
# the numeric parameters following exp/tri1-semi are: 
#   number of gaussians, something like 4096 for diag, 2048 for full
#   number of tree leaves 
#   type of suff-stats interpolation (0 regular, 1 preserves counts)
#   rho-stats, rho value for the smoothing of the statistics (0 for no smoothing)
#   rho-iters, rho value to interpolate the parameters with the last iteration (0 for no interpolation)

steps/train_ubm_lda_etc.sh 1024 data/train data/lang exp/tri2b_ali exp/ubm3f
steps/train_lda_mllt_semi_full.sh data/train data/lang exp/tri2b_ali exp/ubm3f/final.ubm exp/tiedfull3f 2500 1 35 0.2

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


# note on new gselect:
# gmm-gselect --n=50 "sgmm-write-ubm exp/sgmm3d/final.mdl - | fgmm-global-to-gmm - - |" 'ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:data/train/utt2spk ark:exp/tri2b_ali/cmvn.ark scp:data/train/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats exp/sgmm3d/final.mat ark:- ark:- |' ark,t:-  | fgmm-gselect --n=15 "sgmm-write-ubm exp/sgmm3d/final.mdl -|" 'ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:data/train/utt2spk ark:exp/tri2b_ali/cmvn.ark scp:data/train/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats exp/sgmm3d/final.mat ark:- ark:- |' ark:- ark,t:-  | head -1 > f2
