#!/bin/bash


# call the next line with the directory where the RM data is
# (the argument below is just an example).  This should contain
# subdirectories named as follows:
#    rm1_audio1  rm1_audio2	rm2_audio
local/rm_data_prep.sh /mnt/matylda2/data/RM || exit 1;

local/rm_format_data.sh || exit 1;


### FEATURE EXTRACTION ###

# featdir should be some place with a largish disk where you
# want to store features.
featdir=$PWD/exp/kaldi_rm_feats

for x in train test_mar87 test_oct87 test_feb89 test_oct89 test_feb91 test_sep92; do
  steps/make_mfcc.sh data/$x exp/make_mfcc/$x $featdir/mfcc 2  || exit 1;
  steps/make_fbank.sh data/$x exp/make_fbank/$x $featdir/fbank 2 || exit 1;
  ln -s $PWD/data/$x/feats.scp.mfcc data/$x/feats.scp #select mfcc as default
done

scripts/subset_data_dir.sh data/train 1000 data/train.1k  || exit 1;



### TRAIN GMM SYSTEMS TO HAVE ALIGNMENTS FOR MLP TRAINING ###

# train monophone system (use reduced list).
steps/train_mono.sh data/train.1k data/lang exp/mono0a || exit 1;
# build graph
scripts/mkgraph.sh --mono data/lang_test exp/mono0a exp/mono0a/graph || exit 1;
# decode mono
local/decode.sh steps/decode_deltas.sh exp/mono0a || exit 1;
# Get alignments from mono0a system.
steps/align_deltas.sh data/train data/lang exp/mono0a exp/mono0a_ali || exit 1;


# train another monophone system
# we need good monophone system for better MLP targets
# (use full list and more gaussians)
steps/train_mono_ali.sh data/train data/lang exp/mono0a_ali exp/mono1a || exit 1;
# build graph
scripts/mkgraph.sh --mono data/lang_test exp/mono1a exp/mono1a/graph || exit 1;
# decode mono
local/decode.sh steps/decode_deltas.sh exp/mono1a || exit 1;
# Get alignments from mono1a system.
steps/align_deltas.sh data/train data/lang exp/mono1a exp/mono1a_ali || exit 1;


# train tri1 [first triphone pass], use mono0a alignments
steps/train_deltas.sh data/train data/lang exp/mono0a_ali exp/tri1 || exit 1;
# build graph
scripts/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
# decode tri1
local/decode.sh steps/decode_deltas.sh exp/tri1 || exit 1;
# align tri1
steps/align_deltas.sh --graphs "ark,s,cs:gunzip -c exp/tri1/graphs.fsts.gz|" \
    data/train data/lang exp/tri1 exp/tri1_ali || exit 1;


# train tri2a, use 1800 leaves [found to be optimal]
numleaves=1800
# train tri2a [delta+delta-deltas]
steps/train_deltas.sh --numleaves $numleaves data/train data/lang exp/tri1_ali exp/tri2a-$numleaves || exit 1;
# build graph
scripts/mkgraph.sh data/lang_test exp/tri2a-$numleaves exp/tri2a-$numleaves/graph || exit 1;
# decode tri2a
local/decode.sh steps/decode_deltas.sh exp/tri2a-$numleaves || exit 1;
# align tri2a
steps/align_deltas.sh --graphs "ark,s,cs:gunzip -c exp/tri2a-$numleaves/graphs.fsts.gz|" \
    data/train data/lang exp/tri2a-$numleaves exp/tri2a-${numleaves}_ali || exit 1;


#####################################
### HERE START THE MLP EXPERIMENTS
###


# train nnet with monophone targets [mono1a]
steps/train_nnet.sh data/train data/lang exp/mono1a_ali exp/mono1a_nnet || exit 1;
# build graph
scripts/mkgraph.sh --mono data/lang_test exp/mono1a exp/mono1a_nnet/graph || exit 1;
# decode mono
local/decode.sh "steps/decode_nnet.sh --acoustic-scale 0.176275" exp/mono1a_nnet || exit 1;
# tune acousticscale (see the oracle)
#scripts/tune_scalar.py local/decode.sh "steps/decode_nnet.sh --acoustic-scale %g" exp/mono1a_nnet 0.0 0.5 || exit 1;



# train nnets with triphone targets [tri2a-1800]
numleaves=1800
# train nnet with mono1a 
steps/train_nnet.sh data/train data/lang exp/tri2a-${numleaves}_ali exp/tri2a-${numleaves}_nnet || exit 1;
# build graph
scripts/mkgraph.sh data/lang_test exp/tri2a-${numleaves} exp/tri2a-${numleaves}_nnet/graph || exit 1;
# decode 
local/decode.sh "steps/decode_nnet.sh --acousctic-scale 0.199593"  exp/tri2a-${numleaves}_nnet || exit 1;
# tune acousticscale (see the oracle)
#scripts/tune_scalar.py local/decode.sh "steps/decode_nnet.sh --acoustic-scale %g" exp/tri2a-${numleaves}_nnet 0.0 0.5 || exit 1;



