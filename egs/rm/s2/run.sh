#!/bin/bash


# call the next line with the directory where the RM data is
# (the argument below is just an example).  This should contain
# subdirectories named as follows:
#    rm1_audio1  rm1_audio2	rm2_audio
local/rm_data_prep.sh /mnt/matylda2/data/RM || exit 1;

local/rm_format_data.sh || exit 1;

# featdir should be some place with a largish disk where you
# want to store features.
featdir=$PWD/exp/kaldi_rm_feats

for x in train test_mar87 test_oct87 test_feb89 test_oct89 test_feb91 test_sep92; do
  steps/make_mfcc.sh data/$x exp/make_mfcc/$x $featdir/mfcc 2  || exit 1;
  steps/make_plp.sh data/$x exp/make_plp/$x $featdir/plp 2 || exit 1;
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

# train tri2a with different number of leaves
numleavesL=(0500 1000 1200 1400 1600 1800 2000 2200 2400)
for numleaves in ${numleavesL[@]}; do
  # train tri2a [delta+delta-deltas]
  steps/train_deltas.sh --numleaves ${numleaves##0} data/train data/lang exp/tri1_ali exp/tri2a-$numleaves || exit 1;
  # build graph
  scripts/mkgraph.sh data/lang_test exp/tri2a-$numleaves exp/tri2a-$numleaves/graph || exit 1;
  # decode tri2a
  local/decode.sh steps/decode_deltas.sh exp/tri2a-$numleaves || exit 1;
  # align tri2a
  steps/align_deltas.sh --graphs "ark,s,cs:gunzip -c exp/tri2a-$numleaves/graphs.fsts.gz|" \
      data/train data/lang exp/tri2a-$numleaves exp/tri2a-${numleaves}_ali || exit 1;
done

#####################################
### HERE START THE MLP EXPERIMENTS
###

# train nnet with mono1a 
steps/train_nnet.sh data/train data/lang exp/mono1a_ali exp/mono1a_nnet || exit 1;
# build graph
scripts/mkgraph.sh --mono data/lang_test exp/mono1a exp/mono1a_nnet/graph || exit 1;
# decode mono
local/decode.sh "steps/decode_nnet.sh --acoustic-scale 0.176275"  exp/mono1a_nnet || exit 1;
# tune acousticscale
scripts/tune_scalar.py local/decode.sh "steps/decode_nnet.sh --acoustic-scale %g" exp/mono1a_nnet 0.0 0.5 || exit 1;



# train nnets with tri2a-????
numleavesL=(0500 1000 1200 1400 1600 1800 2000 2200)
for numleaves in ${numleavesL[@]}; do
  # train nnet with tri2a-???? labels
  steps/train_nnet.sh data/train data/lang exp/tri2a-${numleaves}_ali exp/tri2a-${numleaves}_nnet || exit 1;
  # build graph
  scripts/mkgraph.sh data/lang_test exp/tri2a-${numleaves} exp/tri2a-${numleaves}_nnet/graph || exit 1;
  # decode 
  local/decode.sh steps/decode_nnet.sh exp/tri2a-${numleaves}_nnet || exit 1;
done
# tune acousticscale
for numleaves in ${numleavesL[@]}; do
  scripts/tune_scalar.py local/decode.sh "steps/decode_nnet.sh --acoustic-scale %g" exp/tri2a-${numleaves}_nnet 0.0 0.5 || exit 1;
done



#tune the prior scale for best system
#THIS DID NOT HELP AT ALL (it worked for Navdeep on TIMIT)
numleaves=1800
priorscaleL=(0.5 0.8 1.0 1.2 1.4 1.6 1.8 2.0)
for priorscale in ${priorscaleL[@]}; do
  dir=exp/tri2a-${numleaves}_nnet_priorscale$priorscale
  mkdir $dir
  cp -r -p exp/tri2a-${numleaves}_nnet/* "$dir"
  scripts/tune_scalar.py local/decode.sh "steps/decode_nnet.sh --acoustic-scale %g --prior-scale $priorscale" $dir 0.0 0.5 || exit 1;
done



#try more parameters in MLP
#train nnet with tri-2a alignments
#=>Optimal is 1000000,
#  the 2000000 did not improve WER, 
#  711000 is the number of GMM parameters
numleaves=1800
modelsizeL=(711000 1000000 1500000 2000000 2500000 3000000)
for modelsize in ${modelsizeL[@]}; do
  steps/train_nnet.sh --model-size $modelsize data/train data/lang exp/tri2a-${numleaves}_ali exp/tri2a-${numleaves}_nnet_modelsize$modelsize || exit 1;
  # build graph
  scripts/mkgraph.sh data/lang_test exp/tri2a-${numleaves} exp/tri2a-${numleaves}_nnet_modelsize$modelsize/graph || exit 1;
  # decode 
  local/decode.sh steps/decode_nnet.sh exp/tri2a-${numleaves}_nnet_modelsize$modelsize || exit 1;
done
# tune acousticscale
for modelsize in ${modelsizeL[@]}; do
  scripts/tune_scalar.py local/decode.sh "steps/decode_nnet.sh --acoustic-scale %g" exp/tri2a-${numleaves}_nnet_modelsize$modelsize 0.0 0.5 || exit 1;
done



#tune L2 regularization constant for best system
#=>There is a frame accuracy improvement for 1e-5, but WER is worse
numleaves=1800
l2penaltyL=(0.0 1e-6 1e-5 1e-4)
for l2penalty in ${l2penaltyL[@]}; do
  steps/train_nnet.sh --l2-penalty $l2penalty data/train data/lang exp/tri2a-${numleaves}_ali exp/tri2a-${numleaves}_nnet_l2-$l2penalty || exit 1;
  # build graph
  scripts/mkgraph.sh data/lang_test exp/tri2a-${numleaves} exp/tri2a-${numleaves}_nnet_l2-$l2penalty/graph || exit 1;
  # decode 
  local/decode.sh steps/decode_nnet.sh exp/tri2a-${numleaves}_nnet_l2-$l2penalty || exit 1;
done
# tune acousticscale
for l2penalty in ${l2penaltyL[@]}; do
  scripts/tune_scalar.py local/decode.sh "steps/decode_nnet.sh --acoustic-scale %g" exp/tri2a-${numleaves}_nnet_l2-$l2penalty 0.0 0.5 || exit 1;
done



#decding with no softmax in the MLP
#=>THIS LEADS TO IDENTICAL RESULTS, AND IS FASTER WITH NON-GPU MACHINES
dir=exp/tri2a-${numleaves}_nnet_nosoftmax
mkdir $dir
cp -r -p exp/tri2a-${numleaves}_nnet/* $dir
scripts/tune_scalar.py local/decode.sh "steps/decode_nnet_nosoftmax.sh --acoustic-scale %g" "$dir" 0.0 0.5 || exit 1;


