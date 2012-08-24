#!/bin/bash

# you can change these commands to just run.pl to make them run
# locally, but in that case you should change the num-jobs to
# the #cpus on your machine or fewer.

#1)BUT
decode_cmd="queue.pl -q all.q@@blade -l ram_free=1200M,mem_free=1200M"
train_cmd="queue.pl -q all.q@@blade -l ram_free=700M,mem_free=700M"
cuda_cmd="queue.pl -q long.q@@pco203 -l gpu=1"
#cuda_cmd="queue.pl -q long.q@pcspeech-gpu"
mkgraph_cmd="queue.pl -q all.q@@servers -l ram_free=4G,mem_free=4G"

#2)JHU
#export train_cmd="queue.pl -q all.q@a*.clsp.jhu.edu -S /bin/bash"
#export decode_cmd="queue.pl -q all.q@a*.clsp.jhu.edu -S /bin/bash"

#3)LOCAL
#export train_cmd=run.pl
#export decode_cmd=run.pl
#export cuda_cmd=run.pl
#export mkgraph_cmd=run.pl



# put the scripts to path
source path.sh

featdir=$PWD/exp/kaldi_wsj_feats

######################################################
###       HERE START THE TANDEM EXPERIMENTS        ###
######################################################

#TANDEM SYSTEM "A"
#- linear BN-features from shallow nnet, with monophone targets
#  - 6 layers, 3 million parameters
#- use SI-84
#
# 1) Train the nnet
dir=exp/mono1a_nnet-linBN-6L
ali=exp/mono1a_ali
bn_dim=30
hidlayers=2
$cuda_cmd $dir/_train_nnet.log \
  steps/train_nnet.sh --hid-layers $hidlayers --bn-dim $bn_dim --lrate 0.0001 --model-size 3000000 --bunchsize 256 \
  data-fbank/train_si84 data-fbank/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || exit 1;
#Dump the BN-features
nndir=$dir
bnroot=$PWD/exp/make_bnfeats_$(basename $dir)
for x in test_eval92 test_eval93 test_dev93 train_si84; do
  steps/make_bnfeats.sh --bn-dim $bn_dim data/$x $dir $bnroot/$x $featdir/bnfeats_$(basename $nndir) 4
done


# 2a) Train GMMs on the BN-features (mixing-up from the tri2b alignments)
#### Train as ``tri2b'', which is LDA+MLLT, on si84 data.
# -I. LDA
# -II. training with occassional MLLT/alignment updates
dir=${nndir}_gmm
ali=exp/tri2b_ali
# Train
steps/train_bnfeats.sh --num-jobs 10 --cmd "$train_cmd" \
   2500 15000 $bnroot/train_si84/ data/lang ${ali}_si84 $dir || exit 1;
# Decode
$mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr $dir $dir/graph_tgpr || exit 1;
scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_bnfeats.sh $dir/graph_tgpr $bnroot/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_bnfeats.sh $dir/graph_tgpr $bnroot/test_dev93 $dir/decode_tgpr_dev93 || exit 1;


# 2b) Re-train the GMMs to new feature space (single pass retraining)
# -I. LDA
# -II. single pass retrining on BN-LDA features
# -III. training with occassional MLLT/alignment updates
dir=${nndir}_gmm_singlepass
ali=exp/tri2b_ali
# Train
steps/train_bnfeats_singlepass.sh --num-jobs 10 --cmd "$train_cmd" \
   2500 15000 $bnroot/train_si84 data-fbank/train_si84/ data/lang ${ali}_si84 $dir || exit 1;
# Decode
$mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr $dir $dir/graph_tgpr || exit 1;
scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_bnfeats.sh $dir/graph_tgpr $bnroot/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_bnfeats.sh $dir/graph_tgpr $bnroot/test_dev93 $dir/decode_tgpr_dev93 || exit 1;


# 3) Two stream system
# Re-train the GMMs using 2-feature-streams (MFCC+LDA+MLLT, BNFEATS+LDA+MLLT) (single pass retraining)
# - using fixed transforms from previous systems
dir=${nndir}_gmm_2stream_singlepass
ali=exp/tri2b_ali
tnd=${nndir}_gmm_singlepass #get dir 1-stream tandem experiment
# Train
steps/train_2stream_singlepass.sh --num-jobs 10 --cmd "$train_cmd" \
   2500 15000 $bnroot/train_si84 data/train_si84/ data/lang ${ali}_si84 $tnd $dir || exit 1;
# Decode
$mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr $dir $dir/graph_tgpr || exit 1;
scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_2stream.sh $dir/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92 $bnroot/test_eval92 || exit 1;
scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_2stream.sh $dir/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93 $bnroot/test_dev93 || exit 1;


