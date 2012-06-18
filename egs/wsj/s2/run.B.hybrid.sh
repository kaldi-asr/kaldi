#!/bin/bash

# you can change these commands to just run.pl to make them run
# locally, but in that case you should change the num-jobs to
# the #cpus on your machine or fewer.
decode_cmd="queue.pl -q all.q@@blade -l ram_free=1200M,mem_free=1200M"
train_cmd="queue.pl -q all.q@@blade -l ram_free=700M,mem_free=700M"
cuda_cmd="queue.pl -q long.q@@pco203 -l gpu=1"
mkgraph_cmd="queue.pl -q all.q@@servers -l ram_free=4G,mem_free=4G"

# put the scripts to path
source path.sh



######################################################
###         HERE START THE MLP EXPERIMENTS         ###
######################################################

# train the MLPs..., use the si84 set (~15h)
# for all setupts use default feature extraction FBANK->cmn->splice->Hamming->DCT->rescale

# ### A : mono1a labels ### #
# train nnet with mono1a labels
dir=exp/mono1a_nnet
ali=exp/mono1a_ali 
$cuda_cmd $dir/_train_nnet.log \
  steps/train_nnet.sh --lrate 0.001 --bunchsize 256 data/train_si84 data/lang ${ali}_si84 $dir || exit 1; 
# build graph
$mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh --mono data/lang_test_tgpr exp/mono1a $dir/graph_tgpr || exit 1;
# decode mono
scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.17" steps/decode_nnet.sh $dir/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.17" steps/decode_nnet.sh $dir/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92 || exit 1;



# ### A2 : mono1a labels ### #
# train nnet with mono1a labels, use dev93 as Cross-Validation set
dir=exp/mono1a_nnet_dev93
ali=exp/mono1a_ali 
$cuda_cmd $dir/_train_nnet.log \
  steps/train_nnet_dev.sh --lrate 0.001 --bunchsize 256 data/train_si84 data/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || exit 1; 
# build graph
$mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh --mono data/lang_test_tgpr exp/mono1a $dir/graph_tgpr || exit 1;
# decode mono
scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.17" steps/decode_nnet.sh $dir/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.17" steps/decode_nnet.sh $dir/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92 || exit 1;



# ### B : tri2a labels ### #
# train nnet with tri2a labels
numleavesL=(2500)
for numleaves in ${numleavesL[@]}; do
  dir=exp/tri2a-${numleaves}_nnet
  ali=exp/tri2a-${numleaves}_ali
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet_dev.sh --lrate 0.001 --bunchsize 256 data/train_si84 data/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || exit 1;
  # build graph
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr exp/tri2a-${numleaves} $dir/graph_tgpr || exit 1;
  # decode 
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
done
  


# ### C : tri2b labels ### #
# train nnet with tri2b labels
numleavesL=(2500)
for numleaves in ${numleavesL[@]}; do
  dir=exp/tri2b-${numleaves}_nnet
  ali=exp/tri2b-${numleaves}_ali
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet_dev.sh --lrate 0.001 --bunchsize 256 data/train_si84 data/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || exit 1;
  # build graph
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr exp/tri2b-${numleaves} $dir/graph_tgpr || exit 1;
  # decode 
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
done



# ### D : tri2a labels, 4-layer MLP ### #
# use better alignments tri2a
numleavesL=(2500)
for numleaves in ${numleavesL[@]}; do
  dir=exp/tri2a-${numleaves}_nnet4L
  ali=exp/tri2a-${numleaves}_ali
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet_dev_MLP4.sh --lrate 0.001 --bunchsize 256 data/train_si84 data/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || exit 1;
  # build graph
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr exp/tri2a-${numleaves} $dir/graph_tgpr || exit 1;
  # decode 
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
done



# ### E : tri2a labels, 4-layer MLP, 3M params, lrate ### #
# try different learning rates
numleaves=2500
lrateL=(0.0005 0.001 0.002 0.004 0.008)
for lrate in ${lrateL[@]}; do
( dir=exp/tri2a-${numleaves}_nnet4L_lrate${lrate}
  ali=exp/tri2a-${numleaves}_ali
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet_dev_MLP4.sh --lrate $lrate --bunchsize 256 data/train_si84 data/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || exit 1;
  # build graph
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr exp/tri2a-${numleaves} $dir/graph_tgpr || exit 1;
  # decode 
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
) &
done

wait



# ### F : tri2a labels, 4-layer MLP, more params ### #
# add more parameters to the MLP4L
numleaves=2500
lrate=0.002
modelsizeL=(1000000 1500000 2000000 2500000 3000000)
for modelsize in ${modelsizeL[@]}; do
( dir=exp/tri2a-${numleaves}_nnet4L_lrate${lrate}_modelsize${modelsize}
  ali=exp/tri2a-${numleaves}_ali
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet_dev_MLP4.sh --model-size $modelsize --lrate $lrate --bunchsize 256 data/train_si84 data/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || exit 1;
  # build graph
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr exp/tri2a-${numleaves} $dir/graph_tgpr || exit 1;
  # decode 
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
) &
done



# ### G : tri2a labels, 4-layer MLP, 3M params, lrate, si284 ### #
# use full training set si284
numleaves=2500
modelsize=3000000
lrate=0.002
( dir=exp/tri2a-${numleaves}_nnet4L_lrate${lrate}_modelsize${modelsize}_si284
  ali=exp/tri2a-${numleaves}_ali
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet_dev_MLP4.sh --model-size $modelsize --lrate $lrate --bunchsize 256 data/train_si284 data/test_dev93 data/lang ${ali}_si284 ${ali}_dev93 $dir || exit 1;
  # build graph
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr exp/tri2a-${numleaves} $dir/graph_tgpr || exit 1;
  # decode 
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
  # re-align 
  $cuda_cmd $dir/_align_nnet.log \
    steps/align_nnet.sh data/train_si284 data/lang $dir ${dir}_ali_si284 || exit 1;
  $cuda_cmd $dir/_align_nnet.log \
    steps/align_nnet.sh data/test_dev93 data/lang $dir ${dir}_ali_dev93 || exit 1;
) &


wait #we need the alignments from previous step (G)!


# ### H : tri2a labels, 4-layer MLP, 3M params, lrate, si284, realign ### #
#Train so far best working pure hybrid system on big dataset si284, realign
numleaves=2500
modelsize=3000000
lrate=0.002
( dir=exp/tri2a-${numleaves}_nnet4L_lrate${lrate}_modelsize${modelsize}_si284_iter2
  ali=exp/tri2a-${numleaves}_nnet4L_lrate${lrate}_modelsize${modelsize}_si284_ali
  #### hack-in the GMM models:
  mkdir -p $dir; cp exp/tri2a-2500/final.mdl $dir/final.mdl
  ####

  # train
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet_dev_MLP4.sh --model-size $modelsize --lrate $lrate data/train_si284 data/test_dev93 data/lang ${ali}_si284 ${ali}_dev93 $dir || exit 1;
  # build graph
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr exp/tri2a-${numleaves} $dir/graph_tgpr || exit 1;
  # decode 
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
) &

wait


