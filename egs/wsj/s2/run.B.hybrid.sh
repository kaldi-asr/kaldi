#!/bin/bash

# you can change these commands to just run.pl to make them run
# locally, but in that case you should change the num-jobs to
# the #cpus on your machine or fewer.

#1)BUT
decode_cmd="queue.pl -q all.q@@blade -l ram_free=1200M,mem_free=1200M"
train_cmd="queue.pl -q all.q@@blade -l ram_free=700M,mem_free=700M"
#cuda_cmd="queue.pl -q long.q@@pco203 -l gpu=1"
cuda_cmd="queue.pl -q long.q@pcspeech-gpu"
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



######################################################
###         HERE START THE MLP EXPERIMENTS         ###
######################################################

# - train the MLPs..., use the si84 set (~15h)
# - for all setupts use default feature extraction FBANK->cmn->splice->Hamming->DCT->rescale
# - for the model selection use dev93 as held-out set (cross-validation)



######################################################
###    SELECT MOST CONVENIENT TRAINING TARFETS     ###
######################################################

# - for a fair comparison we use 4-layer MLP with 3 million parameters
# - we will compare 4 different training targets (baseline systems):
#     A: mono1a, B: tri2a, C: tri2b, D: tri3b


# ### A : mono1a labels, 4-layer MLP, 3M params, lrate, si84 ### #
hidlayers=2
modelsize=3000000
lrate=0.002
( 
  dir=exp/mono1a_nnet4L_3M
  ali=exp/mono1a_ali
  # Train
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet.sh --hid-layers $hidlayers --model-size $modelsize --lrate $lrate --bunchsize 256 \
    data-fbank/train_si84 data-fbank/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || exit 1;
  # Decode
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh --mono data/lang_test_tgpr $dir $dir/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.17" steps/decode_nnet.sh $dir/graph_tgpr data-fbank/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.17" steps/decode_nnet.sh $dir/graph_tgpr data-fbank/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
) &

# ### B : tri2a labels, 4-layer MLP, 3M params, lrate, si84 ### #
hidlayers=2
modelsize=3000000
lrate=0.002
( dir=exp/tri2a_nnet4L_3M
  ali=exp/tri2a_ali
  # Train
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet.sh --hid-layers $hidlayers --model-size $modelsize --lrate $lrate --bunchsize 256 \
    data-fbank/train_si84 data-fbank/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || exit 1;
  # Decode
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr $dir $dir/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data-fbank/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data-fbank/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
) &

# ### C : tri2b labels, 4-layer MLP, 3M params, lrate, si84 ### #
hidlayers=2
modelsize=3000000
lrate=0.002
( dir=exp/tri2b_nnet4L_3M
  ali=exp/tri2b_ali
  # Train
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet.sh --hid-layers $hidlayers --model-size $modelsize --lrate $lrate --bunchsize 256 \
    data-fbank/train_si84 data-fbank/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || exit 1;
  # Decode
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr $dir $dir/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data-fbank/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data-fbank/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
) &

# ### D : tri3b labels, 4-layer MLP, 3M params, lrate, si84 ### #
hidlayers=2
modelsize=3000000
lrate=0.002
( dir=exp/tri3b_nnet4L_3M
  ali=exp/tri3b_ali
  # Train 
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet.sh --num-layers $hidlayers --model-size $modelsize --lrate $lrate --bunchsize 256 \
    data-fbank/train_si84 data-fbank/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || exit 1;
  # Decode
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr $dir $dir/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data-fbank/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data-fbank/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
) &


wait #for (A+B+C+D)


exit 0


######################################################
###        USE MORE TRAINING DATA (SI-284)         ###
######################################################

# ### E : tri2a labels, 4-layer MLP, 3M params, lrate, si284 ### #
# use full training set si284
hidlayers=2
modelsize=3000000
lrate=0.002
{
  dir=exp/tri2a_nnet4L_3M_si284
  ali=exp/tri2a_ali
  # Train
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet.sh --hid-layers $hidlayers --model-size $modelsize --lrate $lrate --bunchsize 256 \
    data-fbank/train_si284 data-fbank/test_dev93 data/lang ${ali}_si284 ${ali}_dev93 $dir || exit 1;
  # Decode
  (
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr $dir $dir/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data-fbank/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data-fbank/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
  ) &
  # Re-align 
  ($cuda_cmd $dir/_align_nnet.log \
    steps/align_nnet.sh data-fbank/train_si284 data/lang $dir ${dir}_ali_si284)&
  ($cuda_cmd $dir/_align_nnet.log \
    steps/align_nnet.sh data-fbank/test_dev93 data/lang $dir ${dir}_ali_dev93)&
  wait
}



######################################################
###                RE-ALIGN BY MLP                 ###
######################################################

# ### F : tri2a labels, 4-layer MLP, 3M params, lrate, si284, realign ### #
# train on the MLP re-aligned training targets
hidlayers=2
modelsize=3000000
lrate=0.002
{
  dir=exp/tri2a_nnet4L_3M_si284_iter2
  ali=exp/tri2a_nnet4L_3M_si284_ali
  # Train
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet.sh --hid-layers $hidlayers --model-size $modelsize --lrate $lrate --bunchsize 256 \
    data-fbank/train_si284 data-fbank/test_dev93 data/lang ${ali}_si284 ${ali}_dev93 $dir || exit 1;
  # Decode
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr $dir $dir/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data-fbank/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data-fbank/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
}


