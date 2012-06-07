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

featdir=$PWD/exp/kaldi_wsj_feats


######################################################
###       HERE START THE TANDEM EXPERIMENTS        ###
######################################################

#TODO


#TANDEM SYSTEM "A"
#trained on linear BN-features, on monophone targets
#
#Train the nnet
$cuda_cmd exp/mono1a_nnet-linBN-5L/_train_nnet.log \
  steps/train_nnet_MLP5-linBN.sh --lrate 0.00025 data/train_si84 data/lang exp/mono1a_ali_si84 exp/mono1a_nnet-linBN-5L || exit 1;
#Dump the BN-features
nndir=exp/mono1a_nnet-linBN-5L
bnroot=$PWD/exp/make_bnfeats_$(basename $nndir)
for x in test_eval92 test_eval93 test_dev93 train_si84; do
  steps/make_bnfeats.sh --bn-dim 30 data/$x $nndir $bnroot/$x $featdir/bnfeats_$(basename $nndir) 4
done


#Re-train the GMMs to new feature space (from scratch)
#### Train ``tri2b'', which is LDA+MLLT, on si84 data.
numleavesL=(2500)
for numleaves in ${numleavesL[@]}; do
  # Train
  steps/train_lda_mllt_bnfeats.sh --num-jobs 10 --cmd "$train_cmd" \
     $numleaves 15000 $bnroot/train_si84/ data/lang exp/tri2b-${numleaves}_ali_si84 exp/mono1a_nnet-linBN-5L.gmm-$numleaves || exit 1;
  # Decode
  scripts/mkgraph.sh data/lang_test_tgpr exp/mono1a_nnet-linBN-5L.gmm-$numleaves exp/mono1a_nnet-linBN-5L.gmm-$numleaves/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_nnet-linBN-5L.gmm-$numleaves/graph_tgpr $bnroot/test_eval92 exp/mono1a_nnet-linBN-5L.gmm-$numleaves/decode_tgpr_eval92 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_nnet-linBN-5L.gmm-$numleaves/graph_tgpr $bnroot/test_dev93 exp/mono1a_nnet-linBN-5L.gmm-$numleaves/decode_tgpr_dev93 || exit 1;
done


#Re-train the GMMs to new feature space (single pass retraining)
numleavesL=(2500)
for numleaves in ${numleavesL[@]}; do
  # Train
  steps/train_lda_mllt_bnfeats_singlepass.sh --num-jobs 10 --cmd "$train_cmd" \
     $numleaves 15000 $bnroot/train_si84 data/train_si84/ data/lang exp/tri2b-${numleaves}_ali_si84 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass || exit 1;
  # Decode
  scripts/mkgraph.sh data/lang_test_tgpr exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass/graph_tgpr $bnroot/test_eval92 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass/decode_tgpr_eval92 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass/graph_tgpr $bnroot/test_dev93 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass/decode_tgpr_dev93 || exit 1;
done


#DIFFERENT MODIFICATIONS OF GMM-INPUT:

#remove LDA + MLLT
#exp/mono1a_nnet-linBN-5L.gmm-2500_singlepass_notsf/decode_tgpr_dev93/wer_20:%WER 36.09 [ 2972 / 8234, 643 ins, 200 del, 2129 sub ]
#exp/mono1a_nnet-linBN-5L.gmm-2500_singlepass_notsf/decode_tgpr_eval92/wer_20:%WER 20.36 [ 1149 / 5643, 265 ins, 64 del, 820 sub ]
#Re-train the GMMs to new feature space (single pass retraining)
numleavesL=(2500)
for numleaves in ${numleavesL[@]}; do
  # Train
  steps/train_lda_mllt_bnfeats_singlepass_notsf.sh --num-jobs 10 --cmd "$train_cmd" \
     $numleaves 15000 $bnroot/train_si84 data/train_si84/ data/lang exp/tri2b-${numleaves}_ali_si84 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_notsf || exit 1;
  # Decode
  scripts/mkgraph.sh data/lang_test_tgpr exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_notsf exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_notsf/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats_notsf.sh exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_notsf/graph_tgpr $bnroot/test_eval92 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_notsf/decode_tgpr_eval92 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats_notsf.sh exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_notsf/graph_tgpr $bnroot/test_dev93 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_notsf/decode_tgpr_dev93 || exit 1;
done


#remove CMN
numleavesL=(2500)
for numleaves in ${numleavesL[@]}; do
  # Train
  steps/train_lda_mllt_bnfeats_singlepass_nocmn.sh --num-jobs 10 --cmd "$train_cmd" \
     $numleaves 15000 $bnroot/train_si84 data/train_si84/ data/lang exp/tri2b-${numleaves}_ali_si84 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_nocmn || exit 1;
  # Decode
  scripts/mkgraph.sh data/lang_test_tgpr exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_nocmn exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_nocmn/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats_nocmn.sh exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_nocmn/graph_tgpr $bnroot/test_eval92 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_nocmn/decode_tgpr_eval92 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats_nocmn.sh exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_nocmn/graph_tgpr $bnroot/test_dev93 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_nocmn/decode_tgpr_dev93 || exit 1;
done


#estimate LDA on monophones
numleavesL=(2500)
for numleaves in ${numleavesL[@]}; do
  # Train
  steps/train_lda_mllt_bnfeats_singlepass_ldamono.sh --num-jobs 10 --cmd "$train_cmd" \
     $numleaves 15000 $bnroot/train_si84 data/train_si84/ data/lang exp/tri2b-${numleaves}_ali_si84 exp/mono1a_ali_si84 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_ldamono || exit 1;
  # Decode
  scripts/mkgraph.sh data/lang_test_tgpr exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_ldamono exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_ldamono/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_ldamono/graph_tgpr $bnroot/test_eval92 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_ldamono/decode_tgpr_eval92 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_ldamono/graph_tgpr $bnroot/test_dev93 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_ldamono/decode_tgpr_dev93 || exit 1;
done


#add deltas
numleavesL=(2500)
for numleaves in ${numleavesL[@]}; do
  # Train
  steps/train_lda_mllt_bnfeats_singlepass_delta.sh --num-jobs 10 --cmd "$train_cmd" \
     $numleaves 15000 $bnroot/train_si84 data/train_si84/ data/lang exp/tri2b-${numleaves}_ali_si84 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_delta || exit 1;
  # Decode
  scripts/mkgraph.sh data/lang_test_tgpr exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_delta exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_delta/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats_delta.sh exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_delta/graph_tgpr $bnroot/test_eval92 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_delta/decode_tgpr_eval92 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats_delta.sh exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_delta/graph_tgpr $bnroot/test_dev93 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_delta/decode_tgpr_dev93 || exit 1;
done


#20iter realign on each, disable MLLT
numleavesL=(2500)
for numleaves in ${numleavesL[@]}; do
  # Train
  steps/train_lda_mllt_bnfeats_singlepass_align.sh --num-jobs 10 --cmd "$train_cmd" \
     $numleaves 15000 $bnroot/train_si84 data/train_si84/ data/lang exp/tri2b-${numleaves}_ali_si84 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_align || exit 1;
  # Decode
  scripts/mkgraph.sh data/lang_test_tgpr exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_align exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_align/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_align/graph_tgpr $bnroot/test_eval92 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_align/decode_tgpr_eval92 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_align/graph_tgpr $bnroot/test_dev93 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_align/decode_tgpr_dev93 || exit 1;
done


#do single pass using tri2a baseline
numleavesL=(2500)
for numleaves in ${numleavesL[@]}; do
  # Train
  steps/train_lda_mllt_bnfeats_singlepass_tri2a.sh --num-jobs 10 --cmd "$train_cmd" \
     $numleaves 15000 $bnroot/train_si84 data/train_si84/ data/lang exp/tri2a-${numleaves}_ali_si84 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a || exit 1;
  # Decode
  scripts/mkgraph.sh data/lang_test_tgpr exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a/graph_tgpr $bnroot/test_eval92 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a/decode_tgpr_eval92 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a/graph_tgpr $bnroot/test_dev93 exp/mono1a_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a/decode_tgpr_dev93 || exit 1;
done




#TANDEM SYSTEM "B"
#trained on linear BN-features, on monophone targets, use dev93 for NN training stopping
#
# Align mono1a with dev93 data
steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
   data/test_dev93 data/lang exp/mono1a exp/mono1a_ali_dev93
# Merge alignments to single archive
gunzip -c exp/mono1a_ali_dev93/*.ali.gz | gzip -c > exp/mono1a_ali_dev93/ali.gz
# Train the nnet
$cuda_cmd exp/mono1a_dev93_nnet-linBN-5L/_train_nnet.log \
  steps/train_nnet_dev_MLP5-linBN.sh --lrate 0.00025 data/train_si84 data/test_dev93 data/lang exp/mono1a_ali_si84 exp/mono1a_ali_dev93 exp/mono1a_dev93_nnet-linBN-5L || exit 1;
#Dump the BN-features
nndir=exp/mono1a_dev93_nnet-linBN-5L
bnroot=$PWD/exp/make_bnfeats_$(basename $nndir)
for x in test_eval92 test_eval93 test_dev93 train_si84; do
  steps/make_bnfeats.sh --bn-dim 30 data/$x $nndir $bnroot/$x $featdir/bnfeats_$(basename $nndir) 4
done
#do single pass using tri2a baseline
numleavesL=(2500)
for numleaves in ${numleavesL[@]}; do
  # Train
  steps/train_lda_mllt_bnfeats_singlepass_tri2a.sh --num-jobs 10 --cmd "$train_cmd" \
     $numleaves 15000 $bnroot/train_si84 data/train_si84/ data/lang exp/tri2a-${numleaves}_ali_si84 exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a || exit 1;
  # Decode
  scripts/mkgraph.sh data/lang_test_tgpr exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a/graph_tgpr $bnroot/test_eval92 exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a/decode_tgpr_eval92 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a/graph_tgpr $bnroot/test_dev93 exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a/decode_tgpr_dev93 || exit 1;
done


#try tuning the alignment scales
transL=(0.5 1.0 2.0);
acwtL=(0.05 0.1 0.2);
loopL=(0.05 0.1 0.2);
for trans in ${transL[@]}; do
  for acwt in ${acwtL[@]}; do
    for loop in ${loopL[@]}; do
      #Re-train the GMMs to new feature space (single pass retraining)
      #( 
      numleaves=2500
      # Train
      steps/train_lda_mllt_bnfeats_singlepass_tri2a.sh --num-jobs 10 --cmd "$train_cmd" \
        --scale-opts "--transition-scale=$trans --acoustic-scale=$acwt --self-loop-scale=$loop" \
        $numleaves 15000 $bnroot/train_si84 data/train_si84/ data/lang exp/tri2a-${numleaves}_ali_si84 exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tr${trans}_aw${acwt}_sl${loop} || exit 1;
      # Decode
      scripts/mkgraph.sh data/lang_test_tgpr exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tr${trans}_aw${acwt}_sl${loop} exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tr${trans}_aw${acwt}_sl${loop}/graph_tgpr || exit 1;
      scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tr${trans}_aw${acwt}_sl${loop}/graph_tgpr $bnroot/test_eval92 exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tr${trans}_aw${acwt}_sl${loop}/decode_tgpr_eval92 || exit 1;
      scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tr${trans}_aw${acwt}_sl${loop}/graph_tgpr $bnroot/test_dev93 exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tr${trans}_aw${acwt}_sl${loop}/decode_tgpr_dev93 || exit 1;
      #) &
    done
    #wait
  done
done


#the best option seems to be: _tr0.5_aw0.05_sl0.1
#train one more system:
numleaves=2500
# Train
steps/train_lda_mllt_bnfeats_singlepass_tri2a.sh --num-jobs 10 --cmd "$train_cmd" \
  --scale-opts "--transition-scale=0.5 --acoustic-scale=0.05 --self-loop-scale=0.1" \
  $numleaves 15000 $bnroot/train_si84 data/train_si84/ data/lang exp/tri2a-${numleaves}_ali_si84 exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali || exit 1;
# Decode
scripts/mkgraph.sh data/lang_test_tgpr exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali/graph_tgpr || exit 1;
scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali/graph_tgpr $bnroot/test_eval92 exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali/decode_tgpr_eval92 || exit 1;
scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali/graph_tgpr $bnroot/test_eval93 exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali/decode_tgpr_eval93 || exit 1;
scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali/graph_tgpr $bnroot/test_dev93 exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali/decode_tgpr_dev93 || exit 1;
#evaluate this system on eval93+eval92: 
#dev93: %WER 17.21 [ 1417 / 8234, 237 ins, 174 del, 1006 sub ]
#eval92: %WER 11.38 [ 642 / 5643, 108 ins, 60 del, 474 sub ]
#eval93: %WER 15.92 [ 549 / 3448, 52 ins, 99 del, 398 sub ]
#eval92+93[is2012]: %WER 13.10 16kHz (compare with is2012: 17.8-PLP-MLE, 15.8-BN, 8kHz)



#TANDEM SYSTEM "C"
#trained on linear BN-features, on monophone targets, use dev93 for NN training stopping
#3million parameters
#
# Train the nnet
$cuda_cmd exp/mono1a_dev93_nnet-linBN-5L-3M/_train_nnet.log \
  steps/train_nnet_dev_MLP5-linBN.sh --model-size 3000000 --lrate 0.000125  data/train_si84 data/test_dev93 data/lang exp/mono1a_ali_si84 exp/mono1a_ali_dev93 exp/mono1a_dev93_nnet-linBN-5L-3M || exit 1;
#Dump the BN-features
nndir=exp/mono1a_dev93_nnet-linBN-5L-3M
bnroot=$PWD/exp/make_bnfeats_$(basename $nndir)
for x in test_eval92 test_eval93 test_dev93 train_si84; do
  steps/make_bnfeats.sh --bn-dim 30 data/$x $nndir $bnroot/$x $featdir/bnfeats_$(basename $nndir) 4
done
#do single pass using tri2a baseline
numleavesL=(2500)
for numleaves in ${numleavesL[@]}; do
  # Train
  steps/train_lda_mllt_bnfeats_singlepass_tri2a.sh --num-jobs 10 --cmd "$train_cmd" \
    --scale-opts "--transition-scale=0.5 --acoustic-scale=0.05 --self-loop-scale=0.1" \
    $numleaves 15000 $bnroot/train_si84 data/train_si84/ data/lang exp/tri2a-${numleaves}_ali_si84 exp/mono1a_dev93_nnet-linBN-5L-3M.gmm-${numleaves}_singlepass_tri2a || exit 1;
  # Decode
  scripts/mkgraph.sh data/lang_test_tgpr exp/mono1a_dev93_nnet-linBN-5L-3M.gmm-${numleaves}_singlepass_tri2a exp/mono1a_dev93_nnet-linBN-5L-3M.gmm-${numleaves}_singlepass_tri2a/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_dev93_nnet-linBN-5L-3M.gmm-${numleaves}_singlepass_tri2a/graph_tgpr $bnroot/test_eval92 exp/mono1a_dev93_nnet-linBN-5L-3M.gmm-${numleaves}_singlepass_tri2a/decode_tgpr_eval92 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_dev93_nnet-linBN-5L-3M.gmm-${numleaves}_singlepass_tri2a/graph_tgpr $bnroot/test_eval93 exp/mono1a_dev93_nnet-linBN-5L-3M.gmm-${numleaves}_singlepass_tri2a/decode_tgpr_eval93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_dev93_nnet-linBN-5L-3M.gmm-${numleaves}_singlepass_tri2a/graph_tgpr $bnroot/test_dev93 exp/mono1a_dev93_nnet-linBN-5L-3M.gmm-${numleaves}_singlepass_tri2a/decode_tgpr_dev93 || exit 1;
done


#TANDEM SYSTEM "D"
#si284 training data
# Train the nnet
$cuda_cmd exp/mono1a_dev93_nnet-linBN-5L-3M_si284/_train_nnet.log \
  steps/train_nnet_dev_MLP5-linBN.sh --model-size 3000000 --lrate 0.0000625  data/train_si284 data/test_dev93 data/lang exp/mono1a_ali_si284 exp/mono1a_ali_dev93 exp/mono1a_dev93_nnet-linBN-5L-3M_si284 || exit 1;
#Dump the BN-features
nndir=exp/mono1a_dev93_nnet-linBN-5L-3M_si284
bnroot=$PWD/exp/make_bnfeats_$(basename $nndir)
for x in test_eval92 test_eval93 test_dev93 train_si84 train_si284; do
  steps/make_bnfeats.sh --bn-dim 30 data/$x $nndir $bnroot/$x $featdir/bnfeats_$(basename $nndir) 4
done
#do single pass using tri2a baseline
numleavesL=(2500)
for numleaves in ${numleavesL[@]}; do
  # Train
  steps/train_lda_mllt_bnfeats_singlepass_tri2a.sh --num-jobs 10 --cmd "$train_cmd" \
    --scale-opts "--transition-scale=0.5 --acoustic-scale=0.05 --self-loop-scale=0.1" \
    $numleaves 15000 $bnroot/train_si84 data/train_si84/ data/lang exp/tri2a-${numleaves}_ali_si84 exp/mono1a_dev93_nnet-linBN-5L-3M_si284.gmm-${numleaves}_singlepass_tri2a || exit 1;
  # Decode
  scripts/mkgraph.sh data/lang_test_tgpr exp/mono1a_dev93_nnet-linBN-5L-3M_si284.gmm-${numleaves}_singlepass_tri2a exp/mono1a_dev93_nnet-linBN-5L-3M_si284.gmm-${numleaves}_singlepass_tri2a/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_dev93_nnet-linBN-5L-3M_si284.gmm-${numleaves}_singlepass_tri2a/graph_tgpr $bnroot/test_eval92 exp/mono1a_dev93_nnet-linBN-5L-3M_si284.gmm-${numleaves}_singlepass_tri2a/decode_tgpr_eval92 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_dev93_nnet-linBN-5L-3M_si284.gmm-${numleaves}_singlepass_tri2a/graph_tgpr $bnroot/test_eval93 exp/mono1a_dev93_nnet-linBN-5L-3M_si284.gmm-${numleaves}_singlepass_tri2a/decode_tgpr_eval93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_dev93_nnet-linBN-5L-3M_si284.gmm-${numleaves}_singlepass_tri2a/graph_tgpr $bnroot/test_dev93 exp/mono1a_dev93_nnet-linBN-5L-3M_si284.gmm-${numleaves}_singlepass_tri2a/decode_tgpr_dev93 || exit 1;
done



#TANDEM SYSTEM "E"
#universal context network, as in ASRU2011...\
#1st) stage MLP
$cuda_cmd exp/mono1a_dev93_nnet-linBN-UC7L-3M_si284_I/_train_nnet.log \
  steps/train_nnet_dev_MLP5-linBN.sh --model-size 1480000 --lrate 0.0000625 --bn-size 80 --splice-lr 5 --dct-basis 6  data/train_si284 data/test_dev93 data/lang exp/mono1a_ali_si284 exp/mono1a_ali_dev93 exp/mono1a_dev93_nnet-linBN-UC7L-3M_si284_I || exit 1;
#cut, normalize, dump feats


#2nd) stage MLP

#TODO









