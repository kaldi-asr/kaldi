#!/usr/bin/env bash

. ./cmd.sh

. ./path.sh


###
### First we need to prepare the FBANK features
### that will be on the input to the BN network.
###
#false && \
{
  #eval2000
  dir=data-fbank/eval2000
  srcdir=data/eval2000
  {
    mkdir -p $dir; cp $srcdir/* $dir; rm $dir/{feats.scp,cmvn.scp};
    steps/make_fbank.sh --cmd "$train_cmd" --nj 20 \
      $dir $dir/_log $dir/_data || exit 1;
    steps/compute_cmvn_stats.sh $dir $dir/_log $dir/_data || exit 1;
  }
  #training set
  dir=data-fbank/train
  srcdir=data/train
  {
    mkdir -p $dir; cp $srcdir/* $dir; rm $dir/{feats.scp,cmvn.scp};
    steps/make_fbank.sh --cmd "$train_cmd" --nj 20 \
      $dir $dir/_log $dir/_data || exit 1;
    steps/compute_cmvn_stats.sh $dir $dir/_log $dir/_data || exit 1;
  }
}

# Now we prepare the subsets as in case of the default MFCC features
# (Some are be unused by this recipe, but can be handy in future...)
data=data-fbank
#false && \
{
  # Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
  utils/subset_data_dir.sh --first $data/train 4000 $data/train_dev # 5hr 6min
  n=$[`cat data/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last $data/train $n $data/train_nodev

  # Prepare data for training mono
  utils/subset_data_dir.sh --shortest $data/train_nodev 100000 $data/train_100kshort
  utils/subset_data_dir.sh  $data/train_100kshort 10000 $data/train_10k
  utils/data/remove_dup_utts.sh 100 $data/train_10k $data/train_10k_nodup

  # Take the first 30k utterances (about 1/8th of the data)
  utils/subset_data_dir.sh --first $data/train_nodev 30000 $data/train_30k
  utils/data/remove_dup_utts.sh 200 $data/train_30k $data/train_30k_nodup

  utils/data/remove_dup_utts.sh 300 $data/train_nodev $data/train_nodup

  # Take the first 100k utterances (just under half the data); we'll use
  # this for later stages of training.
  utils/subset_data_dir.sh --first $data/train_nodev 100000 $data/train_100k
  utils/data/remove_dup_utts.sh 200 $data/train_100k $data/train_100k_nodup
}


### Prepare the alignments (in case these were not yet prepared by the run_hybrid.sh example)
#false && \
{
  #train_dev
  if [ ! -d exp/tri5a_ali_dev ]; then
  steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
    data/train_dev data/lang exp/tri5a exp/tri5a_ali_dev || exit 1
  fi
  #train_100k_nodup
  if [ ! -d exp/tri5a_ali_100k_nodup ]; then
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
    data/train_100k_nodup data/lang exp/tri5a exp/tri5a_ali_100k_nodup || exit 1
  fi
}



###
### Now we can build the universal-context bottleneck network
### - Universal context MLP is a hierarchy of two bottleneck neural networks
### - The first network can see a limited range of frames (11 frames)
### - The second network sees concatenation of bottlneck outputs of the first 
###   network, with temporal shifts -10 -5 0 5 10, (in total a range of 31 frames 
###   in the original feature space)
### - This structure has been reported to produce superior performance
###   compared to a network with single bottleneck
### 


#false && \
{
  # Let's train the first network:
  # - the topology will be 90_1200_1200_80_1200_NSTATES, the bottleneck is linear
  { 
  dir=exp/tri5a_uc-mlp-part1
  ali=exp/tri5a_ali
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet.sh --hid-layers 2 --hid-dim 1200 --bn-dim 80 --apply-cmvn true --feat-type traps --splice 5 --traps-dct-basis 6  --learn-rate 0.008 \
    data-fbank/train_100k_nodup data-fbank/train_dev data/lang ${ali}_100k_nodup ${ali}_dev $dir || exit 1;
  } 

  # Compose feature_transform for the next stage 
  # - remaining part of the first network is fixed
  dir=exp/tri5a_uc-mlp-part1
  feature_transform=$dir/final.feature_transform.part1
  {
    nnet-concat $dir/final.feature_transform \
      "nnet-copy --remove-last-layers=4 --binary=false $dir/final.nnet - |" \
      "utils/nnet/gen_splice.py --fea-dim=80 --splice=2 --splice-step=5 |" \
      $feature_transform
  }

  # Let's train the second network:
  # - the topology will be 400_1200_1200_30_1200_NSTATES, again, the bottleneck is linear
  { # Train the MLP
  dir=exp/tri5a_uc-mlp-part2
  ali=exp/tri5a_ali
  $cuda_cmd $dir/_train_nnet.log \
    steps/train_nnet.sh --hid-layers 2 --hid-dim 1200 --bn-dim 30 --apply-cmvn true --feature-transform $feature_transform --learn-rate 0.008 \
    data-fbank/train_100k_nodup data-fbank/train_dev data/lang ${ali}_100k_nodup ${ali}_dev $dir || exit 1;
  }
}



###
### We have the MLP, so lets do store the BN features
###
#false && \
{ 
  # Prepare the BN-features
  data=data-bn/tri5a_uc-mlp-part2 
  srcdata=data-fbank/
  nnet=exp/tri5a_uc-mlp-part2
  {
    steps/make_bn_feats.sh --cmd "$train_cmd" --nj 20 $data/eval2000 $srcdata/eval2000 $nnet $data/eval2000/_log $data/eval2000/_data || exit 1
    # we will need all the subsets :  even for mono, so we will prepare the train_nodev
    steps/make_bn_feats.sh --cmd "$train_cmd" --nj 40 $data/train $srcdata/train $nnet $data/train/_log $data/train/_data || exit 1
  }
}

#false && \
{
  #compute CMVN of the BN-features
  dir=data-bn/tri5a_uc-mlp-part2/train
  steps/compute_cmvn_stats.sh $dir $dir/_log $dir/_data || exit 1;
  dir=data-bn/tri5a_uc-mlp-part2/eval2000
  steps/compute_cmvn_stats.sh $dir $dir/_log $dir/_data || exit 1;
}



###
### Prepare the BN-feature subsets same as with MFCCs in run.sh 
###
data=data-bn/tri5a_uc-mlp-part2/
#false && \
{
  # Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
  utils/subset_data_dir.sh --first $data/train 4000 $data/train_dev # 5hr 6min
  n=$[`cat data/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last $data/train $n $data/train_nodev

  # Prepare data for training mono
  utils/subset_data_dir.sh --shortest $data/train_nodev 100000 $data/train_100kshort
  utils/subset_data_dir.sh  $data/train_100kshort 10000 $data/train_10k
  utils/data/remove_dup_utts.sh 100 $data/train_10k $data/train_10k_nodup

  # Take the first 30k utterances (about 1/8th of the data)
  utils/subset_data_dir.sh --first $data/train_nodev 30000 $data/train_30k
  utils/data/remove_dup_utts.sh 200 $data/train_30k $data/train_30k_nodup

  utils/data/remove_dup_utts.sh 300 $data/train_nodev $data/train_nodup

  # Take the first 100k utterances (just under half the data); we'll use
  # this for later stages of training.
  utils/subset_data_dir.sh --first $data/train_nodev 100000 $data/train_100k
  utils/data/remove_dup_utts.sh 200 $data/train_100k $data/train_100k_nodup
}



###
### Now start building the tandem GMM system
### - train from mono to tri5a, run also the mmi training
###
bndata=data-bn/tri5a_uc-mlp-part2/

steps/tandem/train_mono.sh --nj 10 --cmd "$train_cmd" \
  data/train_10k_nodup $bndata/train_10k_nodup data/lang exp/tandem2uc-mono0a || exit 1;

steps/tandem/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train_30k_nodup $bndata/train_30k_nodup data/lang exp/tandem2uc-mono0a exp/tandem2uc-mono0a_ali || exit 1;

steps/tandem/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 data/train_30k_nodup $bndata/train_30k_nodup data/lang exp/tandem2uc-mono0a_ali exp/tandem2uc-tri1 || exit 1;
 
utils/mkgraph.sh data/lang_test exp/tandem2uc-tri1 exp/tandem2uc-tri1/graph

steps/tandem/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode_tandem.config \
  exp/tandem2uc-tri1/graph data/eval2000 $bndata/eval2000 exp/tandem2uc-tri1/decode_eval2000 &


steps/tandem/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train_30k_nodup $bndata/train_30k_nodup data/lang exp/tandem2uc-tri1 exp/tandem2uc-tri1_ali || exit 1;

steps/tandem/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 data/train_30k_nodup $bndata/train_30k_nodup data/lang exp/tandem2uc-tri1_ali exp/tandem2uc-tri2 || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tandem2uc-tri2 exp/tandem2uc-tri2/graph || exit 1;
  steps/tandem/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode_tandem.config \
   exp/tandem2uc-tri2/graph data/eval2000 $bndata/eval2000 exp/tandem2uc-tri2/decode_eval2000 || exit 1;
)&



steps/tandem/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_30k_nodup $bndata/train_30k_nodup data/lang exp/tandem2uc-tri2 exp/tandem2uc-tri2_ali || exit 1;

# Train tri3a, which is LDA+MLLT, on 30k_nodup data.
steps/tandem/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   2500 20000 data/train_30k_nodup $bndata/train_30k_nodup data/lang exp/tandem2uc-tri2_ali exp/tandem2uc-tri3a || exit 1;
(
  utils/mkgraph.sh data/lang_test exp/tandem2uc-tri3a exp/tandem2uc-tri3a/graph || exit 1;
  steps/tandem/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode_tandem.config \
   exp/tandem2uc-tri3a/graph data/eval2000 $bndata/eval2000 exp/tandem2uc-tri3a/decode_eval2000 || exit 1;
)&




# From now, we start building a more serious system (with SAT), and we'll
# do the alignment with fMLLR.

steps/tandem/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k_nodup $bndata/train_100k_nodup data/lang exp/tandem2uc-tri3a exp/tandem2uc-tri3a_ali_100k_nodup || exit 1;


steps/tandem/train_sat.sh  --cmd "$train_cmd" \
  2500 20000 data/train_100k_nodup $bndata/train_100k_nodup data/lang exp/tandem2uc-tri3a_ali_100k_nodup exp/tandem2uc-tri4a || exit 1;
(
  utils/mkgraph.sh data/lang_test exp/tandem2uc-tri4a exp/tandem2uc-tri4a/graph
  steps/tandem/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode_tandem.config \
   exp/tandem2uc-tri4a/graph data/eval2000 $bndata/eval2000 exp/tandem2uc-tri4a/decode_eval2000
  steps/tandem/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode_tandem.config \
   exp/tandem2uc-tri4a/graph data/train_dev $bndata/train_dev exp/tandem2uc-tri4a/decode_train_dev
)&


steps/tandem/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k_nodup $bndata/train_100k_nodup data/lang exp/tandem2uc-tri4a exp/tandem2uc-tri4a_ali_100k_nodup


# Building a larger SAT system.
steps/tandem/train_sat.sh --cmd "$train_cmd" \
  3500 100000 data/train_100k_nodup $bndata/train_100k_nodup data/lang exp/tandem2uc-tri4a_ali_100k_nodup exp/tandem2uc-tri5a || exit 1;
(
  utils/mkgraph.sh data/lang_test exp/tandem2uc-tri5a exp/tandem2uc-tri5a/graph || exit 1;
  steps/tandem/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode_tandem.config \
   --nj 30 exp/tandem2uc-tri5a/graph data/eval2000 $bndata/eval2000 exp/tandem2uc-tri5a/decode_eval2000 || exit 1;
)


# MMI starting from system in tandem2uc-tri5a.  Use the same data (100k_nodup).
# Later we'll use all of it.
steps/tandem/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_100k_nodup $bndata/train_100k_nodup data/lang exp/tandem2uc-tri5a exp/tandem2uc-tri5a_ali_100k_nodup || exit 1;
steps/tandem/make_denlats.sh --nj 40 --cmd "$decode_cmd" --transform-dir exp/tandem2uc-tri5a_ali_100k_nodup \
   --config conf/decode_tandem.config \
   --sub-split 50 data/train_100k_nodup $bndata/train_100k_nodup data/lang exp/tandem2uc-tri5a exp/tandem2uc-tri5a_denlats_100k_nodup  || exit 1;

steps/tandem/train_mmi.sh --cmd "$decode_cmd" --boost 0.1 --acwt 0.039 \
  data/train_100k_nodup $bndata/train_100k_nodup data/lang exp/tandem2uc-tri5a_{ali,denlats}_100k_nodup exp/tandem2uc-tri5a_mmi_b0.1 || exit 1;

steps/tandem/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode_tandem.config \
  --transform-dir exp/tandem2uc-tri5a/decode_eval2000 \
  exp/tandem2uc-tri5a/graph data/eval2000 $bndata/eval2000 exp/tandem2uc-tri5a_mmi_b0.1/decode_eval2000 || exit 1;

# The fmmi_mmi training is not in the TANDEM scritps
#
#steps/tandem/train_diag_ubm.sh --silence-weight 0.5 --nj 40 --cmd "$train_cmd" \
#  700 data/train_100k_nodup data/lang exp/tandem2uc-tri5a_ali_100k_nodup exp/tandem2uc-tri5a_dubm

#steps/tandem/train_mmi_fmmi.sh --learning-rate 0.005 \
#  --boost 0.1 --cmd "$train_cmd" \
# data/train_100k_nodup data/lang exp/tandem2uc-tri5a_ali_100k_nodup exp/tandem2uc-tri5a_dubm exp/tandem2uc-tri5a_denlats_100k_nodup \
#   exp/tandem2uc-tri5a_fmmi_b0.1 || exit 1;
# for iter in 4 5 6 7 8; do
#  steps/tandem/decode_fmmi.sh --nj 30 --cmd "$decode_cmd" --iter $iter \
#     --config conf/decode.config --transform-dir exp/tandem2uc-tri5a/decode_eval2000 \
#     exp/tandem2uc-tri5a/graph data/eval2000 exp/tandem2uc-tri5a_fmmi_b0.1/decode_eval2000_it$iter &
# done


