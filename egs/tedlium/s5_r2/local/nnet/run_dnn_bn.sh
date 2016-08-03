#!/bin/bash
#
# Based mostly on the Switchboard recipe. The training database is TED-LIUM,
# it consists of TED talks with cleaned automatic transcripts:
#
# http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus
# http://www.openslr.org/resources (Mirror).
#
# The data is distributed under 'Creative Commons BY-NC-ND 3.0' license,
# which allow free non-commercial use, while only a citation is required.
#
# Copyright  2014 Nickolay V. Shmyrev 
#            2015 Brno University of Technology (Author: Karel Vesely)
#            2015 Alex Glubshev
# Apache 2.0
#

. cmd.sh
. path.sh

nj=32
njdec=11
njfea=10

# label,
exp=BN

# source data,
ali_src=exp/tri3_ali
graph_src=exp/tri3/graph

# fbank features
test=data-fbank/test
train=data-fbank/train

test_original=data/test
train_original=data/train

# bn features,
test_bn=data-fbank-${exp}-bn/test
train_bn=data-fbank-${exp}-bn/train

# fmllr features,
test_bn_fmllr=data-fbank-${exp}-bn-fmllr/test
train_bn_fmllr=data-fbank-${exp}-bn-fmllr/train

stage=0
. utils/parse_options.sh # accept options

# Make the kaldi FBANK+PITCH features,
[ ! -e $test ] && if [ $stage -le 0 ]; then
  # Test set
  utils/copy_data_dir.sh $test_original $test || exit 1; rm $test/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj $njfea --cmd "$train_cmd" \
    $test $test/log $test/data || exit 1;
  steps/compute_cmvn_stats.sh $test $test/log $test/data || exit 1;  
  
  # Train set
  utils/copy_data_dir.sh $train_original $train || exit 1; rm $train/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj $njfea --cmd "$train_cmd" \
     $train $train/log $train/data || exit 1;
    steps/compute_cmvn_stats.sh $train $train/log $train/data || exit 1;
  
  # Split to training 90%, cv 10%
  utils/subset_data_dir_tr_cv.sh $train ${train}_tr90 ${train}_cv10 || exit 1;
fi

# Train the bottleneck network,
lang=data/lang_test
if [ $stage -le 1 ]; then
  dir=exp/dnn8a_${exp}_bn-feat
  ali=$ali_src
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --hid-layers 2 --hid-dim 1500 --bn-dim 40 \
    --cmvn-opts "--norm-means=true --norm-vars=false" --feat-type traps \
    --splice 5 --traps-dct-basis 6 --learn-rate 0.008 \
    ${train}_tr90 ${train}_cv10 $lang $ali $ali $dir || exit 1
  
  # Decode test,
  steps/nnet/decode.sh --nj $njdec --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.10 \
    $graph_src $test $dir/decode_test || exit 1
fi

# Store the bottleneck features,
if [ $stage -le 2 ]; then
  dir=exp/dnn8a_${exp}_bn-feat
  # dev
  steps/nnet/make_bn_feats.sh --nj $njfea --cmd "$train_cmd" $test_bn $test $dir $test_bn/log $test_bn/data || exit 1 
  steps/compute_cmvn_stats.sh $test_bn $test_bn/log $test_bn/data || exit 1;
  # train
  steps/nnet/make_bn_feats.sh --nj $njfea --cmd "$train_cmd" $train_bn $train $dir $train_bn/log $train_bn/data || exit 1
  steps/compute_cmvn_stats.sh $train_bn $train_bn/log $train_bn/data || exit 1;
fi

# Train GMM on bottleneck features,
lang_test=data/lang_test
if [ $stage -le 3 ]; then
  dir=exp/dnn8b_${exp}_bn-gmm
  # Train,
  # gmm on bn features, no cmvn, no lda-mllt,
  steps/train_deltas.sh --power 0.5 --boost-silence 1.5 --cmd "$train_cmd" \
    --delta-opts "--delta-order=0" \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --beam 20 --retry-beam 80 \
    5000 80000 $train_bn $lang $ali_src $dir || exit 1
  # Decode,
  utils/mkgraph.sh $lang_test $dir $dir/graph || exit 1
  steps/decode.sh --nj $njdec --cmd "$decode_cmd" \
    --acwt 0.05 --beam 15.0 --lattice-beam 8.0 \
    $dir/graph $test_bn $dir/decode_$(basename $test_bn) || exit 1
  # Align,
  steps/align_fmllr.sh --boost-silence 1.5 --nj $nj --cmd "$train_cmd" \
    --beam 20 --retry-beam 80 \
    $train_bn $lang $dir ${dir}_ali || exit 1;
fi

# Train SAT-adapted GMM on bottleneck features,
if [ $stage -le 4 ]; then
  dir=exp/dnn8c_${exp}_fmllr-gmm
  ali=exp/dnn8b_${exp}_bn-gmm_ali
  # Train,
  # fmllr-gmm system on bottleneck features, 
  # - no cmvn, put fmllr to the features directly (no lda),
  # - note1 : we don't need cmvn, similar effect has diagonal of fmllr transform,
  # - note2 : lda+mllt was causing a small hit <0.5%,
  steps/train_sat.sh --power 0.5 --boost-silence 1.5 --cmd "$train_cmd" \
    --beam 20 --retry-beam 80 \
    5000 80000 $train_bn $lang $ali $dir || exit 1
  # Decode,
  utils/mkgraph.sh $lang_test $dir $dir/graph || exit 1;
  steps/decode_fmllr.sh --nj $njdec --cmd "$decode_cmd" \
    --acwt 0.05 --beam 15.0 --lattice-beam 8.0 \
    $dir/graph $test_bn $dir/decode_$(basename $test_bn) || exit 1
  # Align,
  steps/align_fmllr.sh --boost-silence 1.5 --nj $nj --cmd "$train_cmd" \
    --beam 20 --retry-beam 80 \
    $train_bn $lang $dir ${dir}_ali || exit 1;
fi

# Store the bottleneck-FMLLR features,
gmm=exp/dnn8c_${exp}_fmllr-gmm # fmllr-feats, dnn-targets,
graph=$gmm/graph
if [ $stage -le 5 ]; then
  # Dev_set
  steps/nnet/make_fmllr_feats.sh --nj $njfea --cmd "$train_cmd" \
     --transform-dir $gmm/decode_$(basename $test_bn) \
     $test_bn_fmllr $test_bn $gmm $test_bn_fmllr/log $test_bn_fmllr/data || exit 1;
  # Training set
  steps/nnet/make_fmllr_feats.sh --nj $njfea --cmd "$train_cmd --max-jobs-run 10" \
     --transform-dir ${gmm}_ali \
     $train_bn_fmllr $train_bn $gmm $train_bn_fmllr/log $train_bn_fmllr/data || exit 1;
  # Split the training set
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train_bn_fmllr ${train_bn_fmllr}_tr90 ${train_bn_fmllr}_cv10
fi

#------------------------------------------------------------------------------------
# Pre-train stack of RBMs (6 layers, 2048 units),
if [ $stage -le 6 ]; then
  dir=exp/dnn8d_${exp}_pretrain-dbn; mkdir -p $dir
  # Create input transform, splice 13 frames [ -10 -5..+5 +10 ],
  echo "<Splice> <InputDim> 40 <OutputDim> 520 <BuildVector> -10 -5:1:5 10 </BuildVector>" >$dir/proto.main
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --feature-transform-proto $dir/proto.main \
    $train_bn_fmllr $dir || exit 1
fi

#------------------------------------------------------------------------------------
# Train the DNN optimizing cross-entropy,
if [ $stage -le 7 ]; then
  dir=exp/dnn8e_${exp}_pretrain-dbn_dnn
  ali=${gmm}_ali
  feature_transform=exp/dnn8d_${exp}_pretrain-dbn/final.feature_transform # re-use
  dbn=exp/dnn8d_${exp}_pretrain-dbn/6.dbn # re-use
  # Train  
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    ${train_bn_fmllr}_tr90 ${train_bn_fmllr}_cv10 $lang $ali $ali $dir || exit 1;
  # Decode test
  steps/nnet/decode.sh --nj $njdec --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.10 \
    $graph $test_bn_fmllr $dir/decode_$(basename $test_bn_fmllr) || exit 1
fi

#------------------------------------------------------------------------------------
# Finally we optimize sMBR criterion, we do Stochastic-GD with per-utterance updates, 
dir=exp/dnn8f_${exp}_pretrain-dbn_dnn_smbr
srcdir=exp/dnn8e_${exp}_pretrain-dbn_dnn
acwt=0.1
#
if [ $stage -le 8 ]; then
  # Generate lattices and alignments
  steps/nnet/align.sh --nj $nj --cmd "$train_cmd" \
    $train_bn_fmllr $lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj $nj --cmd "$decode_cmd" --acwt $acwt \
    $train_bn_fmllr $lang $srcdir ${srcdir}_denlats  || exit 1;
fi
if [ $stage -le 9 ]; then
  # Do 4 epochs of sMBR (leaving out all silence frames and compensating insertions), 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt \
    --do-smbr true --exclude-silphones true --one-silence-class true \
    $train_bn_fmllr $lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode test,
  for ITER in 1 2 3 4; do
    steps/nnet/decode.sh --nj $njdec --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
      --nnet $dir/${ITER}.nnet \
      $graph $test_bn_fmllr $dir/decode_$(basename $test_bn_fmllr)_it${ITER} || exit 1
  done
fi 

echo $0 successs.
exit 0
