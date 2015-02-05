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
#            2014 Brno University of Technology (Author: Karel Vesely)
#            2014 Alex Glubshev
# Apache 2.0
#

# TODO : use pruned trigram?

. cmd.sh
. path.sh

nj=32
decode_nj=8

# label,
exp=sstNO

# source data,
ali_src=exp/tri3_ali
graph_src=exp/tri3/graph

#fbank features
train=data/train
dev=data/test

# bn features,
dev_bn=data-fbank-bn-${exp}/conversational.dev.seg1
train_bn=data-fbank-bn-${exp}/training.seg1

# fmllr features,
dev_bn_fmllr=data-fbank-bn-fmllr-${exp}/conversational.dev.seg1
train_bn_fmllr=data-fbank-bn-fmllr-${exp}/training.seg1

stage=0
. utils/parse_options.sh # accept options

# Data preparation
if [ $stage -le 0 ]; then
  #local/download_data.sh || exit 1
  
  local/prepare_data.sh || exit 1

  local/prepare_dict.sh || exit 1

  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang || exit 1

  local/prepare_lm.sh || exit 1
fi

# Make the kaldi FBANK+PITCH features,
if [ $stage -le 1 ]; then
  # Dev set
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" \
    $dev $dev/log $dev/data || exit 1;
  steps/compute_cmvn_stats.sh $dev $dev/log $dev/data || exit 1;  
  
  # Training set
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" \
     $train $train/log $train/data || exit 1;
    steps/compute_cmvn_stats.sh $train $train/log $train/data || exit 1;
  
  # Split to training 90%, cv 10%
  utils/subset_data_dir_tr_cv.sh $train ${train}_tr90 ${train}_cv10 || exit 1;
fi

# Train the bottleneck network,
lang=data/lang_test
njdec=11
if [ $stage -le 2 ]; then
  dir=exp/dnn8a_bn-feat_${exp}
  ali=$ali_src
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --hid-layers 2 --hid-dim 1500 --bn-dim 40 \
    --cmvn-opts "--norm-means=true --norm-vars=false" --feat-type traps \
    --splice 5 --traps-dct-basis 6 --learn-rate 0.008 \
    ${train}_tr90 ${train}_cv10 $lang $ali $ali $dir || exit 1
  # Decode conversational.dev
  steps/nnet/decode.sh --nj $njdec --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.10 \
    --scoring-opts "$scoring" --num-threads 2 --parallel-opts "-pe smp 2" --max-mem 150000000 \
     $graph_src $dev $dir/decode_test || exit 1
fi

# Store the bottleneck features,
if [ $stage -le 3 ]; then
  dir=exp/dnn8a_bn-feat_${exp}
  # dev
  steps/nnet/make_bn_feats.sh --nj 10 --cmd "$train_cmd" $dev_bn $dev $dir $dev_bn/log $dev_bn/data || exit 1 
  steps/compute_cmvn_stats.sh $dev_bn $dev_bn/log $dev_bn/data || exit 1;
  # train
  steps/nnet/make_bn_feats.sh --nj 10 --cmd "$train_cmd" $train_bn $train $dir $train_bn/log $train_bn/data || exit 1
  steps/compute_cmvn_stats.sh $train_bn $train_bn/log $train_bn/data || exit 1;
fi

# Train GMM on bottleneck features,
lang_test=data/lang_test
if [ $stage -le 4 ]; then
  dir=exp/dnn8b_bn-gmm_${exp}
  # Train,
  # gmm on bn features, no cmvn, no lda-mllt,
  if false
  then
  steps/train_deltas.sh --power 0.5 --boost-silence 1.5 --cmd "$train_cmd" \
    --delta-opts "--delta-order=0" \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --beam 20 --retry-beam 80 \
    6000 26000 $train_bn $lang $ali_src $dir || exit 1
  fi
  # Decode,
  utils/mkgraph.sh $lang_test $dir $dir/graph || exit 1
  steps/decode.sh --nj $njdec --scoring-opts "$scoring" --cmd "$decode_cmd" \
    --num-threads 3 --parallel-opts "-pe smp 3" \
    --acwt 0.05 --beam 15.0 --lattice-beam 8.0 \
    $dir/graph $dev_bn $dir/decode_$(basename $dev_bn) || exit 1
  # Align,
  steps/align_fmllr.sh --boost-silence 1.5 --nj 32 --cmd "$train_cmd" \
    --beam 20 --retry-beam 80 \
    $train_bn $lang $dir ${dir}_ali || exit 1;
fi

# Train SAT-adapted GMM on bottleneck features,
if [ $stage -le 5 ]; then
  dir=exp/dnn8c_fmllr-gmm_${exp}
  ali=exp/dnn8b_bn-gmm_${exp}_ali
  # Train,
  # fmllr-gmm system on bottleneck features, 
  # - no cmvn, put fmllr to the features directly (no lda),
  # - note1 : we don't need cmvn, similar effect has diagonal of fmllr transform,
  # - note2 : lda+mllt was causing a small hit <0.5%,
  steps/train_sat.sh --power 0.5 --boost-silence 1.5 --cmd "$train_cmd" \
    --beam 20 --retry-beam 80 \
    6000 26000 $train_bn $lang $ali $dir || exit 1
  # Decode,
  utils/mkgraph.sh $lang_test $dir $dir/graph || exit 1;
  steps/decode_fmllr.sh --nj $njdec --scoring-opts "$scoring" --cmd "$decode_cmd" \
    --num-threads 3 --parallel-opts "-pe smp 3" \
    --acwt 0.05 --beam 15.0 --lattice-beam 8.0 \
    $dir/graph $dev_bn $dir/decode_$(basename $dev_bn) || exit 1
  # Align,
  steps/align_fmllr.sh --boost-silence 1.5 --nj 32 --cmd "$train_cmd" \
    --beam 20 --retry-beam 80 \
    $train_bn $lang $dir ${dir}_ali || exit 1;
fi

# Store the bottleneck-FMLLR features
gmm=exp/dnn8c_fmllr-gmm_${exp} # fmllr-feats, dnn-targets,
graph=$gmm/graph
if [ $stage -le 6 ]; then
  # Dev_set
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir $gmm/decode_$(basename $dev_bn) \
     $dev_bn_fmllr $dev_bn $gmm $dev_bn_fmllr/log $dev_bn_fmllr/data || exit 1;
  # Training set
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd -tc 10" \
     --transform-dir ${gmm}_ali \
     $train_bn_fmllr $train_bn $gmm $train_bn_fmllr/log $train_bn_fmllr/data || exit 1;
  # Split the training set
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train_bn_fmllr ${train_bn_fmllr}_tr90 ${train_bn_fmllr}_cv10
fi

#------------------------------------------------------------------------------------
# Pre-train stack of RBMs (6 layers, 2048 units)
rbm_iter=3
if [ $stage -le 7 ]; then
  dir=exp/dnn8d_pretrain-dbn_${exp}
  # Create input transform, splice 13 frames [ -10 -5..+5 +10 ],
  mkdir -p $dir
  echo "<NnetProto>
        <Splice> <InputDim> 40 <OutputDim> 520 <BuildVector> -10 -5:1:5 10 </BuildVector>
        </NnetProto>" >$dir/proto.main
  # Do CMVN first, then frame pooling:
  nnet-concat "compute-cmvn-stats scp:${train_bn_fmllr}/feats.scp - | cmvn-to-nnet - - |" "nnet-initialize $dir/proto.main - |" $dir/transf.init || exit 1
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --feature-transform $dir/transf.init --rbm-iter $rbm_iter $train_bn_fmllr $dir || exit 1
fi

#------------------------------------------------------------------------------------
# Train the DNN optimizing cross-entropy.
if [ $stage -le 8 ]; then
  dir=exp/dnn8e_pretrain-dbn_dnn_${exp}
  ali=${gmm}_ali
  feature_transform=exp/dnn8d_pretrain-dbn_${exp}/final.feature_transform # re-use
  dbn=exp/dnn8d_pretrain-dbn_${exp}/6.dbn # re-use
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train  
  $cuda_cmd $dir/log/train_nnet.log steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 ${train_bn_fmllr}_tr90 ${train_bn_fmllr}_cv10 $lang $ali $ali $dir || exit 1;
  # Decode conversational.dev
  steps/nnet/decode.sh --nj $njdec --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.10 \
    --scoring-opts "$scoring" --num-threads 2 --parallel-opts "-pe smp 2" --max-mem 150000000 \
    $graph $dev_bn_fmllr $dir/decode_$(basename $dev_bn_fmllr) || exit 1
fi

exit

#------------------------------------------------------------------------------------
# Train the DNN optimizing cross-entropy (2nd time)
# Re-using RBMs, making alignments from 1st DNN.
# [WER lower by 0.2% on sMBR system] TODO IS IT STILL TRUE ????
if [ $stage -le 9 ]; then
  
  # Re-align
  srcdir=exp/dnn8e_pretrain-dbn_dnn_${exp}
  #steps/nnet/align.sh --nj 32 --cmd "$train_cmd" \
  #  $train_bn_fmllr $lang $srcdir ${srcdir}_ali || exit 1;

  dir=exp/dnn8f_pretrain-dbn_dnn_run2_${exp}
  ali=${srcdir}_ali # DNN alignment
  feature_transform=exp/dnn8d_pretrain-dbn_${exp}/final.feature_transform # re-use
  dbn=exp/dnn8d_pretrain-dbn_${exp}/6.dbn # re-use
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train
  $cuda_cmd $dir/log/train_nnet.log steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 ${train_bn_fmllr}_tr90 ${train_bn_fmllr}_cv10 $lang $ali $ali $dir || exit 1;
  # Decode conversational.dev
  steps/nnet/decode.sh --nj $njdec --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.10 \
    --scoring-opts "$scoring" --num-threads 2 --parallel-opts "-pe smp 2" --max-mem 150000000 \
    $graph $dev_bn_fmllr $dir/decode_$(basename $dev_bn_fmllr) || exit 1
fi

#------------------------------------------------------------------------------------
# Finally we optimize sMBR criterion, we do Stochastic-GD with per-utterance updates. 
# For faster convergence, we re-generate the lattices after 1st epoch.
dir=exp/dnn8g_pretrain-dbn_dnn_run2_smbr_${exp}
srcdir=exp/dnn8f_pretrain-dbn_dnn_run2_${exp}
acwt=0.1
#
if [ $stage -le 10 ]; then
  # Generate lattices and alignments
  steps/nnet/align.sh --nj 30 --cmd "$train_cmd" \
    $train_bn_fmllr $lang $srcdir ${srcdir}_ali || exit 1;
  local/make_symlink_dir.sh --tmp-root $scratch ${srcdir}_denlats || exit 1
  steps/nnet/make_denlats.sh --nj 30 --cmd "$decode_cmd" --acwt $acwt \
    $train_bn_fmllr $lang $srcdir ${srcdir}_denlats  || exit 1;
fi
if [ $stage -le 11 ]; then
  # Train DNN by single iteration of sMBR (leaving out all silence frames), 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt --do-smbr true \
    --unkphonelist $silphonelist \
    $train_bn_fmllr $lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode
  for ITER in 1; do
    steps/nnet/decode.sh --nj $njdec --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
      --scoring-opts "$scoring" --num-threads 2 --parallel-opts "-pe smp 2" --max-mem 150000000 \
      --nnet $dir/${ITER}.nnet \
      $graph $dev_bn_fmllr $dir/decode_$(basename $dev_bn_fmllr)_it${ITER} || exit 1
  done
fi 

#------------------------------------------------------------------------------------
# Run 4 mode sMBR epochs after rebuilding lattices, alignment.
dir=exp/dnn8h_pretrain-dbn_dnn_run2_smbr_run2_${exp}
srcdir=exp/dnn8g_pretrain-dbn_dnn_run2_smbr_${exp}
acwt=0.1
#
if [ $stage -le 12 ]; then
  # Generate lattices and alignments
  steps/nnet/align.sh --nj 30 --cmd "$train_cmd" \
    $train_bn_fmllr $lang $srcdir ${srcdir}_ali || exit 1;
  local/make_symlink_dir.sh --tmp-root $scratch ${srcdir}_denlats || exit 1
  steps/nnet/make_denlats.sh --nj 30 --cmd "$decode_cmd" --acwt $acwt \
    $train_bn_fmllr $lang $srcdir ${srcdir}_denlats  || exit 1;
fi
if [ $stage -le 13 ]; then
  # Train DNN by 4 epochs of sMBR (leaving out all "unk" frames), 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt --do-smbr true \
    --unkphonelist $unkphonelist \
    $train_bn_fmllr $lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode
  for ITER in 1 2 3 4; do
    steps/nnet/decode.sh --nj $njdec --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
      --scoring-opts "$scoring" --num-threads 2 --parallel-opts "-pe smp 2" --max-mem 150000000 \
      --nnet $dir/${ITER}.nnet \
      $graph $dev_bn_fmllr $dir/decode_$(basename $dev_bn_fmllr)_it${ITER} || exit 1
  done 
fi

echo $0 successs.
exit 0 


