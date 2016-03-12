#!/bin/bash

# Copyright 2012-2014 Sri Harish Mallidi 

# This example script trains multistream ASR system
# The training is done as follows:
# 
# 1) Train a multistream bottleneck feature extractor
#    streams as sub-band streams
# 2) Train performance monitor
#    get the best performing bottleneck features
# All the following steps are motivated from Karel Vesely
# 3) Train a GMM on top of bottleneck features
# 4) fMMLR transform on bottleneck features
# 5) RBM pre-training:
#    in this unsupervised stage we train stack of RBMs, 
#    a good starting point for frame cross-entropy trainig.
# 6) frame cross-entropy training:
#    the objective is to classify frames to correct pdfs.
# 7) sequence-training optimizing sMBR: 
#    the objective is to emphasize state-sequences with better 
#    frame accuracy w.r.t. reference alignment.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

### Config:
njdec=60
scratch=/mnt/scratch03/tmp/$USER/

train_id=train_nodup
test_id=eval2000

lang=data/lang
lang_test=data/lang_sw1_tg

gmmdir=exp/tri4
ali_src=exp/tri4_ali_nodup
graph_src=exp/tri4/graph_sw1_tg

decode_bottleneck_featureXtractor=true

stage=0 # resume training with --stage=N
has_fisher=true

# multistream opts
strm_indices="0:30:60:90:120:150:180:210:246:276"
### End of config.
. utils/parse_options.sh || exit 1;
#

set -euxo pipefail

num_streams=`echo $strm_indices | awk -F ":" '{print NF-1}'`
all_stream_combn=`echo 2^$num_streams -1|bc`


# Extract multistream filterbank features
if [ $stage -le 0 ]; then
  ####
  # create multistream-fbank-config
  mkdir -p data-multistream-fbank/conf
  echo "--window-type=hamming" >data-multistream-fbank/conf/fbank_multistream.conf
  echo "--use-energy=false" >>data-multistream-fbank/conf/fbank_multistream.conf
  echo "--sample-frequency=8000" >>data-multistream-fbank/conf/fbank_multistream.conf

  echo "--dither=1" >>data-multistream-fbank/conf/fbank_multistream.conf

  echo "--num-mel-bins=46" >>data-multistream-fbank/conf/fbank_multistream.conf
  echo "--htk-compat=true" >>data-multistream-fbank/conf/fbank_multistream.conf
  ####

  c="$test_id"
  mkdir -p data-multistream-fbank/${c}; 
  cp data/${c}/{glm,reco2file_and_channel,segments,spk2utt,stm,stm.filt,text,utt2spk,wav.scp} data-multistream-fbank/${c}/
  steps/make_fbank.sh --fbank-config data-multistream-fbank/conf/fbank_multistream.conf \
    data-multistream-fbank/$c data-multistream-fbank/$c/log data-multistream-fbank/$c/data || exit 1
  steps/compute_cmvn_stats.sh \
    data-multistream-fbank/$c data-multistream-fbank/$c/log data-multistream-fbank/$c/data || exit 1

  c="$train_id"
  mkdir -p data-multistream-fbank/${c}; 
  cp data/${c}/{reco2file_and_channel,segments,spk2utt,text,utt2spk,wav.scp} data-multistream-fbank/${c}/
  steps/make_fbank.sh --fbank-config data-multistream-fbank/conf/fbank_multistream.conf \
    data-multistream-fbank/$c data-multistream-fbank/$c/log data-multistream-fbank/$c/data || exit 1
  steps/compute_cmvn_stats.sh \
    data-multistream-fbank/$c data-multistream-fbank/$c/log data-multistream-fbank/$c/data || exit 1
fi

## Train multistream bottleneck feature extractor
# This can take a lot of time
if [ $stage -le 1 ]; then
  local/multi-stream/multi-stream_bnfeats_train.sh \
    --ali $ali_src --train data-multistream-fbank/${train_id} \
    --strm-indices $strm_indices --iters-per-epoch 5 --dir exp/dnn5b_multistream_bottleneck_featureXtractor || exit 1;
fi


if [ $stage -le 2 ]; then
  # Train Performance monitor
  local/multi-stream/multi-stream_autoencoder_perf-monitor_train.sh --stage 1 \
    --train data-multistream-fbank/${train_id} --strm-indices $strm_indices \
    --nnet-dir exp/dnn5b_multistream_bottleneck_featureXtractor \
    --aann-dir exp/aann_tandem_dnn5b_multistream_bottleneck_featureXtractor || exit 1;
fi

# Get multi-stream masks
mask_dir=strm-mask/dnn5b_multistream_bottleneck_featureXtractor
train_mask_dir=$mask_dir/Comb${all_stream_combn}/${train_id}
test_mask_dir=$mask_dir/autoencoder_pm/${test_id}

if [ $stage -le 3 ]; then
  # train, no PM
  local/multi-stream/get-CombX_strm-mask.sh \
    --strm-indices "$strm_indices" --comb-num $all_stream_combn \
    --mask-dir $train_mask_dir --test data-multistream-fbank/${train_id} || exit 1;

  # test
  local/multi-stream/get_autoencoder-pm_strm-mask.sh --njdec $njdec \
    --test data-multistream-fbank/${test_id} \
    --strm-indices $strm_indices --tandem-transf-dir tandem_feats/dnn5b_multistream_bottleneck_featureXtractor_tandem_dim120/pca_transf \
    --aann-dir exp/aann_tandem_dnn5b_multistream_bottleneck_featureXtractor/aann \
    --mask-dir $test_mask_dir || exit 1;

fi

# Extract multistream bottleneck features
train=data-multistream-fbank/$train_id
train_bn=data-multistream-fbank-bn/$train_id

test=data-multistream-fbank/$test_id
test_bn=data-multistream-fbank-bn/$test_id

## decode test
# for logging
if [ $stage -le 4 ]; then

  if $decode_bottleneck_featureXtractor == "true"; then
    dir=exp/dnn5b_multistream_bottleneck_featureXtractor
    graph=$graph_src
    test=data-multistream-fbank/${test_id}

    multi_stream_opts="--cross-validate=true --stream-mask=scp:${test_mask_dir}/feats.scp $strm_indices"
    steps/multi-stream-nnet/decode.sh --nj $njdec --cmd "${decode_cmd}" --num-threads 3 \
      $graph $test "$multi_stream_opts" $dir/decode_$(basename $test)_$(basename $graph)_autoencoder_pm || exit 1;
  fi

  nnet_dir=exp/dnn5b_multistream_bottleneck_featureXtractor
  # train, 
  steps/multi-stream-nnet/make_bn_feats.sh --nj $njdec --cmd "$decode_cmd" \
    --remove-last-components 2 \
  $train_bn $train "--cross-validate=true --stream-mask=scp:${train_mask_dir}/feats.scp $strm_indices" $nnet_dir $train_bn/log $train_bn/data || exit 1
  steps/compute_cmvn_stats.sh $train_bn $train_bn/log $train_bn/data || exit 1;

  # test
  steps/multi-stream-nnet/make_bn_feats.sh --nj $njdec --cmd "$decode_cmd" \
    --remove-last-components 2 \
  $test_bn $test "--cross-validate=true --stream-mask=scp:${test_mask_dir}/feats.scp $strm_indices" $nnet_dir $test_bn/log $test_bn/data || exit 1
  steps/compute_cmvn_stats.sh $test_bn $test_bn/log $test_bn/data || exit 1;

fi

# train GMM BNF
dir=exp/dnn5b_multistream_bn-gmm
ali=$ali_src
if [ $stage -le 5 ]; then
  #Train
  # gmm on bn features, no cmvn, no lda-mllt,
  false && {  
  steps/train_deltas.sh --power 0.5 --boost-silence 1.5 --cmd "$train_cmd" \
    --delta-opts "--delta-order=0" \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --beam 20 --retry-beam 80 \
    6000 26000 $train_bn $lang $ali $dir || exit 1  
  }
  utils/mkgraph.sh $lang_test $dir $dir/$(basename $graph_src) || exit 1;
  steps/decode.sh --nj $njdec --cmd "$decode_cmd" \
    --num-threads 3 --parallel-opts "-pe smp 3" \
    --acwt 0.1 --beam 15.0 --lattice-beam 8.0 \
    $dir/$(basename $graph_src) $test_bn $dir/decode_$(basename $test_bn)_$(basename $graph_src) || exit 1

  # Align,
  steps/align_fmllr.sh --boost-silence 1.5 --nj 40 --cmd "$train_cmd" \
    --beam 20 --retry-beam 80 \
    $train_bn $lang $dir ${dir}_ali || exit 1;

fi

# Train SAT-adapted GMM on bottleneck features,
dir=exp/dnn5c_multistream_bn_fmllr-gmm
ali=exp/dnn5b_multistream_bn-gmm_ali

if [ $stage -le 6 ]; then
  # Train,
  # fmllr-gmm system on bottleneck features, 
  # - no cmvn, put fmllr to the features directly (no lda),
  # - note1 : we don't need cmvn, similar effect has diagonal of fmllr transform,
  # - note2 : lda+mllt was causing a small hit <0.5%,
  steps/train_sat.sh --power 0.5 --boost-silence 1.5 --cmd "$train_cmd" \
    --beam 20 --retry-beam 80 \
    6000 26000 $train_bn $lang $ali $dir || exit 1
  # Decode,
  utils/mkgraph.sh $lang_test $dir $dir/$(basename $graph_src) || exit 1;
  steps/decode_fmllr.sh --nj $njdec --cmd "$decode_cmd" \
    --num-threads 3 --parallel-opts "-pe smp 3" \
    --acwt 0.1 --beam 15.0 --lattice-beam 8.0 \
    $dir/$(basename $graph_src) $test_bn $dir/decode_$(basename $test_bn)_$(basename $graph_src) || exit 1

  # Align,
  steps/align_fmllr.sh --boost-silence 1.5 --nj 40 --cmd "$train_cmd" \
    --beam 20 --retry-beam 80 \
    $train_bn $lang $dir ${dir}_ali || exit 1;
fi

# Store the bottleneck-FMLLR features
gmm=exp/dnn5c_multistream_bn_fmllr-gmm # fmllr-feats, dnn-targets,
graph=$gmm/$(basename $graph_src)

train_bn_fmllr=data-multistream-fbank-bn_fmllr/$train_id
test_bn_fmllr=data-multistream-fbank-bn_fmllr/$test_id
if [ $stage -le 7 ]; then

  # Test set
  steps/nnet/make_fmllr_feats.sh --nj $njdec --cmd "$train_cmd" \
     --transform-dir $gmm/decode_$(basename $test_bn)_$(basename $graph_src) \
     $test_bn_fmllr $test_bn $gmm $test_bn_fmllr/log $test_bn_fmllr/data || exit 1;

  # Training set
  steps/nnet/make_fmllr_feats.sh --nj 30 --cmd "$train_cmd -tc 10" \
     --transform-dir ${gmm}_ali \
     $train_bn_fmllr $train_bn $gmm $train_bn_fmllr/log $train_bn_fmllr/data || exit 1;
fi


#------------------------------------------------------------------------------------
# Pre-train stack of RBMs (6 layers, 2048 units)
dir=exp/dnn5d_multistream_bn_pretrain-dbn
if [ $stage -le 8 ]; then
  # Create input transform, splice 13 frames [ -10 -5..+5 +10 ],
  mkdir -p $dir
  echo "<NnetProto>
        <Splice> <InputDim> 40 <OutputDim> 520 <BuildVector> -10 -5:1:5 10 </BuildVector>
        </NnetProto>" >$dir/proto.main
  # Do CMVN first, then frame pooling:
  nnet-concat "compute-cmvn-stats scp:${train_bn_fmllr}/feats.scp - | cmvn-to-nnet - - |" "nnet-initialize $dir/proto.main - |" $dir/transf.init || exit 1
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --feature-transform $dir/transf.init $train_bn_fmllr $dir || exit 1
fi

#------------------------------------------------------------------------------------
# Train the DNN optimizing cross-entropy.
dir=exp/dnn5e_multistream_bn_pretrain-dbn_dnn
feature_transform=exp/dnn5d_multistream_bn_pretrain-dbn/final.feature_transform # re-use
dbn=exp/dnn5d_multistream_bn_pretrain-dbn/6.dbn

ali=${gmm}_ali
graph=${gmm}/$(basename $graph_src)

if [ $stage -le 9 ]; then
  # Split the training set
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train_bn_fmllr ${train_bn_fmllr}_tr90 ${train_bn_fmllr}_cv10 
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 \
    ${train_bn_fmllr}_tr90 ${train_bn_fmllr}_cv10 $lang $ali $ali $dir || exit 1;
  
  # Decode
  steps/nnet/decode.sh --nj $njdec --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.10 \
    --num-threads 3 --parallel-opts "-pe smp 2" --max-mem 150000000 \
    $graph $test_bn_fmllr $dir/decode_$(basename $test_bn_fmllr)_$(basename $graph) || exit 1
fi

#------------------------------------------------------------------------------------
# Finally we optimize sMBR criterion, we do Stochastic-GD with per-utterance updates. 
# For faster convergence, we re-generate the lattices after 1st epoch.
dir=exp/dnn5f_multistream_bn_pretrain-dbn_dnn_smbr
srcdir=exp/dnn5e_multistream_bn_pretrain-dbn_dnn
acwt=0.1
graph=${gmm}/$(basename $graph_src)
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
      $train_bn_fmllr $lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1

  # Decode conversational.dev
  steps/nnet/decode.sh --nj $njdec --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.10 \
    --num-threads 3 --parallel-opts "-pe smp 2" --max-mem 150000000 \
    $graph $test_bn_fmllr $dir/decode_$(basename $test_bn_fmllr)_$(basename $graph) || exit 1

fi 

#------------------------------------------------------------------------------------
# Run 4 mode sMBR epochs after rebuilding lattices, alignment.
dir=exp/dnn5g_multistream_bn_pretrain-dbn_dnn_smbr_run2
srcdir=exp/dnn5f_multistream_bn_pretrain-dbn_dnn_smbr
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

  # Decode conversational.dev
  for ITER in 4 3 2 1; do
    steps/nnet/decode.sh --nj $njdec --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.10 \
      --nnet $dir/${ITER}.nnet \
      --num-threads 3 --parallel-opts "-pe smp 2" --max-mem 150000000 \
      $graph $test_bn_fmllr $dir/decode_$(basename $test_bn_fmllr)_$(basename $graph) || exit 1
  done

fi


exit 0;


if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,
  # eval2000
  dir=$data_fmllr/eval2000
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_eval2000_sw1_tg \
     $dir data/eval2000 $gmmdir $dir/log $dir/data
  # train
  dir=$data_fmllr/train_nodup
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali_nodup \
     $dir data/train_nodup $gmmdir $dir/log $dir/data
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10
fi

if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs
  dir=exp/dnn5b_pretrain-dbn
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --rbm-iter 1 $data_fmllr/train_nodup $dir
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn5b_pretrain-dbn_dnn
  ali=${gmmdir}_ali_nodup
  feature_transform=exp/dnn5b_pretrain-dbn/final.feature_transform
  dbn=exp/dnn5b_pretrain-dbn/6.dbn
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/train_nodup_tr90 $data_fmllr/train_nodup_cv10 data/lang $ali $ali $dir
  # Decode with the trigram swbd language model.
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" \
    --config conf/decode_dnn.config --acwt 0.08333 \
    $gmmdir/graph_sw1_tg $data_fmllr/eval2000 \
    $dir/decode_eval2000_sw1_tg
  if $has_fisher; then
    # Rescore with the 4gram swbd+fisher language model.
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_sw1_{tg,fsh_fg} data/eval2000 \
      $dir/decode_eval2000_sw1_{tg,fsh_fg}
  fi
fi


# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. The typical acwt value is around 0.1
dir=exp/dnn5b_pretrain-dbn_dnn_smbr
srcdir=exp/dnn5b_pretrain-dbn_dnn
acwt=0.0909

if [ $stage -le 3 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 250 --cmd "$train_cmd" \
    $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_ali
  steps/nnet/make_denlats.sh --nj 10 --sub-split 100 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --acwt $acwt $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_denlats
fi

if [ $stage -le 4 ]; then
  # Re-train the DNN by 1 iteration of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt --do-smbr true \
    $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir
  # Decode (reuse HCLG graph)
  for ITER in 4 3 2 1; do
    # Decode with the trigram swbd language model.
    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" \
      --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph_sw1_tg $data_fmllr/eval2000 \
      $dir/decode_eval2000_sw1_tg_it$ITER
    if $has_fisher; then
      # Rescore with the 4gram swbd+fisher language model.
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_sw1_{tg,fsh_fg} data/eval2000 \
        $dir/decode_eval2000_sw1_{tg,fsh_fg}_it$ITER
    fi
  done 
fi

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
