#!/bin/bash

. ./cmd.sh
. ./path.sh 

stage=1
scratch=/export/a06/$USER/tmp/

nj=30

lang=data/lang

# source data,
ali_src=exp/tri2b_multi_ali_si84
graph_src=exp/tri2b_multi/graph_tgpr_5k

exp="fbank-traps_mstrm_9strms-2BarkPerStrm_CMN_bnfeats_splice5_traps_dct_basis6_iters-per-epoch5"
bn_nnet_proto=
append_feature_transform=

# bnfeatXtractor opts
bn_splice=5
bn_traps_dct_basis=6

# fbank features,
train_id=train_si84_multi

#mstrm opts
strm_indices="0:30:60:90:120:150:186:216:252:378"
scheduler_opts="--iters-per-epoch 5"

. utils/parse_options.sh

train_bn=data-fbank-bn-${exp}/${train_id}
train_bn_fmllr=data-fbank-bn-fmllr-${exp}/${train_id}

# list of unk phones, for frame selection:
unkphonelist=$(grep UNK $lang/phones.txt | awk '{ print $2; }' | tr '\n' ':' | sed 's|:$||')
silphonelist=$(cat $lang/phones/silence.csl)

##############################
# number of stream combinations
num_streams=`echo $strm_indices | awk -F ":" '{print NF-1}'`
all_stream_combn=`echo 2^$num_streams -1|bc`

## Filterbank creation and 
# preprocessing steps
if [ $stage -le 0 ]; then
  ####
  # create multistream-fbank-config
  mkdir -p data-multistream-fbank/conf
  echo "--window-type=hamming" >data-multistream-fbank/conf/fbank_multistream.conf
  echo "--use-energy=false" >>data-multistream-fbank/conf/fbank_multistream.conf
  echo "--sample-frequency=16000" >>data-multistream-fbank/conf/fbank_multistream.conf

  echo "--dither=1" >>data-multistream-fbank/conf/fbank_multistream.conf

  echo "--num-mel-bins=63" >>data-multistream-fbank/conf/fbank_multistream.conf
  echo "--htk-compat=true" >>data-multistream-fbank/conf/fbank_multistream.conf
  ####

  c="$train_id"
  mkdir -p data-multistream-fbank/${c}; 
  cp data/${c}/{spk2utt,spk2gender,text,utt2spk,wav.scp} data-multistream-fbank/${c}/
  steps/make_fbank.sh --fbank-config data-multistream-fbank/conf/fbank_multistream.conf \
    data-multistream-fbank/$c data-multistream-fbank/$c/log data-multistream-fbank/$c/data || exit 1
  steps/compute_cmvn_stats.sh \
    data-multistream-fbank/$c data-multistream-fbank/$c/log data-multistream-fbank/$c/data || exit 1

fi

train=data-multistream-fbank/$train_id

dir=exp/dnn8a_bn-feat_${exp}
ali=$ali_src

# BNfeats neural network
if [ $stage -le 1 ]; then
  utils/subset_data_dir_tr_cv.sh ${train} ${train}_tr90 ${train}_cv10 || exit 1;

  #create append_feature_transform
  num_fbank=$(feat-to-dim "ark:copy-feats scp:${train}/feats.scp ark:- |" -)

  mkdir -p $dir
  if [ -z "$append_feature_transform" ]; then
python -c "
num_fbank=${num_fbank}
traps_dct_basis=${bn_traps_dct_basis}
print '<Nnet>'
print '<Copy> '+str(num_fbank*traps_dct_basis)+' '+str(num_fbank*traps_dct_basis)
print '[',
for i in xrange(0, num_fbank):
  for j in xrange(0, traps_dct_basis):
    print (i+num_fbank*j)+1,
print ']'
print '</Nnet>'
" >$dir/append_rearrange_subband-group_hamm_dct_${num_fbank}Fbank_${bn_traps_dct_basis}dctbasis.nnet 

append_feature_transform="$dir/append_rearrange_subband-group_hamm_dct_${num_fbank}Fbank_${bn_traps_dct_basis}dctbasis.nnet"
  fi

  $cuda_cmd $dir/log/train_nnet.log \
  steps/multi-stream-nnet/train_new.sh --copy-feats-tmproot /mnt/data/$USER/tmp.XXXXX \
    ${labels:+ --labels "$labels" --num-tgt "$num_tgt"} \
    --scheduler-opts "$scheduler_opts" \
    --cmvn-opts "--norm-means=true --norm-vars=false" \
    --feat-type traps --splice $bn_splice --traps-dct-basis $bn_traps_dct_basis --learn-rate 0.008 \
    ${bn_nnet_proto:+ --nnet-proto $bn_nnet_proto} ${append_feature_transform:+ --append-feature-transform "$append_feature_transform"} \
    --proto-opts "--bottleneck-before-last-affine" --bn-dim 40 --hid-dim 1500 --hid-layers 3 \
    ${train}_tr90 ${train}_cv10 $lang $ali $ali $strm_indices $dir || exit 1

fi

#Extract bottleneck feats
if [ $stage -le 2 ]; then
  local/make_symlink_dir.sh --tmp-root $scratch $train_bn/data
  steps/multi-stream-nnet/make_bn_feats.sh --nj $nj --cmd "$train_cmd" \
    --remove-last-components 2 \
    $train_bn $train "--cross-validate=true --stream-combination=$all_stream_combn $strm_indices" $dir $train_bn/log $train_bn/data || exit 1
  steps/compute_cmvn_stats.sh $train_bn $train_bn/log $train_bn/data || exit 1;

fi

dir=exp/dnn8b_bn-gmm_${exp}
# Train GMM on bottleneck features,
if [ $stage -le 3 ]; then
  # Train,
  # gmm on bn features, no cmvn, no lda-mllt,
  steps/train_deltas.sh --power 0.5 --boost-silence 1.5 --cmd "$train_cmd" \
    --delta-opts "--delta-order=0" \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --beam 20 --retry-beam 80 \
    6000 26000 $train_bn $lang $ali_src $dir || exit 1

  # Align,
  steps/align_fmllr.sh --boost-silence 1.5 --nj 40 --cmd "$train_cmd" \
    --beam 20 --retry-beam 80 \
    $train_bn $lang $dir ${dir}_ali || exit 1;
fi


#########################

# Train SAT-adapted GMM on bottleneck features,
dir=exp/dnn8c_fmllr-gmm_${exp}
ali=exp/dnn8b_bn-gmm_${exp}_ali
if [ $stage -le 4 ]; then
  # Train,
  # fmllr-gmm system on bottleneck features, 
  # - no cmvn, put fmllr to the features directly (no lda),
  # - note1 : we don't need cmvn, similar effect has diagonal of fmllr transform,
  # - note2 : lda+mllt was causing a small hit <0.5%,
  steps/train_sat.sh --power 0.5 --boost-silence 1.5 --cmd "$train_cmd" \
    --beam 20 --retry-beam 80 \
    6000 26000 $train_bn $lang $ali $dir || exit 1

  # Align,
  steps/align_fmllr.sh --boost-silence 1.5 --nj 40 --cmd "$train_cmd" \
    --beam 20 --retry-beam 80 \
    $train_bn $lang $dir ${dir}_ali || exit 1;
fi

# Store the bottleneck-FMLLR features
gmm=exp/dnn8c_fmllr-gmm_${exp} # fmllr-feats, dnn-targets,
graph=$gmm/graph
if [ $stage -le 5 ]; then
  # Training set
  steps/nnet/make_fmllr_feats.sh --nj 30 --cmd "$train_cmd -tc 10" \
     --transform-dir ${gmm}_ali \
     $train_bn_fmllr $train_bn $gmm $train_bn_fmllr/log $train_bn_fmllr/data || exit 1;
fi

#------------------------------------------------------------------------------------
# Pre-train stack of RBMs (6 layers, 2048 units)
dir=exp/dnn8d_pretrain-dbn_${exp}
if [ $stage -le 6 ]; then
  # Create input transform, splice 13 frames [ -10 -5..+5 +10 ],
  mkdir -p $dir
  echo "<NnetProto>
        <Splice> <InputDim> 40 <OutputDim> 520 <BuildVector> -10 -5:1:5 10 </BuildVector>
        </NnetProto>" >$dir/proto.main
  # Do CMVN first, then frame pooling:
  nnet-concat "compute-cmvn-stats scp:${train_bn_fmllr}/feats.scp - | cmvn-to-nnet - - |" "nnet-initialize $dir/proto.main - |" $dir/transf.init || exit 1
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --copy-feats-tmproot /mnt/data/$USER/tmp.XXXXX --hid-dim 1024 --feature-transform $dir/transf.init $train_bn_fmllr $dir || exit 1
fi

#------------------------------------------------------------------------------------
# Train the DNN optimizing cross-entropy.
dir=exp/dnn8e_pretrain-dbn_dnn_${exp}
feature_transform=exp/dnn8d_pretrain-dbn_${exp}/final.feature_transform # re-use
dbn=exp/dnn8d_pretrain-dbn_${exp}/6.dbn # re-use

#ali=$ali_src
ali=${gmm}_ali

if [ $stage -le 7 ]; then
  # Split the training set
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train_bn_fmllr ${train_bn_fmllr}_tr90 ${train_bn_fmllr}_cv10 
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --copy-feats-tmproot /mnt/data/$USER/tmp.XXXXX \
    ${train_bn_fmllr}_tr90 ${train_bn_fmllr}_cv10 $lang $ali $ali $dir || exit 1;
fi

#------------------------------------------------------------------------------------
# Finally we optimize sMBR criterion, we do Stochastic-GD with per-utterance updates. 
# For faster convergence, we re-generate the lattices after 1st epoch.
dir=exp/dnn8f_pretrain-dbn_dnn_smbr_${exp}
srcdir=exp/dnn8e_pretrain-dbn_dnn_${exp}
acwt=0.1
#
if [ $stage -le 8 ]; then
  # Generate lattices and alignments
  steps/nnet/align.sh --nj $nj --cmd "$train_cmd" \
    $train_bn_fmllr $lang $srcdir ${srcdir}_ali || exit 1;
  local/make_symlink_dir.sh --tmp-root $scratch ${srcdir}_denlats || exit 1
  steps/nnet/make_denlats.sh --nj $nj --cmd "$decode_cmd" --acwt $acwt \
    $train_bn_fmllr $lang $srcdir ${srcdir}_denlats  || exit 1;
fi
if [ $stage -le 9 ]; then
  # Train DNN by single iteration of sMBR (leaving out all silence frames), 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt --do-smbr true \
    --unkphonelist $silphonelist \
    $train_bn_fmllr $lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
fi 

exit 0;


