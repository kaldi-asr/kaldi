#!/bin/bash

# Copyright 2012-2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains DNN with <MultiSoftmax> output on top of FBANK features.
# The network is trained on RM and WSJ84 simultaneously.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

dev=data-fbank-multisoftmax/test
train=data-fbank-multisoftmax/train
wsj=data-fbank-multisoftmax/wsj
train_tr90_wsj=data-fbank-multisoftmax/train_tr90_wsj

dev_original=data/test
train_original=data/train
wsj_original=../../wsj/s5/data/train_si284
[ ! -e $wsj_original ] && echo "Missing $wsj_original" && exit 1

gmm=exp/tri3b
wsj_ali=../../wsj/s5/exp/tri4b_ali_si284
[ ! -e $wsj_ali ] && echo "Missing $wsj_ali" && exit 1

stage=0
. utils/parse_options.sh || exit 1;

# Make the FBANK features,
if [ $stage -le 0 ]; then
  # Make datadir copies,
  utils/copy_data_dir.sh $dev_original $dev || exit 1; rm $dev/{cmvn,feats}.scp 2>/dev/null
  utils/copy_data_dir.sh $train_original $train || exit 1; rm $train/{cmvn,feats}.scp 2>/dev/null
  utils/copy_data_dir.sh --utt-prefix wsj --spk-prefix wsj $wsj_original $wsj || exit 1; rm $wsj/{cmvn,feats}.scp 2>/dev/null
  
  # Feature extraction,
  # Dev set,
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" \
    $dev $dev/log $dev/data || exit 1;
  steps/compute_cmvn_stats.sh $dev $dev/log $dev/data || exit 1;
  # Training set,
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd -tc 10" \
    $train $train/log $train/data || exit 1;
  steps/compute_cmvn_stats.sh $train $train/log $train/data || exit 1;
  # Wsj,
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd -tc 10" \
    $wsj $wsj/log $wsj/data || exit 1;
  steps/compute_cmvn_stats.sh $wsj $wsj/log $wsj/data || exit 1;

  # Split the rm training set,
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train ${train}_tr90 ${train}_cv10 || exit 1
  # Merge-in the wsj set with train-set,
  utils/combine_data.sh $train_tr90_wsj ${train}_tr90 $wsj || exit 1
fi

# Prepare the merged targets,
dir=exp/dnn4e-fbank_multisoftmax
ali1_dim=$(hmm-info ${gmm}_ali/final.mdl | grep pdfs | awk '{ print $NF }')
ali2_dim=$(hmm-info ${wsj_ali}/final.mdl | grep pdfs | awk '{ print $NF }')
#
output_dim=$((ali1_dim + ali2_dim))
#
ali1_pdf="ark:ali-to-pdf ${gmm}_ali/final.mdl 'ark:gzcat ${gmm}_ali/ali.*.gz |' ark:- |"
ali1_dir=${gmm}_ali
#
if [ $stage -le 1 ]; then
  mkdir -p $dir/log
  copy-int-vector "ark:gzcat ${wsj_ali}/ali.*.gz |" ark,t:- | awk -v prefix=wsj '{ $1=prefix $1; print; }' | gzip -c >$dir/ali_wsj.gz # Mapping utt key,
  featlen="ark:feat-to-len 'scp:cat $train/feats.scp $wsj/feats.scp |' ark,t:- |"
  ali1="ark:ali-to-pdf ${gmm}_ali/final.mdl 'ark:gzcat ${gmm}_ali/ali.*.gz |' ark:- | ali-to-post ark:- ark:- |"
  ali2="ark:ali-to-pdf ${wsj_ali}/final.mdl 'ark:gzcat $dir/ali_wsj.gz |' ark:- | ali-to-post ark:- ark:- |" 
  paste-post "$featlen" $ali1_dim:$ali2_dim "$ali1" "$ali2" ark,scp:$dir/pasted_post.ark,$dir/pasted_post.scp 2>$dir/log/paste_post.log || exit 1
fi

# Train <MultiSoftmax> system,
if [ $stage -le 2 ]; then
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh \
      --cmvn-opts "--norm-means=true --norm-vars=true" \
      --delta-opts "--delta-order=2" --splice 5 \
      --labels "scp:$dir/pasted_post.scp" --num-tgt $output_dim \
      --proto-opts "--block-softmax-dims='$ali1_dim:$ali2_dim'" \
      --learn-rate 0.008 \
      ${train_tr90_wsj} ${train}_cv10 lang-dummy ali-dummy ali-dummy $dir || exit 1;
  # Create files used in decdoing, missing due to --labels use,
  analyze-counts --binary=false "$ali1_pdf" $dir/ali_train_pdf.counts || exit 1
  copy-transition-model --binary=false $ali1_dir/final.mdl $dir/final.mdl || exit 1
  cp $ali1_dir/tree $dir/tree || exit 1
  # Rebuild network, <MultiSoftmax> is removed, and neurons from 1st block are selected,
  nnet-concat "nnet-copy --remove-last-components=1 $dir/final.nnet - |" \
    "echo '<Copy> <InputDim> $output_dim <OutputDim> $ali1_dim <BuildVector> 1:$ali1_dim </BuildVector>' | nnet-initialize - - |" \
    $dir/final.nnet.lang1 || exit 1
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    --nnet $dir/final.nnet.lang1 \
    $gmm/graph $dev $dir/decode || exit 1;
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    --nnet $dir/final.nnet.lang1 \
    $gmm/graph_ug $dev $dir/decode_ug || exit 1;
fi

# TODO, 
# make nnet-copy support block selection, 
# - either by replacing <BlockSoftmax> by <Softmax> and shrinking <AffineTransform>,
# - or by appending <Copy> transform,
#
# Will it be compatible with other scripts/tools which assume <Softmax> at the end?
# Or is it better to do everything visually in master script as now?... 
# Hmmm, need to think about it...

# Train baseline system with <Softmax>,
if [ $stage -le 3 ]; then
  dir=exp/dnn4e-fbank_baseline
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh \
      --cmvn-opts "--norm-means=true --norm-vars=true" \
      --delta-opts "--delta-order=2" --splice 5 \
      --learn-rate 0.008 \
      ${train}_tr90 ${train}_cv10 data/lang ${gmm}_ali ${gmm}_ali $dir || exit 1;
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmm/graph $dev $dir/decode || exit 1;
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmm/graph_ug $dev $dir/decode_ug || exit 1;
fi

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
