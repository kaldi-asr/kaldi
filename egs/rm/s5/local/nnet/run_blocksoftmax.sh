#!/usr/bin/env bash

# Copyright 2012-2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains DNN with <BlockSoftmax> output on top of FBANK features.
# The network is trained on RM and WSJ84 simultaneously.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

dev=data-fbank-blocksoftmax/test
train=data-fbank-blocksoftmax/train
wsj=data-fbank-blocksoftmax/wsj
train_tr90_wsj=data-fbank-blocksoftmax/train_tr90_wsj

dev_original=data/test
train_original=data/train
wsj_original=../../wsj/s5/data/train_si284
[ ! -e $wsj_original ] && echo "Missing $wsj_original" && exit 1

gmm=exp/tri3b
wsj_ali=../../wsj/s5/exp/tri4b_ali_si284
[ ! -e $wsj_ali ] && echo "Missing $wsj_ali" && exit 1

stage=0
. utils/parse_options.sh || exit 1;

set -euxo pipefail

# Make the FBANK features,
[ ! -e $dev ] && if [ $stage -le 0 ]; then
  # Make datadir copies,
  utils/copy_data_dir.sh $dev_original $dev; rm $dev/{cmvn,feats}.scp
  utils/copy_data_dir.sh $train_original $train; rm $train/{cmvn,feats}.scp
  utils/copy_data_dir.sh --utt-prefix wsj --spk-prefix wsj $wsj_original $wsj; rm $wsj/{cmvn,feats}.scp
  
  # Feature extraction,
  # Dev set,
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" \
    $dev $dev/log $dev/data
  steps/compute_cmvn_stats.sh $dev $dev/log $dev/data
  # Training set,
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd --max-jobs-run 10" \
    $train $train/log $train/data
  steps/compute_cmvn_stats.sh $train $train/log $train/data
  # Wsj,
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd --max-jobs-run 10" \
    $wsj $wsj/log $wsj/data
  steps/compute_cmvn_stats.sh $wsj $wsj/log $wsj/data

  # Split the rm training set,
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train ${train}_tr90 ${train}_cv10
  # Merge-in the wsj set with train-set,
  utils/combine_data.sh $train_tr90_wsj ${train}_tr90 $wsj
fi


# Prepare the merged targets,
dir=exp/dnn4e-fbank_blocksoftmax
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
  # Mapping keys in wsj alignment to have prefix 'wsj',
  copy-int-vector "ark:gzcat ${wsj_ali}/ali.*.gz |" ark,t:- | awk -v prefix=wsj_ '{ $1=prefix $1; print; }' | \
    gzip -c >$dir/ali_wsj.gz 

  # Store single-stream posteriors to disk, indexed by 'scp' for pasting w/o caching,
  ali-to-pdf ${gmm}_ali/final.mdl "ark:gzcat ${gmm}_ali/ali.*.gz |" ark:- | \
    ali-to-post ark:- ark,scp:$dir/post1.ark,$dir/post1.scp
  ali-to-pdf ${wsj_ali}/final.mdl "ark:gzcat $dir/ali_wsj.gz |" ark:- | \
    ali-to-post ark:- ark,scp:$dir/post2.ark,$dir/post2.scp

  # Paste the posteriors from the 'scp' inputs,
  featlen="ark:feat-to-len 'scp:cat $train/feats.scp $wsj/feats.scp |' ark,t:- |"
  paste-post --allow-partial=true "$featlen" $ali1_dim:$ali2_dim \
    scp:$dir/post1.scp scp:$dir/post2.scp \
    ark,scp:$dir/pasted_post.ark,$dir/pasted_post.scp 2>$dir/log/paste_post.log
fi


# Train NN with '<BlockSoftmax>' output, we use 'MultiTask' objective function,
objw1=1; objw2=0.1; # we'll use lower weight for 'wsj' data,
if [ $stage -le 2 ]; then
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh \
      --cmvn-opts "--norm-means=true --norm-vars=true" \
      --delta-opts "--delta-order=2" --splice 5 \
      --labels "scp:$dir/pasted_post.scp" --num-tgt $output_dim \
      --proto-opts "--block-softmax-dims='$ali1_dim:$ali2_dim'" \
      --train-tool "nnet-train-frmshuff --objective-function=multitask,xent,$ali1_dim,$objw1,xent,$ali2_dim,$objw2" \
      --learn-rate 0.008 \
      ${train_tr90_wsj} ${train}_cv10 lang-dummy ali-dummy ali-dummy $dir
  # Create files used in decdoing, missing due to --labels use,
  analyze-counts --binary=false "$ali1_pdf" $dir/ali_train_pdf.counts
  copy-transition-model --binary=false $ali1_dir/final.mdl $dir/final.mdl
  cp $ali1_dir/tree $dir/tree
  # Rebuild network, <BlockSoftmax> is removed, and neurons from 1st block are selected,
  nnet-concat "nnet-copy --remove-last-components=1 $dir/final.nnet - |" \
    "echo '<Copy> <InputDim> $output_dim <OutputDim> $ali1_dim <BuildVector> 1:$ali1_dim </BuildVector>' | nnet-initialize - - |" \
    $dir/final.nnet.lang1
  # Decode (reuse HCLG graph),
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    --nnet $dir/final.nnet.lang1 \
    $gmm/graph $dev $dir/decode
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    --nnet $dir/final.nnet.lang1 \
    $gmm/graph_ug $dev $dir/decode_ug
fi

exit 0

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
      ${train}_tr90 ${train}_cv10 data/lang ${gmm}_ali ${gmm}_ali $dir
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmm/graph $dev $dir/decode
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmm/graph_ug $dev $dir/decode_ug
fi

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
