#!/bin/bash

# Copyright 2015  Vimal Manohar
# Apache 2.0

set -e 
set -o pipefail

. path.sh

# This script computes per-frame SNR from time-frequency bin SNR predicted 
# by an SNR predictor nnet and the original noisy fbank features

cmd=run.pl
nj=4
use_gpu=no
iter=final
prediction_type="FbankMask"
copy_opts= # Due to code change, the log(Irm) predicted might have previously been log(sqrt(Irm)). Hence use "matrix-scale --scale=2.0 ark:- ark:- \|". Also for log(Snr), it might have been log(sqrt(Snr)). 
stage=0

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <snr-nnet-dir> <corrupted-data-dir> <corrupted-fbank-dir> <dir>"
  echo " e.g.: $0 exp/nnet3_snr_predictor/nnet_tdnn_a data/train_si284_corrupted_hires data/train_si284_corrupted_fbank exp/frame_snrs_train_si284_corrupted"
  exit 1
fi

snr_predictor_nnet_dir=$1
corrupted_data_dir=$2
corrupted_fbank_dir=$3
dir=$4

split_data.sh $corrupted_data_dir $nj
split_data.sh $corrupted_fbank_dir $nj

sdata=$corrupted_data_dir/split$nj

cmvn_opts=$(cat $snr_predictor_nnet_dir/cmvn_opts 2>/dev/null) || exit 1

feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"

if [ -f $snr_predictor_nnet_dir/final.mat ]; then
  feat_type=lda
  
  splice_opts=`cat $snr_predictor_nnet_dir/splice_opts 2>/dev/null`
  feats="$feats splice-feats $splice_opts ark:- ark:- | transform-feats $snr_predictor_nnet_dir/final.mat ark:- ark:- |"
fi

gpu_cmd=$cmd
if [ $use_gpu != "no" ]; then
  gpu_cmd="$cmd --gpu 1"
fi

if [ $stage -le 0 ]; then
  $gpu_cmd JOB=1:$nj $dir/log/compute_nnet_pred.JOB.log \
    nnet3-compute --use-gpu=$use_gpu $snr_predictor_nnet_dir/$iter.raw "$feats" \
    ark:- \| ${copy_opts}copy-feats --compress=true ark:- ark:$dir/nnet_pred.JOB.ark || exit 1
fi

if [ $stage -le 1 ]; then
  case $prediction_type in 
    "Irm")
      # nnet_pred is log (clean energy / (clean energy + noise energy) )
      $cmd JOB=1:$nj $dir/log/compute_frame_snrs.JOB.log \
        compute-frame-snrs --prediction-type="Irm" \
        scp:$corrupted_fbank_dir/split$nj/JOB/feats.scp \
        ark:$dir/nnet_pred.JOB.ark \
        ark,scp:$dir/frame_snrs.JOB.ark,$dir/frame_snrs.JOB.scp \
        ark:$dir/clean_pred.JOB.ark
        
      ;;
    "FbankMask")
      # nnet_pred is log (clean feat / noisy feat)
      $cmd JOB=1:$nj $dir/log/compute_frame_snrs.JOB.log \
        compute-frame-snrs --prediction-type="FbankMask" \
        scp:$corrupted_fbank_dir/split$nj/JOB/feats.scp \
        ark:$dir/nnet_pred.JOB.ark \
        ark,scp:$dir/frame_snrs.JOB.ark,$dir/frame_snrs.JOB.scp \
        ark:$dir/clean_pred.JOB.ark
      ;;
    "FrameSnr")
      $cmd JOB=1:$nj $dir/log/compute_frame_snrs.JOB.log \
        extract-column 0 $dir/nnet_pred.JOB.ark \
        ark,scp:$dir/frame_snrs.JOB.ark,$dir/frame_snrs.JOB.scp || exit 1
      ;;
    "Snr")
      $cmd JOB=1:$nj $dir/log/compute_frame_snrs.JOB.log \
        compute-frame-snrs --prediction-type="Snr" \
        scp:$corrupted_fbank_dir/split$nj/JOB/feats.scp \
        ark:$dir/nnet_pred.JOB.ark \
        ark,scp:$dir/frame_snrs.JOB.ark,$dir/frame_snrs.JOB.scp \
        ark:$dir/clean_pred.JOB.ark
      ;;
    *)
      echo "Unknown prediction-type '$prediction_type'" && exit 1
  esac
fi

for n in `seq $nj`; do
  cat $dir/frame_snrs.$n.scp
done > $dir/frame_snrs.scp
