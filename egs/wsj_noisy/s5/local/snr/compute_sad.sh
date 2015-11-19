#!/bin/bash

# Copyright 2015  Vimal Manohar
# Apache 2.0

set -o pipefail

. cmd.sh
. path.sh 

method=LogisticRegression
nj=40
stage=-10
iter=final
splice_opts="--left-context=10 --right-context=10"
model_dir=exp/nnet3_sad_snr/tdnn_train_si284_corrupted_splice21
snr_pred_dir=exp/frame_snrs_lwr_snr_reverb_dev_aspire_whole/
dir=exp/nnet3_sad_snr/sad_train_si284_corrupted
quantization_bins=-2.5:2.5:7.5:12.5:17.5

. utils/parse_options.sh 

if [ $# -ne 3 ]; then
  echo "Usage: $0 <sad-model-dir> <snr-pred-dir> <dir>"
  echo " e.g.: $0 $model_dir $snr_pred_dir $dir"
  exit 1
fi

model_dir=$1
snr_pred_dir=$2
dir=$3

if [ ! -s $snr_pred_dir/nnet_pred_snrs.scp ]; then  
  echo "$0: Could not read $snr_pred_dir/nnet_pred_snrs.scp or it is empty" 
  exit 1
fi

feat_type=`cat $model_dir/feat_type` || exit 1

echo $nj > $dir/num_jobs

if [ $stage -le 1 ]; then
  case $method in 
    "LogisticRegressionSubsampled")
      model=$model_dir/$iter.mdl

      $decode_cmd --mem 8G JOB=1:$nj $dir/log/eval_logistic_regression.JOB.log \
        logistic-regression-eval-on-feats "$model" \
        "ark:utils/split_scp.pl -j $nj \$[JOB-1] $snr_pred_dir/nnet_pred_snrs.scp | splice-feats $splice_opts scp:- ark:- |" \
        ark,scp:$dir/log_posteriors.JOB.ark,$dir/log_posteriors.JOB.scp || exit 1
      ;;
    "LogisticRegression"|"Dnn")
      model=$model_dir/$iter.raw

      if [ $feat_type != "sparse" ]; then
        $decode_cmd --mem 8G JOB=1:$nj $dir/log/eval_tdnn.JOB.log \
          nnet3-compute --apply-exp=false --use-gpu=no "$model" \
          "scp:utils/split_scp.pl -j $nj \$[JOB-1] $snr_pred_dir/nnet_pred_snrs.scp |" \
          ark,scp:$dir/log_posteriors.JOB.ark,$dir/log_posteriors.JOB.scp || exit 1
      else 
        num_bins=`echo $quantization_bins | awk -F ':' '{print NF + 1}' 2>/dev/null` || exit 1
        feat_dim=`head -n 1 $snr_pred_dir/nnet_pred_snrs.scp | feat-to-dim scp:- - 2>/dev/null` || exit 1
        sparse_input_dim=$[num_bins * feat_dim]

        train_num_bins=`cat $model_dir/quantization_bin_boundaries | awk -F ':' '{print NF + 1}' 2>/dev/null` || exit 1

        if [ $num_bins -ne $train_num_bins ]; then
          echo "$0: Mismatch in number of bins during test and train; $num_bins vs $train_num_bins"
          exit 1
        fi

        $decode_cmd --mem 8G JOB=1:$nj $dir/log/eval_tdnn.JOB.log \
          nnet3-compute-from-sparse-input --apply-exp=false --use-gpu=no --sparse-input-dim=$sparse_input_dim "$model" \
          "ark:utils/split_scp.pl -j $nj \$[JOB-1] $snr_pred_dir/nnet_pred_snrs.scp | quantize-feats scp:- $quantization_bins ark:- |" \
          ark,scp:$dir/log_posteriors.JOB.ark,$dir/log_posteriors.JOB.scp || exit 1
      fi
      ;;
    *)
      echo "Unknown method $method" 
      exit 1
  esac
fi

if [ $stage -le 2 ]; then 
  $decode_cmd JOB=1:$nj $dir/log/extract_logits.JOB.log \
    vector-sum "ark:extract-column --column-index=1 scp:$dir/log_posteriors.JOB.scp ark:- |" \
    "ark:extract-column --column-index=0 scp:$dir/log_posteriors.JOB.scp ark:- | vector-scale --scale=-1 ark:- ark:- |" \
    ark,scp:$dir/logits.JOB.ark,$dir/logits.JOB.scp || exit 1

  $decode_cmd JOB=1:$nj $dir/log/extract_prob.JOB.log \
    vector-apply-log --invert=true \
    "ark:extract-column --column-index=1 scp:$dir/log_posteriors.JOB.scp ark:- |" \
    ark,scp:$dir/speech_prob.JOB.ark,$dir/speech_prob.JOB.scp || exit 1
fi
