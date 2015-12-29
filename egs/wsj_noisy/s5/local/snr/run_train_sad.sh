#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

set -o pipefail
set -e 
set -u

. cmd.sh

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
train_stage=-10
num_epochs=8
splice_indexes=`seq -s',' -50 50`
initial_effective_lrate=0.005
final_effective_lrate=0.0005
relu_dim=
sigmoid_dim=50
train_data_dir=data/train_si284_corrupted_hires
snr_scp=
vad_scp=
final_vad_scp=
max_change_per_sample=0.075
datadir=
egs_dir=
dir=
nj=40
method=LogisticRegression
splice_opts="--left-context=50 --right-context=50"
max_param_change=1
quantization_bins="-7.5:-2.5:2.5:7.5:12.5:17.5"
feat_type=
sparsity_constants=
positivity_constraints=
segments_file=
seg2utt_file=

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $method == "Dnn" ]; then
  num_hidden_layers=`echo $splice_indexes | perl -ane 'print scalar @F'` || exit 1
else
  num_hidden_layers=0
fi

if [ -z "$dir" ]; then
  dir=exp/nnet3_sad_snr/nnet_tdnn_a
fi

case $method in 
  "Dnn")
    dir=${dir} #_i${relu_dim}_n${num_hidden_layers}_lrate${initial_effective_lrate}_${final_effective_lrate}
    ;;
  "LogisticRegressionSubsampled")
    dir=${dir}
    ;;
  "LogisticRegression")
    dir=${dir}
    ;;
  "Gmm")
    dir=${dir}_gmm
    ;;
esac

if ! cuda-compiled; then
  cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi
  
mkdir -p $dir

if [ -z "$datadir" ]; then
  datadir=$dir/snr_data
  if [ $stage -le 0 ]; then
    rm -rf $datadir
    utils/copy_data_dir.sh --extra-files utt2uniq \
      $train_data_dir $datadir
    if [ ! -f $train_data_dir/segments ]; then
      if [ ! -z "$seg2utt_file" ]; then
        local/snr/create_segmented_data_dir_from_vad.sh \
          --cmd "$train_cmd" --nj $nj --feats $snr_scp \
          $train_data_dir $segments_file $seg2utt_file \
          $dir/snr_data/log $dir/snr_feats $datadir || exit 1

        steps/compute_cmvn_stats.sh --fake $datadir $dir/snr_data/log $dir/snr_feats

        #[ -z "$segments_file" ] && echo "$0: segments file is needed if --seg2utt-file is specified" && exit 1

        #rm -f $datadir/{cmvn.scp,feats.scp,utt2spk,utt2uniq,spk2utt,text}
        #utils/filter_scp.pl -f 2 $train_data_dir/utt2spk $segments_file > $datadir/segments.tmp
        #cat $datadir/segments.tmp | utils/apply_map.pl -f 2 $train_data_dir/utt2spk > $datadir/segments
        #utils/filter_scp.pl -f 2 $train_data_dir/utt2spk $seg2utt_file | \
        #  utils/apply_map.pl -f 2 $train_data_dir/utt2spk > $datadir/utt2spk
        #utils/utt2spk_to_spk2utt.pl $datadir/utt2spk > $datadir/spk2utt
        #
        #if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/snr_feats/storage ]; then
        #  utils/create_split_dir.pl \
        #    /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj_noisy-$(date +'%m_%d_%H_%M')/s5/$dir/snr_feats/storage $dir/snr_feats/storage
        #fi

        #$train_cmd JOB=1:$nj $dir/log/extract_feature_segments.JOB.log \
        #  extract-feature-segments scp:$snr_scp \
        #  "ark,t:utils/split_scp.pl -j $nj \$[JOB-1] $datadir/segments.tmp |" \
        #  ark:- \| copy-feats --compress=true ark:- \
        #  ark,scp:$dir/snr_feats/raw_snr.JOB.ark,$dir/snr_feats/raw_snr.JOB.scp

        #for n in `seq $nj`; do 
        #  cat $dir/snr_feats/raw_snr.$n.scp
        #done | sort -k1,1 > $datadir/feats.scp 

        #utils/fix_data_dir.sh $datadir
      else
        cp $snr_scp $datadir/feats.scp
      fi
    else 
      cp $snr_scp $datadir/feats.scp
    fi
    steps/compute_cmvn_stats.sh --fake $datadir $datadir/log snr
  fi

  if [ -z "$final_vad_scp" ] && [ $method != "Gmm" ]; then 
    if [ $stage -le 1 ]; then
      mkdir -p $dir/vad/split$nj
      if [ ! -z "$seg2utt_file" ]; then
        utils/filter_scp.pl -f 2 $train_data_dir/utt2spk $segments_file > $datadir/segments.tmp
        $train_cmd JOB=1:$nj $dir/log/extract_vad_segments.JOB.log \
          extract-int-vector-segments scp:$vad_scp \
          "ark,t:utils/split_scp.pl -j $nj \$[JOB-1] $datadir/segments.tmp |" \
          ark:- \| segmentation-init-from-ali ark:- ark:- \| \
          segmentation-post-process --merge-labels=0:2 --merge-dst-label=0 ark:- ark:- \| \
          segmentation-to-ali ark:- ark,scp:$dir/vad/split$nj/vad.JOB.ark,$dir/vad/split$nj/vad.JOB.scp || exit 1
      else
        vad_scp_splits=()
        for n in `seq $nj`; do
          vad_scp_splits+=($dir/vad/vad.tmp.$n.scp)
        done
        utils/split_scp.pl $vad_scp ${vad_scp_splits[@]} || exit 1

        cat <<EOF > $dir/vad/vad_map
0 0
1 1
2 0
3 0
EOF
        $train_cmd JOB=1:$nj $dir/vad/log/convert_vad.JOB.log \
          copy-int-vector scp:$dir/vad/vad.tmp.JOB.scp ark,t:- \| \
          utils/apply_map.pl -f 2- $dir/vad/vad_map \| \
          copy-int-vector ark,t:- \
          ark,scp:$dir/vad/split$nj/vad.JOB.ark,$dir/vad/split$nj/vad.JOB.scp || exit 1
      fi

      for n in `seq $nj`; do
        cat $dir/vad/split$nj/vad.$n.scp
      done | sort -k1,1 > $dir/vad/vad.scp
    fi
    final_vad_scp=$dir/vad/vad.scp
  fi 
fi

if [ ! -s $final_vad_scp ]; then
  echo "$0: $final_vad_scp file is empty!" && exit 1
fi

feats_opts=(--feat-type $feat_type)
#if [ "$feat_type" == "sparse" ]; then
#  if [ $stage -le 2 ]; then
#    split_data.sh $datadir $nj
#    mkdir -p $dir/quantized_data
#
#    $train_cmd JOB=1:$nj $dir/log/quantize_feats.JOB.log \
#      quantize-feats scp:$datadir/split$nj/JOB/feats.scp $quantization_bins \
#      ark,scp:$dir/quantized_data/feats.JOB.ark,$dir/quantized_data/feats.JOB.scp || exit 1
#
#    num_bins=`echo $quantization_bins | awk -F ':' '{print NF + 1}'` || exit 1
#    feat_dim=`feat-to-dim scp:$datadir/feats.scp - 2>/dev/null` || exit 1
#    sparse_input_dim=$[num_bins * feat_dim]
#
#    for n in `seq $nj`; do 
#      cat $dir/quantized_data/feats.$n.scp
#    done > $dir/quantized_data/feats.scp
#
#    echo $sparse_input_dim > $dir/quantized_data/sparse_input_dim
#  fi
#
#  sparse_input_dim=`cat $dir/quantized_data/sparse_input_dim`
#  feats_scp=$dir/quantized_data/feats.scp
#  feats_opts="--feat-type $feat_type --feats-scp $feats_scp --sparse-input-dim $sparse_input_dim"
#fi

if [ "$feat_type" == "sparse" ]; then
  feats_opts+=(--quantization-bin-boundaries "$quantization_bins")
fi

if [ $stage -le 3 ]; then
  case $method in
    "Gmm") 
      diarization/train_vad_gmm_supervised.sh \
        --ignore-energy false --add-zero-crossing-feats false \
        --add-frame-snrs false \
        --nj $nj --cmd "$train_cmd" \
        $datadir $final_vad_scp $dir || exit 1
      ;;
    "LogisticRegressionSubsampled")
      $train_cmd --mem 8G $dir/log/train_logistic_regression.log \
        logistic-regression-train-on-feats --num-frames=8000000 --num-targets=2 \
        "ark:cat $datadir/feats.scp | splice-feats $splice_opts scp:- ark:- |" \
        scp:$final_vad_scp $dir/0.mdl || exit 1
      ;;
    "LogisticRegression")
      if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
        utils/create_split_dir.pl \
          /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj_noisy-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
      fi

      steps/nnet3/train_tdnn_raw.sh --stage $train_stage \
        --num-epochs $num_epochs --num-jobs-initial 1 --num-jobs-final 4 \
        --splice-indexes "$splice_indexes" --no-hidden-layers true --minibatch-size 512 \
        --egs-dir "$egs_dir" "${feats_opts[@]}" \
        --cmvn-opts "--norm-means=false --norm-vars=false" \
        --max-param-change $max_param_change \
        --positivity-constraints "$positivity_constraints" \
        --sparsity-constants "$sparsity_constants" \
        --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
        --cmd "$decode_cmd" --nj 40 --objective-type linear --use-presoftmax-prior-scale false \
        --skip-final-softmax false --skip-lda true --posterior-targets true \
        --num-targets 2 --cleanup false --max-param-change $max_param_change \
        $datadir "$final_vad_scp" $dir || exit 1;
      ;;
    "Dnn")
      if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
        utils/create_split_dir.pl \
          /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj_noisy-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
      fi

      bash -x steps/nnet3/train_tdnn_raw.sh --stage $train_stage \
        --num-epochs $num_epochs --num-jobs-initial 2 --num-jobs-final 4 \
        --splice-indexes "$splice_indexes" \
        --egs-dir "$egs_dir" ${feats_opts[@]} \
        --cmvn-opts "--norm-means=false --norm-vars=false" \
        --max-param-change $max_param_change \
        --positivity-constraints "$positivity_constraints" \
        --sparsity-constants "$sparsity_constants" \
        --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
        --cmd "$decode_cmd" --nj 40 --objective-type linear --cleanup true --use-presoftmax-prior-scale true \
        --skip-final-softmax false --skip-lda true --posterior-targets true \
        --num-targets 2 --max-param-change $max_param_change \
        --cleanup false${relu_dim:+ --relu-dim $relu_dim}${sigmoid_dim:+ --sigmoid-dim $sigmoid_dim} \
        $datadir "$final_vad_scp" $dir || exit 1;
      ;;
    *)
      echo "Unknown method $method" 
      exit 1
  esac
fi

