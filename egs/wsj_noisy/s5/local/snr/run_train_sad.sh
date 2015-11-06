#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

set -o pipefail

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
relu_dim=50
train_data_dir=data/train_si284_corrupted_hires
snr_scp=data/train_si284_corrupted_hires/snr_targets.scp
vad_scp=data/train_si284_corrupted_hires/vad.scp
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
    utils/copy_data_dir.sh --extra-files utt2uniq \
      $train_data_dir $datadir
    cp $snr_scp $datadir/feats.scp
    steps/compute_cmvn_stats.sh --fake $datadir $datadir/log snr
  fi

  if [ $method != "Gmm" ]; then 
    if [ $stage -le 1 ]; then
      mkdir -p $datadir/vad
      vad_scp_splits=()
      for n in `seq $nj`; do
        vad_scp_splits+=($datadir/vad/vad.tmp.$n.scp)
      done
      utils/split_scp.pl $vad_scp ${vad_scp_splits[@]} || exit 1

      cat <<EOF > $datadir/vad/vad_map
0 0
1 1
2 0
3 0
EOF

      $train_cmd JOB=1:$nj $datadir/vad/log/convert_vad.JOB.log \
        copy-int-vector scp:$datadir/vad/vad.tmp.JOB.scp ark,t:- \| \
        utils/apply_map.pl -f 2- $datadir/vad/vad_map \| \
        copy-int-vector ark,t:- \
        ark,scp:$datadir/vad/vad.JOB.ark,$datadir/vad/vad.JOB.scp || exit 1

      for n in `seq $nj`; do 
        cat $datadir/vad/vad.$n.scp
      done > $datadir/vad/vad.scp
    fi
    vad_scp=$datadir/vad/vad.scp

    if [ -z "$vad_scp" ] || [ ! -s $vad_scp ]; then
      echo "$0: $vad_scp file is empty!" && exit 1
    fi
  fi 
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
        $datadir $vad_scp $dir || exit 1
      ;;
    "LogisticRegressionSubsampled")
      $train_cmd --mem 8G $dir/log/train_logistic_regression.log \
        logistic-regression-train-on-feats --num-frames=8000000 --num-targets=2 \
        "ark:cat $datadir/feats.scp | splice-feats $splice_opts scp:- ark:- |" \
        scp:$vad_scp $dir/0.mdl || exit 1
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
        --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
        --cmd "$decode_cmd" --nj 40 --objective-type linear --use-presoftmax-prior-scale false \
        --skip-final-softmax false --skip-lda true --posterior-targets true \
        --num-targets 2 --cleanup false --max-param-change $max_param_change \
        $datadir "$vad_scp" $dir || exit 1;
      ;;
    "Dnn")
      if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
        utils/create_split_dir.pl \
          /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj_noisy-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
      fi

      bash -x steps/nnet3/train_tdnn_raw.sh --stage $train_stage \
        --num-epochs $num_epochs --num-jobs-initial 2 --num-jobs-final 14 \
        --splice-indexes "$splice_indexes" \
        --egs-dir "$egs_dir" ${feats_opts[@]} \
        --cmvn-opts "--norm-means=false --norm-vars=false" \
        --max-change-per-sample $max_change_per_sample \
        --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
        --cmd "$decode_cmd" --nj 40 --objective-type linear --cleanup false --use-presoftmax-prior-scale true \
        --skip-final-softmax false --skip-lda true --posterior-targets true \
        --num-targets 2 --cleanup false --max-param-change $max_param_change \
        --relu-dim $relu_dim \
        $datadir "$vad_scp" $dir || exit 1;
      ;;
    *)
      echo "Unknown method $method" 
      exit 1
  esac
fi

