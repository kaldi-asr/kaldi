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
splice_indexes=""
initial_effective_lrate=0.005
final_effective_lrate=0.0005
pnorm_input_dim=2000
pnorm_output_dim=250
train_data_dir=data/train_si284_corrupted_hires
snr_scp=data/train_si284_corrupted_hires/snr_targets.scp
vad_scp=data/train_si284_corrupted_hires/vad.scp
max_change_per_sample=0.075
egs_dir=
dir=
nj=40
method=LogisticRegression

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $method == "LogisticRegression" ]; then
  num_hidden_layers=`echo $splice_indexes | perl -ane 'print scalar @F'` || exit 1
else
  num_hidden_layers=0
fi

if [ -z "$dir" ]; then
  dir=exp/nnet3_sad_snr/nnet_tdnn_a
fi

case $method in 
  "Dnn")
    dir=${dir}_i${pnorm_input_dim}_o${pnorm_output_dim}_n${num_hidden_layers}_lrate${initial_effective_lrate}_${final_effective_lrate}
    ;;
  "LogisticRegression")
    dir=${dir}_lrate${initial_effective_lrate}_${final_effective_lrate}
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

if [ $stage -le 0 ]; then
  utils/copy_data_dir.sh --extra-files utt2uniq \
    $train_data_dir $dir/snr_data
  cp $snr_scp $dir/snr_data/feats.scp
  steps/compute_cmvn_stats.sh --fake $dir/snr_data $dir/snr_data/log snr
fi

if [ $method != "Gmm" ]; then 
  if [ $stage -le 1 ]; then
    mkdir -p $dir/vad
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
      ark,scp:$dir/vad/vad.JOB.ark,$dir/vad/vad.JOB.scp || exit 1

    for n in `seq $nj`; do 
      cat $dir/vad/vad.$n.scp
    done > $dir/vad/vad.scp
  fi
  vad_scp=$dir/vad/vad.scp

  if [ -z "$vad_scp" ] || [ ! -s $vad_scp ]; then
    echo "$0: $vad_scp file is empty!" && exit 1
  fi
fi 

if [ $stage -le 2 ]; then
  case $method in
    "Gmm") 
      diarization/train_vad_gmm_supervised.sh \
        --ignore-energy false --add-zero-crossing-feats false \
        --add-frame-snrs false \
        --nj $nj --cmd "$train_cmd" --io-opts "" \
        $dir/snr_data $vad_scp $dir || exit 1
      ;;
    "LogisticRegression")
      if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
        utils/create_split_dir.pl \
          /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj_noisy-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
      fi

      steps/nnet3/train_tdnn_raw.sh --stage $train_stage \
        --num-epochs $num_epochs --num-jobs-initial 2 --num-jobs-final 14 \
        --splice-indexes "" --no-hidden-layers true \
        --feat-type raw --egs-dir "$egs_dir" \
        --cmvn-opts "--norm-means=false --norm-vars=false" \
        --io-opts "--max-jobs-run 12" --max-change-per-sample $max_change_per_sample \
        --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
        --cmd "$decode_cmd" --nj 40 --objective-type linear \
        --skip-final-softmax false --skip-lda false --posterior-targets true \
        --num-targets 2 \
        --pnorm-input-dim $pnorm_input_dim \
        --pnorm-output-dim $pnorm_output_dim \
        $dir/snr_data "$vad_scp" $dir || exit 1;
      ;;
    "Dnn")
      if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
        utils/create_split_dir.pl \
          /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj_noisy-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
      fi

      steps/nnet3/train_tdnn_raw.sh --stage $train_stage \
        --num-epochs $num_epochs --num-jobs-initial 2 --num-jobs-final 14 \
        --splice-indexes "$splice_indexes" \
        --feat-type raw --egs-dir "$egs_dir" \
        --cmvn-opts "--norm-means=false --norm-vars=false" \
        --io-opts "--max-jobs-run 12" --max-change-per-sample $max_change_per_sample \
        --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
        --cmd "$decode_cmd" --nj 40 --objective-type linear \
        --skip-final-softmax false --skip-lda false --posterior-targets true \
        --pnorm-input-dim $pnorm_input_dim \
        --pnorm-output-dim $pnorm_output_dim \
        $dir/snr_data "$vad_scp" $dir || exit 1;
      ;;
    default)
      echo "Unknown method $method" 
      exit 1
  esac
fi

