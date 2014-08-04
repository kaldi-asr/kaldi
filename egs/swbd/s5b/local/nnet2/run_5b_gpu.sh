#!/bin/bash


stage=0
train_stage=-100
temp_dir=

# This (run_5b_gpu.sh) is comparable to run_5a_gpu.sh in that it runs on the 100
# hour features, but this recipe trains only unadapted (just cepstral mean
# normalized) features, and uses various combinations of VTLN warping factor and
# time-warping factor to artificially expand the amount of data.



. ./cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
. utils/parse_options.sh  # to parse the --stage option, if given

if [ $# != 0 ]; then
  echo "Usage: local/run_4b.sh [--stage <stage> --train-stage <train-stage>]"
  echo "Options: "
  echo "    --stage <stage>              # controls partial re-runs"
  echo "    --train-stage <train-stage>  #  use with --stage 2 to control partial rerun of training"
  echo "    --temp-dir <temp-dir>        # e.g. --temp-dir /export/my-machine/dpovey/wsj-temp-5b"
  echo "                                 # (puts temporary data including MFCC and egs at this location)"
  exit 1;
fi

set -e

if [ $stage -le 0 ]; then 
  # Create the training data.
  
  if [ ! -z "$temp_dir" ]; then
    mkdir -p $temp_dir/mfcc_5b_gpu
    featdir=$temp_dir/mfcc_5b_gpu
  else
    featdir=`pwd`/mfcc/nnet5b_gpu; 
    mkdir -p $featdir
  fi
  fbank_conf=conf/fbank_40.conf
  echo "--num-mel-bins=40" > $fbank_conf
  steps/nnet2/get_perturbed_feats.sh --cmd "$train_cmd" \
    $fbank_conf $featdir exp/perturbed_fbanks_100k_nodup data/train_100k_nodup data/train_100k_nodup_perturbed_fbank &
  steps/nnet2/get_perturbed_feats.sh --cmd "$train_cmd" --feature-type mfcc \
    conf/mfcc.conf $featdir exp/perturbed_mfcc_100k_nodup data/train_100k_nodup data/train_100k_nodup_perturbed_mfcc &
  wait
fi

if [ $stage -le 1 ]; then
  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/train_100k_nodup_perturbed_mfcc data/lang exp/tri4b exp/tri4b_ali_100k_nodup_perturbed_mfcc
fi 

if [ $stage -le 2 ]; then
  if [ ! -z "$temp_dir" ] && [ ! -e exp/nnet5b_gpu/egs ]; then
    mkdir -p exp/nnet5b_gpu
    mkdir -p $temp_dir/nnet5b_gpu/egs
    ln -s $temp_dir/nnet5b/egs exp/nnet5b_gpu/
  fi

  steps/nnet2/train_block.sh --stage "$train_stage" \
     --num-threads 1 --max-change 40.0 --minibatch-size 512 \
     --parallel-opts "-l gpu=1" \
     --initial-learning-rate 0.01 --final-learning-rate 0.001 \
     --num-epochs 10 --num-epochs-extra 5 \
     --cmd "$decode_cmd" \
     --hidden-layer-dim 1536 \
     --num-block-layers 3 --num-normal-layers 3 \
      data/train_100k_nodup_perturbed_fbank data/lang exp/tri4b_ali_100k_nodup_perturbed_mfcc exp/nnet5b_gpu  || exit 1
fi

if [ $stage -le 3 ]; then # create testing fbank data.
  featdir=`pwd`/mfcc
  fbank_conf=conf/fbank_40.conf
  for x in test_eval92 test_eval93 test_dev93; do 
    cp -rT data/$x data/${x}_fbank
    rm -r ${x}_fbank/split* || true
    steps/make_fbank.sh --fbank-config "$fbank_conf" --nj 8 \
      --cmd "$train_cmd" data/${x}_fbank exp/make_fbank/$x $featdir  || exit 1;
    steps/compute_cmvn_stats.sh data/${x}_fbank exp/make_fbank/$x $featdir  || exit 1;
  done
fi

if [ $stage -le 4 ]; then
  for lm_suffix in tg fsh_tgpr; do
    steps/decode.sh --cmd "$decode_cmd" --nj 30 --config conf/decode.config \
      exp/tri4a/graph_sw1_${lm_suffix} data/eval2000 exp/nnet5b_gpu/decode_eval2000_sw1_${lm_suffix} &
  done
  wait
fi


exit 0;
