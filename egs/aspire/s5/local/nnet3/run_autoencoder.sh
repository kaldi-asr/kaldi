#!/bin/bash

# this is an example to show a "tdnn" system in raw nnet configuration
# i.e. without a transition model

. cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
affix=
train_stage=-10
common_egs_dir=
num_data_reps=10

remove_egs=true

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

dir=exp/nnet3/tdnn_raw
dir=$dir${affix:+_$affix}

clean_data_dir=data/train
data_dir=data/train_rvb
targets_scp=$dir/targets.scp

mkdir -p $dir

# Create copies of clean feats with prefix "rev$x_" to match utterance names of
# the noisy feats
for x in `seq 1 $num_data_reps`; do
  awk -v x=$x '{print "rev"x"_"$0}' $clean_data_dir/feats.scp | sort -k1,1 > $targets_scp
done

if [ $stage -le 9 ]; then
  echo "$0: creating neural net configs";
  
  num_targets=`feat-to-dim scp:$targets_scp - 2>/dev/null` || exit 1

  # create the config files for nnet initialization
  python steps/nnet3/tdnn/make_configs.py  \
     --splice-indexes "-2,-1,0,1,2 -1,2 -3,3 -7,2 0"  \
     --feat-dir ${data_dir} \
     --relu-dim=1024 \
     --add-lda=false \
     --objective-type=quadratic \
     --add-final-sigmoid=false \
     --include-log-softmax=false \
     --use-presoftmax-prior-scale=false \
     --num-targets=$num_targets \
     $dir/configs || exit 1;
fi

if [ $stage -le 10 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/tdnn/train_raw_nnet.sh --stage $train_stage \
    --cmd "$decode_cmd" \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-epochs 2 \
    --num-jobs-initial 3 \
    --num-jobs-final 16 \
    --initial-effective-lrate 0.0017 \
    --final-effective-lrate 0.00017 \
    --egs-dir "$common_egs_dir" \
    --remove-egs $remove_egs \
    --use-gpu true \
    --dense-targets true \
    ${data_dir} $targets_scp $dir || exit 1
fi

