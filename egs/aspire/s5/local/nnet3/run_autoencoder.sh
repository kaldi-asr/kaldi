#!/usr/bin/env bash

# this is an example to show a "tdnn" system in raw nnet configuration
# i.e. without a transition model
# It uses corrupted (reverberation + noise) speech as input and clean speech 
# as output.

. ./cmd.sh

stage=0
affix=
train_stage=-10
common_egs_dir=
egs_opts=
num_data_reps=10

remove_egs=true

. ./cmd.sh
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

if [ -e $targets_scp.unsorted ]; then
  rm $targets_scp.unsorted
fi

# Create copies of clean feats with prefix "rev$x-" to match utterance names of
# the noisy feats
for x in `seq 1 $num_data_reps`; do
  awk -v x=$x '{print "rev"x"-"$0}' $clean_data_dir/feats.scp >> $targets_scp.unsorted
done
sort -k1,1 $targets_scp.unsorted > $targets_scp
rm $targets_scp.unsorted

if [ $stage -le 9 ]; then
  echo "$0: creating neural net configs";
  num_targets=`feat-to-dim scp:$targets_scp - 2>/dev/null` || exit 1
  feat_dim=`feat-to-dim scp:$data_dir/feats.scp - 2>/dev/null` || exit 1

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=$feat_dim name=input

  relu-renorm-layer name=tdnn1 dim=1024 input=Append(-2,-1,0,1,2)
  relu-renorm-layer name=tdnn2 dim=1024 input=Append(-1,2)
  relu-renorm-layer name=tdnn3 dim=1024 input=Append(-3,3)
  relu-renorm-layer name=tdnn4 dim=1024 input=Append(-7,2)
  relu-renorm-layer name=tdnn5 dim=1024
  output-layer name=output dim=$num_targets max-change=1.5 objective-type=quadratic include-log-softmax=false
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 10 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.0017 \
    --trainer.optimization.final-effective-lrate 0.00017 \
    --trainer.optimization.minibatch-size 512 \
    --egs.dir "$common_egs_dir" --egs.opts "$egs_opts" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 50 \
    --nj=30 \
    --use-dense-targets=true \
    --feat-dir=${data_dir} \
    --targets-scp=$targets_scp \
    --dir=$dir || exit 1;
fi
