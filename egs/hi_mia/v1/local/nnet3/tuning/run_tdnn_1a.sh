#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University (Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)
#           2020 AIShell-Foundation (Author: Bengu WU) 
#           2020 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU) 
# Apache 2.0
set -e
stage=0
train_stage=-10
affix=kws
common_egs_dir=

# training options
initial_effective_lrate=0.0015
final_effective_lrate=0.00015
num_epochs=2
num_jobs_initial=1
num_jobs_final=1
nj=30
remove_egs=true
num_targets=
. parse_options.sh || exit 1;

# End configuration section.

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

dir=exp/nnet3/tdnn_test${affix:+_$affix}
gmm_dir=exp/tri3
test_sets="fbank/dev fbank/test"
train_set=fbank/train
graph_dir=$gmm_dir/graph

if [ $stage -le 7 ]; then
  echo "$0: creating neural net configs";


  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=71 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=850
  relu-batchnorm-layer name=tdnn2 dim=850 input=Append(-1,0,2)
  relu-batchnorm-layer name=tdnn3 dim=850 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn4 dim=850 input=Append(-7,0,2)
  relu-batchnorm-layer name=tdnn5 dim=850 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn6 dim=850
  output-layer name=output input=tdnn6 dim=$num_targets max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
  echo "num_targets=$num_targets" >> $dir/configs/vars

fi

if [ $stage -le 8 ]; then
  #if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
  #  utils/create_split_dir.pl \
  #   /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aishell-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  #fi

  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --use-dense-targets false \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 500 \
    --use-gpu true \
    --egs.stage -10 \
    --feat-dir=data/fbank/train \
    --targets-scp=exp/kws_ali_test/ali.scp \
    --dir=$dir  || exit 1;
fi

echo "local/nnet3/run_tdnn.sh succeeded"
exit 0;
