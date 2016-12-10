#!/bin/bash

set -o pipefail
set -e 
set -u

. cmd.sh

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
train_stage=-10
get_egs_stage=-10
egs_opts=

splice_indexes="-3,-2,-1,0,1,2,3 -6,0 -9,0,3 0"
relu_dim=256

# training options
num_epochs=20
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=3
num_jobs_final=8
samples_per_iter=400000
remove_egs=false
max_param_change=1

num_utts_subset_valid=6
num_utts_subset_train=6

use_dense_targets=false
extra_egs_copy_cmd="nnet3-copy-egs-overlap-detection ark:- ark:- |"

# target options
train_data_dir=data/sdm1/train_whole_sp_hires_bp
targets_scp=exp/sdm1/overlap_speech_train_cleaned_sp/overlap_feats.scp
deriv_weights_scp=exp/sdm1/overlap_speech_train_cleaned_sp/deriv_weights.scp
egs_dir=
nj=40
feat_type=raw
config_dir=
compute_objf_opts=

mic=sdm1
dir=
affix=a

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

num_hidden_layers=`echo $splice_indexes | perl -ane 'print scalar @F'` || exit 1
if [ -z "$dir" ]; then
  dir=exp/$mic/nnet3_raw/nnet_tdnn
fi

dir=$dir${affix:+_$affix}_n${num_hidden_layers}


if ! cuda-compiled; then
  cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi
  
mkdir -p $dir

if [ $stage -le 3 ]; then
  steps/nnet3/tdnn/make_raw_configs.py \
    --self-repair-scale-nonlinearity=0.00001 \
    --feat-dir=$train_data_dir --num-targets=3 \
    --relu-dim=$relu_dim \
    --splice-indexes="$splice_indexes" \
    --include-log-softmax=false --add-lda=false --add-idct=true \
    --add-final-sigmoid=true \
    --objective-type=xent \
    $dir/configs

fi

if [ $stage -le 4 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
      /export/b{05,06,11,12}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5b/$dir/egs/storage $dir/egs/storage
  fi

  if [ ! -z "$deriv_weights_scp" ]; then
    egs_opts="$egs_opts --deriv-weights-scp $deriv_weights_scp"
  fi

  egs_opts="$egs_opts --num-utts-subset-train $num_utts_subset_train --num-utts-subset-valid $num_utts_subset_valid"

  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --egs.dir="$egs_dir" --egs.stage=$get_egs_stage --egs.opts="$egs_opts" \
    --trainer.num-epochs=$num_epochs \
    --trainer.samples-per-iter=$samples_per_iter \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.max-param-change=$max_param_change \
    ${config_dir:+--configs-dir=$config_dir} \
    --cmd="$decode_cmd" --nj 40 \
    --cleanup=true \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=true \
    --use-dense-targets=$use_dense_targets \
    --egs.extra-copy-cmd="$extra_egs_copy_cmd" \
    --feat-dir=$train_data_dir \
    --targets-scp="$targets_scp" \
    --dir=$dir || exit 1
fi

