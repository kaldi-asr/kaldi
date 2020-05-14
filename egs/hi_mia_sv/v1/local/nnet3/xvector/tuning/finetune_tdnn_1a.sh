#!/bin/bash
# This script uses weight transfer as a transfer learning method to transfer
# already trained neural net model on aishell2 to a finetune data set.

set -e
. ./path.sh
. ./cmd.sh

data_set=himia/train
data_dir=data/$data_set
src_dir=exp/xvector_nnet_1a
dir=${src_dir}_finetune

momentum=0.5
num_jobs_initial=1
num_jobs_final=1
num_epochs=5
initial_effective_lrate=0.0005
final_effective_lrate=0.00002
minibatch_size=128
frames_per_eg=1

stage=1
train_stage=-10
nj=4


if [ $stage -le 1 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    $data_dir ${data_dir}_no_sil exp/${data_set}_no_sil
  utils/fix_data_dir.sh ${data_dir}_no_sil || exit 1;
fi

if [ $stage -le 2 ]; then
  mkdir -p $dir
  $train_cmd $dir/log/generate_input_model.log \
    nnet3-copy --raw=true $src_dir/final.raw $dir/input.raw
fi

dropout_schedule='0,0@0.20,0.1@0.50,0'
srand=123
egs_dir=exp/${data_set}_no_sil
if [ $stage -le 3 ]; then
  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --trainer.optimization.proportional-shrink 10 \
    --trainer.optimization.momentum=$momentum \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.optimization.minibatch-size=$minibatch_size \
    --trainer.input-model $dir/input.raw \
    --trainer.srand=$srand \
    --trainer.max-param-change=2 \
    --trainer.num-epochs=$num_epochs \
    --trainer.dropout-schedule="$dropout_schedule" \
    --trainer.shuffle-buffer-size=1000 \
    --egs.frames-per-eg=$frames_per_eg \
    --egs.dir="$egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=true \
    --dir=$dir  || exit 1;
fi
