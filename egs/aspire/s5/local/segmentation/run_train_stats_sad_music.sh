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
get_egs_stage=-10
egs_opts=

splice_indexes="-3,-2,-1,0,1,2,3,mean(-99:3:9:99) -6,0 -9,0,3 0"
relu_dim=256
chunk_width=20
extra_left_context=100
extra_right_context=20
min_extra_left_context=20
min_extra_right_context=0

# training options
num_epochs=2
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=3
num_jobs_final=8
remove_egs=false
max_param_change=1
extra_egs_copy_cmd=

num_utts_subset_valid=50
num_utts_subset_train=50

# target options
train_data_dir=data/train_azteec_whole_sp_corrupted_hires

snr_scp=
speech_feat_scp=
music_labels_scp=

deriv_weights_scp=
deriv_weights_for_irm_scp=
use_irm_targets=

egs_dir=
nj=40
feat_type=raw
config_dir=

dir=
affix=a

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

num_hidden_layers=`echo $splice_indexes | perl -ane 'print scalar @F'` || exit 1
if [ -z "$dir" ]; then
  dir=exp/nnet3_sad_snr/nnet_tdnn
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

num_snr_bins=`feat-to-dim scp:$snr_scp -`

if [ $stage -le 3 ]; then
  local/segmentation/make_sad_tdnn_configs.py \
    --feat-dir=$train_data_dir \
    --splice-indexes="$splice_indexes" \
    --relu-dim=$relu_dim \
    --add-lda=false \
    ${use_irm_targets:+--output-node-parameters "--output-suffix=snr --dim=$num_snr_bins --add-final-sigmoid=false --include-log-softmax=false --objective-type=quadratic"} \
    --output-node-parameters "--output-suffix=speech --dim=2 --include-log-softmax=true --objective-type=linear" \
    --output-node-parameters "--output-suffix=music --dim=2 --include-log-softmax=true --objective-type=linear" \
    $dir/configs

fi

if [ -z "$egs_dir" ]; then
  egs_dir=$dir/egs
  if [ $stage -le 4 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
      utils/create_split_dir.pl \
        /export/b{03,04,05,06}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
    fi

    . $dir/configs/vars

    steps/nnet3/get_egs_multiple_targets.py --cmd="$decode_cmd" \
      --feat.dir="$train_data_dir" \
      --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
      --frames-per-eg=$chunk_width \
      --left-context=$[model_left_context + extra_left_context] \
      --right-context=$[model_right_context + extra_right_context] \
      --num-utts-subset-train=$num_utts_subset_train \
      --num-utts-subset-valid=$num_utts_subset_valid \
      --samples-per-iter=20000 \
      --stage=$get_egs_stage \
      ${use_irm_targets:+--targets-parameters="--output-name=output-snr --target-type=dense --targets-scp=$snr_scp --deriv-weights-scp=$deriv_weights_for_irm_scp --compress=true"} \
      --targets-parameters="--output-name=output-speech --target-type=sparse --dim=2 --targets-scp=$speech_feat_scp --deriv-weights-scp=$deriv_weights_scp --scp2ark-cmd=\"extract-column --column-index=0 scp:- ark,t:- | steps/segmentation/quantize_vector.pl | ali-to-post ark,t:- ark:- |\" --compress=true" \
      --targets-parameters="--output-name=output-music --target-type=sparse --dim=2 --targets-scp=$music_labels_scp --scp2ark-cmd=\"ali-to-post scp:- ark:- |\" --compress=true" \
      --dir=$dir/egs
  fi
fi

if [ $stage -le 5 ]; then
  steps/nnet3/train_raw_rnn.py --stage=$train_stage \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --egs.chunk-width=20 \
    --egs.dir="$egs_dir" --egs.stage=$get_egs_stage --egs.opts="$egs_opts" \
    --egs.chunk-left-context=$extra_left_context \
    --egs.chunk-right-context=$extra_right_context \
    ${extra_egs_copy_cmd:+--egs.extra-copy-cmd="$extra_egs_copy_cmd"} \
    --trainer.min-chunk-left-context=$min_extra_left_context \
    --trainer.min-chunk-right-context=$min_extra_right_context \
    --trainer.num-epochs=$num_epochs \
    --trainer.samples-per-iter=20000 \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.optimization.shrink-value=1.0 \
    --trainer.rnn.num-chunk-per-minibatch=64 \
    --trainer.rnn.num-bptt-steps=20 \
    --trainer.max-param-change=$max_param_change \
    --cmd="$decode_cmd" --nj 40 \
    --cleanup=true \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=true \
    --use-dense-targets=false \
    --feat-dir=$train_data_dir \
    --targets-scp="$speech_feat_scp" \
    --dir=$dir || exit 1
fi

if [ $stage -le 6 ]; then
  $train_cmd JOB=1:100 $dir/log/compute_post_output-speech.JOB.log \
    extract-column "scp:utils/split_scp.pl -j 100 \$[JOB-1] $speech_feat_scp |" ark,t:- \| \
    steps/segmentation/quantize_vector.pl \| \
    ali-to-post ark,t:- ark:- \| \
    weight-post ark:- scp:$deriv_weights_scp ark:- \| \
    post-to-feats --post-dim=2 ark:- ark:- \| \
    matrix-sum-rows ark:- ark:- \| \
    vector-sum ark:- $dir/post_output-speech.vec.JOB
  eval vector-sum $dir/post_output-speech.vec.{`seq -s, 100`} $dir/post_output-speech.vec
  
  $train_cmd JOB=1:100 $dir/log/compute_post_output-music.JOB.log \
    ali-to-post "scp:utils/split_scp.pl -j 100 \$[JOB-1] $music_labels_scp |" ark:- \| \
    post-to-feats --post-dim=2 ark:- ark:- \| \
    matrix-sum-rows ark:- ark:- \| \
    vector-sum ark:- $dir/post_output-music.vec.JOB
  eval vector-sum $dir/post_output-music.vec.{`seq -s, 100`} $dir/post_output-music.vec
fi
