#!/bin/bash

# This is a script to train a time-delay neural network for speech activity detection (SAD) and
# music-id using statistic pooling component for long-context information.
# This script is same as 1e, but removes the stats component.

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
egs_opts=   # Directly passed to get_egs_multiple_targets.py

# TDNN options
relu_dim=256
chunk_width=20  # We use chunk training for training TDNN
num_chunk_per_minibatch=64

extra_left_context=79 # Maximum left context in egs apart from TDNN's left context
extra_right_context=11  # Maximum right context in egs apart from TDNN's right context

# We randomly select an extra {left,right} context for each job between
# min_extra_*_context and extra_*_context so that the network can get used
# to different contexts used to compute statistics.
min_extra_left_context=20
min_extra_right_context=0

# training options
num_epochs=2
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=3
num_jobs_final=8
remove_egs=false
max_param_change=0.2  # Small max-param change for small network
extra_egs_copy_cmd=   # Used if you want to do some weird stuff to egs
                      # such as removing one of the targets

num_utts_subset_valid=50    # "utts" is actually recording. So this is prettly small.
num_utts_subset_train=50

# target options
train_data_dir=data/train_aztec_small_unsad_whole_all_corrupted_sp_hires_bp

speech_feat_scp=data/train_aztec_unsad_whole_corrupted_sp_hires_bp_2400/speech_feat.scp
deriv_weights_scp=data/train_aztec_unsad_whole_corrupted_sp_hires_bp_2400/deriv_weights.scp
music_labels_scp=data/train_aztec_small_unsad_whole_music_corrupted_sp_hires_bp/music_labels.scp

extra_egs_copy_cmd="nnet3-copy-egs --keep-outputs=output-speech,output-music ark:- ark:- |"

egs_dir=
nj=40
feat_type=raw
config_dir=

dir=
affix=a

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ -z "$dir" ]; then
  dir=exp/nnet3_stats_sad_music/nnet_tdnn
fi

dir=$dir${affix:+_$affix}

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

mkdir -p $dir

if [ $stage -le 1 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=`feat-to-dim scp:$train_data_dir/feats.scp -` name=input
  output name=output-temp input=Append(-3,-2,-1,0,1,2,3)

  relu-renorm-layer name=tdnn1 input=Append(input@-3, input@-2, input@-1, input, input@1, input@2, input@3) dim=256 add-log-stddev=true
  relu-renorm-layer name=tdnn2 input=Append(tdnn1@-9, tdnn1@-3, tdnn1, tdnn1@3) dim=256 add-log-stddev=true
  relu-renorm-layer name=tdnn3 input=Append(tdnn2@-27, tdnn2@-9, tdnn2, tdnn2@9) dim=256 add-log-stddev=true

  output-layer name=output-speech include-log-softmax=true dim=2 input=tdnn3
  output-layer name=output-music include-log-softmax=true dim=2 input=tdnn3
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig \
    --config-dir $dir/configs/ \
    --nnet-edits="rename-node old-name=output-speech new-name=output"

  cat <<EOF >> $dir/configs/vars
add_lda=false
EOF
fi

samples_per_iter=`perl -e "print int(400000 / $chunk_width)"`
speech_data_dir=$dir/`basename $train_data_dir`_speech
music_data_dir=$dir/`basename $train_data_dir`_music

if [ -z "$egs_dir" ]; then
  egs_dir=$dir/egs_multi
  if [ $stage -le 2 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs_speech/storage ]; then
      utils/create_split_dir.pl \
        /export/b{03,04,05,06}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$dir/egs_speech/storage $dir/egs_speech/storage
    fi


    . $dir/configs/vars

    utils/subset_data_dir.sh --utt-list $speech_feat_scp ${train_data_dir} $dir/`basename ${train_data_dir}`_speech

    steps/nnet3/get_egs_multiple_targets.py --cmd="$decode_cmd" \
      $egs_opts \
      --feat.dir="$speech_data_dir" \
      --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
      --frames-per-eg=$chunk_width \
      --left-context=$[model_left_context + extra_left_context] \
      --right-context=$[model_right_context + extra_right_context] \
      --num-utts-subset-train=$num_utts_subset_train \
      --num-utts-subset-valid=$num_utts_subset_valid \
      --samples-per-iter=$samples_per_iter \
      --stage=$get_egs_stage \
      --targets-parameters="--output-name=output-speech --target-type=sparse --dim=2 --targets-scp=$speech_feat_scp --deriv-weights-scp=$deriv_weights_scp --scp2ark-cmd=\"extract-column --column-index=0 scp:- ark,t:- | steps/segmentation/quantize_vector.pl | ali-to-post ark,t:- ark:- |\" --compress=true" \
      --generate-egs-scp=true \
      --dir=$dir/egs_speech
  fi

  if [ $stage -le 3 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs_music/storage ]; then
      utils/create_split_dir.pl \
        /export/b{03,04,05,06}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$dir/egs_music/storage $dir/egs_music/storage
    fi

    . $dir/configs/vars
    utils/subset_data_dir.sh --utt-list $music_labels_scp ${train_data_dir} $dir/`basename ${train_data_dir}`_music

    steps/nnet3/get_egs_multiple_targets.py --cmd="$decode_cmd" \
      $egs_opts \
      --feat.dir="$music_data_dir" \
      --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
      --frames-per-eg=$chunk_width \
      --left-context=$[model_left_context + extra_left_context] \
      --right-context=$[model_right_context + extra_right_context] \
      --num-utts-subset-train=$num_utts_subset_train \
      --num-utts-subset-valid=$num_utts_subset_valid \
      --samples-per-iter=$samples_per_iter \
      --stage=$get_egs_stage \
      --targets-parameters="--output-name=output-music --target-type=sparse --dim=2 --targets-scp=$music_labels_scp --scp2ark-cmd=\"ali-to-post scp:- ark:- |\" --compress=true" \
      --generate-egs-scp=true \
      --dir=$dir/egs_music
  fi

  if [ $stage -le 4 ]; then
    steps/nnet3/multilingual/get_egs.sh \
      --minibatch-size $[chunk_width * num_chunk_per_minibatch] \
      --samples-per-iter $samples_per_iter \
      2 $dir/egs_speech $dir/egs_music $dir/egs_multi
  fi
fi

if [ $stage -le 5 ]; then
  steps/nnet3/train_raw_rnn.py --stage=$train_stage \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --egs.chunk-width=$chunk_width \
    --egs.dir="$egs_dir" --egs.stage=$get_egs_stage \
    --egs.chunk-left-context=$extra_left_context \
    --egs.chunk-right-context=$extra_right_context \
    --egs.use-multitask-egs=true --egs.rename-multitask-outputs=false \
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
    --trainer.rnn.num-chunk-per-minibatch=$num_chunk_per_minibatch \
    --trainer.deriv-truncate-margin=8 \
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

