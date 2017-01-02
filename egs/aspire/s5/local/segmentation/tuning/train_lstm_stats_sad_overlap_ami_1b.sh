#!/bin/bash

# This is a script to train a LSTM for overlapped speech activity detection
# and SAD.

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

chunk_width=20  
num_chunk_per_minibatch=128

extra_left_context=40  # Maximum left context in egs apart from TDNN's left context
extra_right_context=0  # Maximum right context in egs apart from TDNN's right context

# training options
num_epochs=8
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=3
num_jobs_final=8
remove_egs=false
max_param_change=0.2  # Small max-param change for small network
extra_egs_copy_cmd=   # Used if you want to do some weird stuff to egs
                      # such as removing one of the targets

data_dir=data/ami_sdm1_train_whole_hires_bp
labels_scp=exp/sad_ami_sdm1_train/ref/overlapping_sad_labels.scp
deriv_weights_scp=exp/sad_ami_sdm1_train/ref/deriv_weights_for_overlapping_sad.scp

egs_dir=
nj=40
feat_type=raw
config_dir=

dir=
affix=a

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

num_utts=`cat $data_dir/utt2spk | wc -l`
num_utts_subset_valid=`perl -e '$n=int($ARGV[0] * 0.005); $n = ($n > 4000 ? 4000 : $n); print ($n < 6 ? 6 : $n)' $num_utts`
num_utts_subset_train=`perl -e '$n=int($ARGV[0] * 0.005); $n = ($n > 4000 ? 4000 : $n); print ($n < 6 ? 6 : $n)' $num_utts`

if [ -z "$dir" ]; then
  dir=exp/nnet3_ovlp_sad_ami/nnet_lstm
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

if [ $stage -le 0 ]; then
  utils/split_data.sh $data_dir $nj
  $train_cmd JOB=1:$nj $dir/log/compute_post_output-overlapping_sad.JOB.log \
    ali-to-post "scp:utils/filter_scp.pl $data_dir/split$nj/JOB/utt2spk $labels_scp |" ark:- \| \
    weight-post ark:- scp:$deriv_weights_scp ark:- \| \
    post-to-feats --post-dim=3 ark:- ark:- \| \
    matrix-sum-rows ark:- ark:- \| \
    vector-sum ark:- $dir/post_output-overlapping_sad.vec.JOB
  eval vector-sum $dir/post_output-overlapping_sad.vec.{`seq -s, $nj`} $dir/post_output-overlapping_sad.vec
  rm $dir/post_output-overlapping_sad.vec.*
fi

num_frames=`copy-vector --binary=false $dir/post_output-overlapping_sad.vec - | awk '{print $2+$3}'`

copy-vector --binary=false $dir/post_output-overlapping_sad.vec $dir/post_output-overlapping_sad.txt

write_presoftmax_scale() {
  python -c 'import sys
sys.path.insert(0, "steps")
import libs.nnet3.train.common as common_train_lib
import libs.common as common_lib
priors_file = sys.argv[1]
output_file = sys.argv[2]

pdf_counts = common_lib.read_kaldi_matrix(priors_file)[0]
scaled_counts = common_train_lib.smooth_presoftmax_prior_scale_vector(
  pdf_counts, -1.00, smooth=0.0)
common_lib.write_kaldi_matrix(output_file, [scaled_counts])' $1 $2
}

write_presoftmax_scale $dir/post_output-overlapping_sad.txt \
  $dir/presoftmax_prior_scale_output-overlapping_sad.txt

if [ $stage -le 1 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=`feat-to-dim scp:$data_dir/feats.scp -` name=input
  output name=output-temp input=Append(-2,-1,0,1,2)

  relu-renorm-layer name=tdnn1 input=Append(input@-2, input@-1, input, input@1, input@2) dim=256
  relu-renorm-layer name=tdnn2 input=Append(-1,0,1,2) dim=256
  relu-renorm-layer name=tdnn3 input=Append(-3,0,3,6) dim=256
  lstmp-layer name=lstm1 cell-dim=256 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-3
  relu-renorm-layer name=tdnn4 input=Append(-6,0,6) dim=256
  lstmp-layer name=lstm2 cell-dim=256 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-6

  output-layer name=output-overlapping_sad include-log-softmax=true dim=3 input=lstm2 presoftmax-scale-file=$dir/presoftmax_prior_scale_output-overlapping_sad.txt learning-rate-factor=0.05
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig \
    --config-dir $dir/configs/ \
    --nnet-edits="rename-node old-name=output-overlapping_sad new-name=output"

  cat <<EOF >> $dir/configs/vars
add_lda=false
EOF
fi

samples_per_iter=`perl -e "print int(400000 / $chunk_width)"`

if [ -z "$egs_dir" ]; then
  egs_dir=$dir/egs_overlapping_sad
  if [ $stage -le 2 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs_overlapping_sad/storage ]; then
      utils/create_split_dir.pl \
        /export/b{01,02,05,06}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$dir/egs_overlapping_sad/storage $dir/egs_overlapping_sad/storage
    fi

    . $dir/configs/vars

    steps/nnet3/get_egs_multiple_targets.py --cmd="$decode_cmd" \
      $egs_opts \
      --feat.dir="$data_dir" \
      --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
      --frames-per-eg=$chunk_width \
      --left-context=$[model_left_context + extra_left_context] \
      --right-context=$[model_right_context + extra_right_context] \
      --num-utts-subset-train=$num_utts_subset_train \
      --num-utts-subset-valid=$num_utts_subset_valid \
      --samples-per-iter=$samples_per_iter \
      --stage=$get_egs_stage \
      --targets-parameters="--output-name=output-overlapping_sad --target-type=sparse --dim=3 --targets-scp=$labels_scp --deriv-weights-scp=$deriv_weights_scp --scp2ark-cmd=\"ali-to-post scp:- ark: |\" --compress=true" \
      --generate-egs-scp=true \
      --dir=$dir/egs_overlapping_sad
  fi
fi

if [ $stage -le 5 ]; then
  steps/nnet3/train_raw_rnn.py --stage=$train_stage \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --egs.chunk-width=$chunk_width \
    --egs.dir="$egs_dir" --egs.stage=$get_egs_stage \
    --egs.chunk-left-context=$extra_left_context \
    --egs.chunk-right-context=$extra_right_context \
    --egs.use-multitask-egs=false --egs.rename-multitask-outputs=false \
    ${extra_egs_copy_cmd:+--egs.extra-copy-cmd="$extra_egs_copy_cmd"} \
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
    --trainer.compute-per-dim-accuracy=true \
    --cmd="$decode_cmd" --nj 40 \
    --cleanup=true \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=true \
    --use-dense-targets=false \
    --feat-dir=$data_dir \
    --targets-scp="$labels_scp" \
    --dir=$dir || exit 1
fi

