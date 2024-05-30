#!/usr/bin/env bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.
set -e

stage=0
train_stage=-10
has_fisher=true
mic=ihm
use_sat_alignments=true
affix=
speed_perturb=true
common_egs_dir=
splice_indexes="-2,-1,0,1,2 -1,2 -3,3 -7,2 -3,3 0 0"
subset_dim=0
remove_egs=true
relu_dim=850
num_epochs=3

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

local/nnet3/run_ivector_common.sh --stage $stage \
                                  --mic $mic \
                                  --use-ihm-ali $use_ihm_ali \
                                  --use-sat-alignments $use_sat_alignments || exit 1;

# we still support this option as all the TDNN, LSTM, BLSTM systems were built
# using tri3a alignments
if [ $use_sat_alignments == "true" ]; then
  gmm=tri4a
else
  gmm=tri3a
fi

if [ $use_ihm_ali == "true" ]; then
  gmm_dir=exp/ihm/$gmm
  mic=${mic}_cleanali
  ali_dir=${gmm_dir}_train_parallel_sp_ali
else
  gmm_dir=exp/$mic/$gmm
  ali_dir=${gmm_dir}_train_sp_ali
fi

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$gmm_dir/graph_${LM}
dir=exp/$mic/nnet3/tdnn${speed_perturb:+_sp}${affix:+_$affix}



if [ $stage -le 10 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/tdnn/train.sh --stage $train_stage \
    --num-epochs $num_epochs --num-jobs-initial 2 --num-jobs-final 12 \
    --splice-indexes "$splice_indexes" \
    --subset-dim "$subset_dim" \
    --feat-type raw \
    --online-ivector-dir exp/$mic/nnet3/ivectors_${train_set}_hires \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --egs-dir "$common_egs_dir" \
    --initial-effective-lrate 0.0015 --final-effective-lrate 0.00015 \
    --cmd "$decode_cmd" \
    --relu-dim "$relu_dim" \
    --remove-egs "$remove_egs" \
    data/$mic/train_sp_hires data/lang $ali_dir $dir  || exit 1;
fi

if [ $stage -le 11 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for decode_set in dev eval; do
      (
      num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      decode_dir=${dir}/decode_${decode_set}

      steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
          --online-ivector-dir exp/$mic/nnet3/ivectors_${decode_set} \
         $graph_dir data/$mic/${decode_set}_hires $decode_dir || exit 1;
      ) &
  done
fi
wait;
exit 0;
