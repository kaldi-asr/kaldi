#!/usr/bin/env bash

# Copyright 2015  Johns Hopkins University (Author: Daniel Povey).
#           2015  Vijayaditya Peddinti
#           2015  Xingyu Na
#           2015  Pegah Ghahrmani
# Apache 2.0.


# this is a basic lstm script, it can also be used to train blstm models.
# the blstm can be run using local/nnet3/run_blstm.sh which invokes this script
# with the necessary parameters
# Note: lstm script runs for more epochs than the tdnn script

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call lstm/train.sh with --gpu false

stage=0
train_stage=-10
mic=ihm
use_ihm_ali=false
use_sat_alignments=false # if true, use tri4a alignments are used
                         # by default GMM-HMM systems are not built to this stage
                         # in SDM and MDM systems. So run the tri4a stage if you
                         # want to use this option
affix=
common_egs_dir=

# LSTM options
splice_indexes="-2,-1,0,1,2 0 0"
lstm_delay=" -1 -2 -3 "
label_delay=5
num_lstm_layers=3
cell_dim=1024
hidden_dim=1024
recurrent_projection_dim=256
non_recurrent_projection_dim=256
chunk_width=20
chunk_left_context=40
chunk_right_context=0
shrink=0.99
max_param_change=2.0

# training options
num_epochs=10
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=2
num_jobs_final=12
momentum=0.5
num_chunk_per_minibatch=100
samples_per_iter=20000
remove_egs=true
realign_times=

# feature options
use_ivectors=true

#decode options
extra_left_context=
extra_right_context=
frames_per_chunk=
decode_iter=

# End configuration section.

echo "$0 $@"  # Print the command line for logging

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


# set the variable names
use_delay=false
if [ $label_delay -gt 0 ]; then use_delay=true; fi

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
  ali_dir=${gmm_dir}_${mic}_train_parallel_sp_ali
else
  gmm_dir=exp/$mic/$gmm
  ali_dir=${gmm_dir}_${mic}_train_sp_ali
fi

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$gmm_dir/graph_${LM}
dir=exp/$mic/nnet3/lstm${affix:+_$affix}${use_delay:+_ld$label_delay}




if [ $stage -le 10 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,5,6,7}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi
  if [ "$use_ivectors" == "true" ]; then
    ivector_opts=" --online-ivector-dir exp/$mic/nnet3/ivectors_train_sp_hires "
    cmvn_opts="--norm-means=false --norm-vars=false"
  else
    ivector_opts=
    cmvn_opts="--norm-means=true --norm-vars=true"
  fi

  steps/nnet3/lstm/train.sh $ivector_opts \
    --stage $train_stage \
    --label-delay $label_delay \
    --num-epochs $num_epochs --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
    --num-chunk-per-minibatch $num_chunk_per_minibatch \
    --samples-per-iter $samples_per_iter \
    --splice-indexes "$splice_indexes" \
    --add-lda $add_lda \
    --feat-type raw \
    --cmvn-opts "$cmvn_opts" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --momentum $momentum \
    --max-param-change $max_param_change \
    --lstm-delay "$lstm_delay" \
    --shrink $shrink \
    --cmd "$decode_cmd" \
    --num-lstm-layers $num_lstm_layers \
    --cell-dim $cell_dim \
    --hidden-dim $hidden_dim \
    --recurrent-projection-dim $recurrent_projection_dim \
    --non-recurrent-projection-dim $non_recurrent_projection_dim \
    --chunk-width $chunk_width \
    --chunk-left-context $chunk_left_context \
    --chunk-right-context $chunk_right_context \
    --egs-dir "$common_egs_dir" \
    --remove-egs $remove_egs \
    --realign-times "$realign_times" \
    data/$mic/train_sp_hires data/lang $ali_dir $dir  || exit 1;
fi

if [ $stage -le 11 ]; then
  if [ -z $extra_left_context ]; then
    extra_left_context=$chunk_left_context
  fi
  if [ -z $extra_right_context ]; then
    extra_right_context=$chunk_right_context
  fi
  if [ -z $frames_per_chunk ]; then
    frames_per_chunk=$chunk_width
  fi
  model_opts=
  [ ! -z $decode_iter ] && model_opts=" --iter $decode_iter ";
  for decode_set in dev eval; do
      (
      num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      decode_dir=${dir}/decode_${decode_set}
      if [ "$use_ivectors" == "true" ]; then
        ivector_opts=" --online-ivector-dir exp/$mic/nnet3/ivectors_${decode_set} "
      else
        ivector_opts=
      fi
      steps/nnet3/decode.sh --nj 250 --cmd "$decode_cmd" \
        $ivector_opts $model_opts \
        --extra-left-context $extra_left_context  \
        --extra-right-context $extra_right_context  \
        --frames-per-chunk "$frames_per_chunk" \
        $graph_dir data/$mic/${decode_set}_hires $decode_dir || exit 1;
      ) &
  done
fi
wait;
exit 0;
