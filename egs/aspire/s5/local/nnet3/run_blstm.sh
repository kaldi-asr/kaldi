#!/usr/bin/env bash

# based on egs/fisher_swbd/s5/local/nnet3/run_lstm.sh

stage=7
train_stage=-10
egs_stage=
affix=
common_egs_dir=
reporting_email=

# LSTM options
label_delay=0
cell_dim=1024
hidden_dim=1024
recurrent_projection_dim=128
non_recurrent_projection_dim=128
chunk_width=20
chunk_left_context=40
chunk_right_context=40


# training options
num_epochs=6
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=4
num_jobs_final=22
momentum=0.5
num_chunk_per_minibatch=100
samples_per_iter=20000
remove_egs=true

#decode options
extra_left_context=50
extra_right_context=50

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

if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  cmd_opts=" --config conf/queue_only_k80.conf --only-k80 false"
fi


dir=exp/nnet3/lstm_bidirectional
dir=$dir${affix:+_$affix}
if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi
ali_dir=exp/tri5a_rvb_ali

local/nnet3/run_ivector_common.sh --stage $stage || exit 1;

if [ $stage -le 7 ]; then
  num_targets=$(tree-info $ali_dir/tree | grep num-pdfs | awk '{print $2}')
  [ -z $num_targets ] && { echo "$0: error getting num-targets"; exit 1; }

  lstm_opts="decay-time=20 cell-dim=$cell_dim"
  lstm_opts+=" recurrent-projection-dim=$recurrent_projection_dim"
  lstm_opts+=" non-recurrent-projection-dim=$non_recurrent_projection_dim"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda delay=$label_delay input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmp-layer name=blstm1-forward input=lda delay=-1 $lstm_opts
  fast-lstmp-layer name=blstm1-backward input=lda delay=1 $lstm_opts

  fast-lstmp-layer name=blstm2-forward input=Append(blstm1-forward, blstm1-backward) delay=-2 $lstm_opts
  fast-lstmp-layer name=blstm2-backward input=Append(blstm1-forward, blstm1-backward) delay=2 $lstm_opts

  fast-lstmp-layer name=blstm3-forward input=Append(blstm2-forward, blstm2-backward) delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm3-backward input=Append(blstm2-forward, blstm2-backward delay=3 $lstm_opts

  output-layer name=output output-delay=$label_delay dim=$num_targets max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs || exit 1
fi

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_rnn.py --stage=$train_stage \
    --cmd="$decode_cmd $cmd_opts" \
    --feat.online-ivector-dir=exp/nnet3/ivectors_train \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs=$num_epochs \
    --trainer.samples-per-iter=$samples_per_iter \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.rnn.num-chunk-per-minibatch=$num_chunk_per_minibatch \
    --trainer.optimization.momentum=$momentum \
    --egs.opts " --nj 12 " \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
    --egs.dir="$common_egs_dir" \
    --egs.stage "$egs_stage" \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=100 \
    --use-gpu=true \
    --feat-dir=data/train_rvb_hires \
    --ali-dir=$ali_dir \
    --lang=data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi

#ASpIRE decodes
if [ $stage -le 14 ]; then
  local/nnet3/prep_test_aspire.sh --stage 1 --decode-num-jobs 30 --affix "v7" \
   --extra-left-context 40 --extra-right-context 40 --frames-per-chunk 20 \
   --extra-left-context-initial 0 \
   --extra-right-context-final 0 \
   --sub-speaker-frames 6000 --window 10 --overlap 5 --max-count 75 --pass2-decode-opts "--min-active 1000" \
   --ivector-scale 0.75  dev_aspire data/lang exp/tri5a/graph_pp exp/nnet3/lstm_bidirectional
fi
exit 0;

# final result
# %WER 29.4 | 2120 27210 | 77.0 14.7 8.3 6.4 29.4 77.9 | -1.227 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iterfinal_pp_fg/score_13/penalty_0.25/ctm.filt.filt.sys

#%WER 35.4 | 2120 27216 | 71.2 19.2 9.6 6.6 35.4 80.8 | -0.956 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter200_pp_fg/score_14/penalty_0.0/ctm.filt.filt.sys
#%WER 33.6 | 2120 27215 | 72.7 18.1 9.2 6.3 33.6 79.1 | -1.018 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter300_pp_fg/score_12/penalty_0.0/ctm.filt.filt.sys
#%WER 33.0 | 2120 27215 | 73.3 17.4 9.3 6.3 33.0 80.6 | -1.127 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter400_pp_fg/score_12/penalty_0.25/ctm.filt.filt.sys
#%WER 31.6 | 2120 27216 | 74.5 16.4 9.1 6.1 31.6 79.4 | -1.119 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter700_pp_fg/score_13/penalty_0.25/ctm.filt.filt.sys
#%WER 31.8 | 2120 27220 | 74.9 16.3 8.8 6.7 31.8 80.1 | -1.233 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter800_pp_fg/score_12/penalty_0.25/ctm.filt.filt.sys
#%WER 31.6 | 2120 27222 | 75.0 16.1 8.8 6.7 31.6 80.7 | -1.208 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter900_pp_fg/score_12/penalty_0.5/ctm.filt.filt.sys
#%WER 30.0 | 2120 27212 | 76.0 15.4 8.6 6.1 30.0 79.4 | -1.193 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter1700_pp_fg/score_13/penalty_0.25/ctm.filt.filt.sys
#%WER 30.2 | 2120 27211 | 76.4 15.3 8.3 6.7 30.2 79.4 | -1.099 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter1800_pp_fg/score_14/penalty_0.0/ctm.filt.filt.sys
#%WER 30.3 | 2120 27215 | 76.8 15.5 7.8 7.1 30.3 78.8 | -1.317 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter1900_pp_fg/score_11/penalty_0.25/ctm.filt.filt.sys
#%WER 29.8 | 2120 27215 | 76.8 15.0 8.2 6.6 29.8 78.6 | -1.219 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter1996_pp_fg/score_13/penalty_0.25/ctm.filt.filt.sys
#%WER 30.1 | 2120 27213 | 76.3 14.7 9.0 6.4 30.1 79.2 | -1.204 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter2124_pp_fg/score_14/penalty_0.5/ctm.filt.filt.sys
