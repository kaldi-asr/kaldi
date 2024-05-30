#!/usr/bin/env bash

# note: see the newer, better script run_nnet2_wsj_joint.sh

# This script assumes you have previously run the WSJ example script including
# the optional part local/online/run_online_decoding_nnet2.sh.  It builds a
# neural net for online decoding on top of the network we previously trained on
# WSJ, by keeping everything but the last layer of that network and then
# training just the last layer on our data.  We then train the whole thing.

stage=0
set -e

train_stage=-10
use_gpu=true
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=512
  dir=exp/nnet2_online_wsj/nnet_a
  trainfeats=exp/nnet2_online_wsj/wsj_activations_train
  # later we'll change the script to download the trained model from kaldi-asr.org.
  srcdir=../../wsj/s5/exp/nnet2_online/nnet_a_gpu_online
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="--num-threads $num_threads"
  dir=exp/nnet2_online_wsj/nnet_a
  trainfeats=exp/nnet2_online_wsj/wsj_activations_train
  srcdir=../../wsj/s5/exp/nnet2_online/nnet_a_online
fi


if [ $stage -le 0 ]; then
  echo "$0: dumping activations from WSJ model"
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $trainfeats/feats/storage ]; then
    # this shows how you can split the data across multiple file-systems; it's optional.
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/rm-$date/s5/$trainfeats/feats/storage \
       $trainfeats/feats/storage
  fi
  steps/online/nnet2/dump_nnet_activations.sh --cmd "$train_cmd" --nj 30 \
     data/train $srcdir $trainfeats
fi

if [ $stage -le 1 ]; then
  echo "$0: training 0-hidden-layer model on top of WSJ activations"
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/rm-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet2/retrain_fast.sh --stage $train_stage \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --cmd "$decode_cmd" \
    --num-jobs-nnet 4 \
    --mix-up 4000 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
     $trainfeats/data data/lang exp/tri3b_ali $dir
fi

if [ $stage -le 2 ]; then
  echo "$0: formatting combined model for online decoding."
  steps/online/nnet2/prepare_online_decoding_retrain.sh $srcdir $dir ${dir}_online
fi

if [ $stage -le 3 ]; then
  # do online decoding with the combined model.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph data/test ${dir}_online/decode &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph_ug data/test ${dir}_online/decode_ug || exit 1;
  wait
fi

if [ $stage -le 4 ]; then
  # do online per-utterance decoding with the combined model.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
     --per-utt true \
    exp/tri3b/graph data/test ${dir}_online/decode_utt &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
     --per-utt true \
    exp/tri3b/graph_ug data/test ${dir}_online/decode_ug_utt || exit 1;
  wait
fi

## From this point on we try something else: we try training all the layers of
## the model on this dataset.  First we need to create a combined version of the
## model.
if [ $stage -le 5 ]; then
  steps/nnet2/create_appended_model.sh $srcdir $dir ${dir}_combined_init

  # Set the learning rate in this initial value to our guess of a suitable value.
  # note: we initially tried 0.005, and this gave us WERs of (1.40, 1.48, 7.24, 7.70) vs.
  # (1.32, 1.38, 7.20, 7.44) with a learning rate of 0.01.
  initial_learning_rate=0.01
  nnet-am-copy --learning-rate=$initial_learning_rate ${dir}_combined_init/final.mdl ${dir}_combined_init/final.mdl
fi

if [ $stage -le 6 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
      /export/b0{1,2,3,4}/$USER/kaldi-data/rm-$(date +'%m_%d_%H_%M')/s5/${dir}_combined/egs/storage \
        $dir_combined/egs/storage
  fi

  # This version of the get_egs.sh script does the feature extraction and iVector
  # extraction in a single binary, reading the config, as part of the script.
  steps/online/nnet2/get_egs.sh --cmd "$train_cmd" --num-jobs-nnet 4 \
    data/train exp/tri3b_ali ${dir}_online ${dir}_combined
fi

if [ $stage -le 7 ]; then
  steps/nnet2/train_more.sh --learning-rate-factor 0.1 --cmd "$train_cmd" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
     ${dir}_combined_init/final.mdl  ${dir}_combined/egs ${dir}_combined
fi

if [ $stage -le 8 ]; then
  # Create an online-decoding dir corresponding to what we just trained above.
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh data/lang $srcdir/ivector_extractor \
    ${dir}_combined ${dir}_combined_online || exit 1;
fi

if [ $stage -le 9 ]; then
  # do the online decoding on top of the retrained _combined_online model, and
  # also the per-utterance version of the online decoding.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph data/test ${dir}_combined_online/decode &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph_ug data/test ${dir}_combined_online/decode_ug &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    --per-utt true exp/tri3b/graph data/test ${dir}_combined_online/decode_per_utt &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    --per-utt true exp/tri3b/graph_ug data/test ${dir}_combined_online/decode_ug_per_utt || exit 1;
  wait
fi



exit 0;

# Here are the results when we just retrain the last layer:
# grep WER exp/nnet2_online_wsj/nnet_a_online/decode/wer_* | utils/best_wer.sh
#%WER 1.60 [ 201 / 12533, 22 ins, 46 del, 133 sub ] exp/nnet2_online_wsj/nnet_a_online/decode/wer_3
#a11:s5: grep WER exp/nnet2_online_wsj/nnet_a_online/decode_ug/wer_* | utils/best_wer.sh
#%WER 8.02 [ 1005 / 12533, 74 ins, 155 del, 776 sub ] exp/nnet2_online_wsj/nnet_a_online/decode_ug/wer_6

# and with per-utterance decoding:
# %WER 8.47 [ 1061 / 12533, 88 ins, 157 del, 816 sub ] exp/nnet2_online_wsj/nnet_a_online/decode_ug_utt/wer_6
# %WER 1.70 [ 213 / 12533, 24 ins, 46 del, 143 sub ] exp/nnet2_online_wsj/nnet_a_online/decode_utt/wer_3



#, here when we retrain the whole thing:
#%WER 1.42 [ 178 / 12533, 16 ins, 44 del, 118 sub ] exp/nnet2_online_wsj/nnet_a_combined_online/decode/wer_4
#%WER 7.08 [ 887 / 12533, 74 ins, 133 del, 680 sub ] exp/nnet2_online_wsj/nnet_a_combined_online/decode_ug/wer_6

# and the same with per-utterance decoding:
# %WER 1.56 [ 196 / 12533, 31 ins, 26 del, 139 sub ] exp/nnet2_online_wsj/nnet_a_combined_online/decode_per_utt/wer_2
# %WER 7.86 [ 985 / 12533, 59 ins, 171 del, 755 sub ] exp/nnet2_online_wsj/nnet_a_combined_online/decode_ug_per_utt/wer_8

# And this is a suitable baseline: a system trained on RM only.
#a11:s5: grep WER exp/nnet2_online/nnet_a_online/decode/wer_* | utils/best_wer.sh
#%WER 2.20 [ 276 / 12533, 25 ins, 69 del, 182 sub ] exp/nnet2_online/nnet_a_online/decode/wer_8
#a11:s5: grep WER exp/nnet2_online/nnet_a_online/decode_ug/wer_* | utils/best_wer.sh
#%WER 10.14 [ 1271 / 12533, 127 ins, 198 del, 946 sub ] exp/nnet2_online/nnet_a_online/decode_ug/wer_11
