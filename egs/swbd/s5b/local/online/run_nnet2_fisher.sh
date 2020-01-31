#!/usr/bin/env bash


# This script trains a Switchboard system starting from a neural net trained for
# Fisher English.  It builds a
# neural net for online decoding on top of the network we previously trained on
# WSJ, by keeping everything but the last layer of that network and then
# training just the last layer on our data.

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
  dir=exp/nnet2_online_wsj/nnet_gpu
  trainfeats=exp/nnet2_online_wsj/wsj_activations_train_gpu
  srcdir=../../wsj/s5/exp/nnet2_online/nnet_a_gpu_online
  # the following things are needed while training the combined model.
  srcdir_orig=../../wsj/s5/exp/nnet2_online/nnet_a_gpu
  ivector_src=../../wsj/s5/exp/nnet2_online/extractor
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="--num-threads $num_threads"
  dir=exp/nnet2_online_wsj/nnet
  trainfeats=exp/nnet2_online_wsj/wsj_activations_train
  srcdir=../../wsj/s5/exp/nnet2_online/nnet_a_online
  # the following things are needed while training the combined model.
  srcdir_orig=../../wsj/s5/exp/nnet2_online/nnet_a
  ivector_src=../../wsj/s5/exp/nnet2_online/extractor
fi


if [ $stage -le 0 ]; then
  echo "$0: dumping activations from WSJ model"
  steps/online/nnet2/dump_nnet_activations.sh --cmd "$train_cmd" --nj 30 \
     data/train $srcdir $trainfeats
fi

if [ $stage -le 1 ]; then
  echo "$0: training 0-hidden-layer model on top of WSJ activations"
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

# Note: at this point it might be possible to further train the combined model
# by doing backprop through all of it.  We haven't implemented this yet.

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
  steps/nnet2/create_appended_model.sh $srcdir_orig $dir ${dir}_combined_init

  # Set the learning rate in this initial value to our guess of a suitable value.
  # note: we initially tried 0.005, and this gave us WERs of (1.40, 1.48, 7.24, 7.70) vs.
  # (1.32, 1.38, 7.20, 7.44) with a learning rate of 0.01.
  initial_learning_rate=0.01
  nnet-am-copy --learning-rate=$initial_learning_rate ${dir}_combined_init/final.mdl ${dir}_combined_init/final.mdl
fi

# In order to train the combined model, we'll need to dump iVectors.
if [ $stage -le 6 ]; then
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/train data/train_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 10 \
    data/train_max2 $ivector_src exp/nnet2_online_wsj/ivectors || exit 1;
fi

if [ $stage -le 7 ]; then
  # assume left and right context of model are identical.
  splice_width=$(nnet-am-info exp/nnet2_online_wsj/nnet_gpu_combined_init/final.mdl | grep '^left-context' | awk '{print $2}') || exit 1;

  # Note: in general the get_egs.sh script would get things like the LDA matrix
  # from exp/tri3b_ali, which would be the wrong thing to do as we want to get
  # them from the original model dir.  In this case we're using raw MFCC
  # features so it's not an issue.  But in general we'd probably have to create
  # a temporary dir and copy or link both the alignments and feature-related
  # things to it.
  steps/nnet2/get_egs.sh  --cmd "$train_cmd" \
    --feat-type raw --cmvn-opts "--norm-means=false --norm-vars=false" \
    --online-ivector-dir exp/nnet2_online_wsj/ivectors \
    --num-jobs-nnet 4 --splice-width $splice_width \
    data/train data/lang exp/tri3b_ali ${dir}_combined
fi

if [ $stage -le 8 ]; then
  steps/nnet2/train_more.sh --learning-rate-factor 0.1 --cmd "$train_cmd" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
     ${dir}_combined_init/final.mdl  ${dir}_combined/egs ${dir}_combined
fi

if [ $stage -le 9 ]; then
  # Create an online-decoding dir corresponding to what we just trained above.
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh data/lang $ivector_src \
    ${dir}_combined ${dir}_combined_online || exit 1;
fi

if [ $stage -le 10 ]; then
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
# grep WER exp/nnet2_online_wsj/nnet_gpu_online/decode/wer_* | utils/best_wer.sh
#%WER 1.61 [ 202 / 12533, 22 ins, 46 del, 134 sub ] exp/nnet2_online_wsj/nnet_gpu_online/decode/wer_3
#a11:s5: grep WER exp/nnet2_online_wsj/nnet_gpu_online/decode_ug/wer_* | utils/best_wer.sh
#%WER 7.99 [ 1002 / 12533, 74 ins, 153 del, 775 sub ] exp/nnet2_online_wsj/nnet_gpu_online/decode_ug/wer_6

# and with per-utterance decoding:
# %WER 1.72 [ 216 / 12533, 26 ins, 45 del, 145 sub ] exp/nnet2_online_wsj/nnet_gpu_online/decode_utt/wer_3
# %WER 8.40 [ 1053 / 12533, 85 ins, 158 del, 810 sub ] exp/nnet2_online_wsj/nnet_gpu_online/decode_ug_utt/wer_6

#, here when we retrain the whole thing:
# %WER 1.32 [ 165 / 12533, 14 ins, 34 del, 117 sub ] exp/nnet2_online_wsj/nnet_gpu_combined_online/decode/wer_3
# %WER 7.20 [ 902 / 12533, 78 ins, 127 del, 697 sub ] exp/nnet2_online_wsj/nnet_gpu_combined_online/decode_ug/wer_6

# and with per-utterance decoding:
# %WER 1.38 [ 173 / 12533, 19 ins, 32 del, 122 sub ] exp/nnet2_online_wsj/nnet_gpu_combined_online/decode_per_utt/wer_3
# %WER 7.44 [ 932 / 12533, 57 ins, 163 del, 712 sub ] exp/nnet2_online_wsj/nnet_gpu_combined_online/decode_ug_per_utt/wer_8

# And this is a suitable baseline: a system trained on RM only.
#a11:s5: grep WER exp/nnet2_online/nnet_gpu_online/decode/wer_* | utils/best_wer.sh
#%WER 2.20 [ 276 / 12533, 25 ins, 69 del, 182 sub ] exp/nnet2_online/nnet_gpu_online/decode/wer_8
#a11:s5: grep WER exp/nnet2_online/nnet_gpu_online/decode_ug/wer_* | utils/best_wer.sh
#%WER 10.14 [ 1271 / 12533, 127 ins, 198 del, 946 sub ] exp/nnet2_online/nnet_gpu_online/decode_ug/wer_11
