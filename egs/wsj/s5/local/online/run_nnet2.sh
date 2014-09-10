#!/bin/bash

# This is our online neural net build.
# After this you can do discriminative training with run_nnet2_discriminative.sh.

. cmd.sh


stage=1
train_stage=-10
use_gpu=true
. cmd.sh
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
  parallel_opts="-l gpu=1" 
  num_threads=1
  minibatch_size=512
  # the _a is in case I want to change the parameters.
  dir=exp/nnet2_online/nnet_a_gpu 
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
  dir=exp/nnet2_online/nnet_a
fi


if [ $stage -le 1 ]; then
  mkdir -p exp/nnet2_online
  # To train a diagonal UBM we don't need very much data, so use just the si84 data.
  # the tri3b is the input dir; the choice of this is not critical as we just use
  # it for the LDA matrix.  Since the iVectors don't make a great deal of difference,
  # we'll use 256 Gaussians for speed.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 200000 \
    data/train_si84 256 exp/tri3b exp/nnet2_online/diag_ubm
fi

if [ $stage -le 2 ]; then
  # even though $nj is just 10, each job uses multiple processes and threads.
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/train_si284 exp/nnet2_online/diag_ubm exp/nnet2_online/extractor || exit 1;
fi

if [ $stage -le 3 ]; then
  # We extract iVectors on all the train_si284 data, which will be what we
  # train the system on.
  steps/online/nnet2/extract_ivectors_online2.sh --cmd "$train_cmd" --nj 30 \
    data/train_si284 exp/nnet2_online/extractor exp/nnet2_online/ivectors2_train_si284 || exit 1;
fi


if [ $stage -le 4 ]; then
  # Because we have a lot of data here and we don't want the training to take
  # too long so we reduce the number of epochs from the defaults (15 + 5) to (8
  # + 4).

  # Regarding the number of jobs, we decided to let others run their jobs too so
  # we only use 6 GPUs) (we only have 10 GPUs on our queue at JHU).  The number
  # of parameters is a bit smaller than the baseline system we had in mind
  # (../nnet2/run_5d.sh), which had pnorm input/output dim 2000/400 and 4 hidden
  # layers, versus our 2400/300 and 4 hidden layers, even though we're training
  # on more data than the baseline system (we're changing the proportions of
  # input/output dim to be closer to 10:1 which is what we believe works well).
  # The motivation of having fewer parameters is that we want to demonstrate the
  # capability of doing real-time decoding, and if the network was too bug we
  # wouldn't be able to decode in real-time using a CPU.
  #
  # I copied the learning rates from ../nnet2/run_5d.sh
  steps/nnet2/train_pnorm_fast.sh --stage $train_stage \
    --num-epochs 8 --num-epochs-extra 4 \
    --splice-width 7 --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors2_train_si284 \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-jobs-nnet 6 \
    --num-hidden-layers 4 \
    --mix-up 4000 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 2400 \
    --pnorm-output-dim 300 \
    data/train_si284 data/lang exp/tri4b_ali_si284 $dir  || exit 1;
fi

if [ $stage -le 5 ]; then
  for data in test_eval92 test_dev93; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 8 \
      data/${data} exp/nnet2_online/extractor exp/nnet2_online/ivectors_${data} || exit 1;
  done
fi


if [ $stage -le 6 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
  for lm_suffix in tgpr bd_tgpr; do
    graph_dir=exp/tri4b/graph_${lm_suffix}
    # use already-built graphs.
    for year in eval92 dev93; do
      steps/nnet2/decode.sh --nj 8 --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet2_online/ivectors_test_$year \
         $graph_dir data/test_$year $dir/decode_${lm_suffix}_${year} || exit 1;
    done
  done
fi


if [ $stage -le 7 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh data/lang exp/nnet2_online/extractor \
    "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 8 ]; then
  # do the actual online decoding with iVectors, carrying info forward from 
  # previous utterances of the same speaker.
  for lm_suffix in tgpr bd_tgpr; do
    graph_dir=exp/tri4b/graph_${lm_suffix}
    for year in eval92 dev93; do
      steps/online/nnet2/decode.sh --cmd "$decode_cmd" --nj 8 \
        "$graph_dir" data/test_${year} ${dir}_online/decode_${lm_suffix}_${year} || exit 1;
    done
  done
fi

if [ $stage -le 9 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for lm_suffix in tgpr bd_tgpr; do
    graph_dir=exp/tri4b/graph_${lm_suffix}
    for year in eval92 dev93; do
      steps/online/nnet2/decode.sh --cmd "$decode_cmd" --nj 8 \
        --per-utt true \
        "$graph_dir" data/test_${year} ${dir}_online/decode_${lm_suffix}_${year}_utt || exit 1;
    done
  done
fi

if [ $stage -le 10 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.  By setting --online false we
  # let it estimate the iVector from the whole utterance; it's then given to all
  # frames of the utterance.  So it's not really online.
  for lm_suffix in tgpr bd_tgpr; do
    graph_dir=exp/tri4b/graph_${lm_suffix}
    for year in eval92 dev93; do
      steps/online/nnet2/decode.sh --cmd "$decode_cmd" --nj 8 \
        --per-utt true --online false \
        "$graph_dir" data/test_${year} ${dir}_online/decode_${lm_suffix}_${year}_utt_offline || exit 1;
    done
  done
fi



exit 0;

# Here are results.

# first, this is the baseline.  We choose as a baseline our best fMLLR+p-norm system trained
# on si284, so this is a very good baseline.  For others you can see ../RESULTS.


# %WER 7.13 [ 587 / 8234, 72 ins, 93 del, 422 sub ] exp/nnet5d_gpu/decode_bd_tgpr_dev93/wer_13
# %WER 4.06 [ 229 / 5643, 31 ins, 16 del, 182 sub ] exp/nnet5d_gpu/decode_bd_tgpr_eval92/wer_14
# %WER 9.35 [ 770 / 8234, 161 ins, 78 del, 531 sub ] exp/nnet5d_gpu/decode_tgpr_dev93/wer_12
# %WER 6.59 [ 372 / 5643, 91 ins, 15 del, 266 sub ] exp/nnet5d_gpu/decode_tgpr_eval92/wer_12


# Here is the offline decoding of our system (note: it still has the iVectors estimated frame
# by frame, and for each utterance independently).

for x in exp/nnet2_online/nnet_a_gpu/decode_*; do grep WER $x/wer_* | utils/best_wer.sh; done | grep -v utt
%WER 7.53 [ 620 / 8234, 63 ins, 105 del, 452 sub ] exp/nnet2_online/nnet_a_gpu/decode_bd_tgpr_dev93/wer_12
%WER 4.47 [ 252 / 5643, 27 ins, 22 del, 203 sub ] exp/nnet2_online/nnet_a_gpu/decode_bd_tgpr_eval92/wer_13
%WER 9.91 [ 816 / 8234, 164 ins, 90 del, 562 sub ] exp/nnet2_online/nnet_a_gpu/decode_tgpr_dev93/wer_12
%WER 7.12 [ 402 / 5643, 91 ins, 22 del, 289 sub ] exp/nnet2_online/nnet_a_gpu/decode_tgpr_eval92/wer_13

 # Here is the version of the above without iVectors, as done by
 # ./run_nnet2_baseline.sh.  It's about 0.5% absolute worse.
 # There is also an _online version of that decode directory, which is
 # essentially the same (we don't show the results here, as it's not really interesting).
 for x in exp/nnet2_online/nnet_a_gpu_baseline/decode_*; do grep WER $x/wer_* | utils/best_wer.sh; done
 %WER 8.03 [ 661 / 8234, 80 ins, 105 del, 476 sub ] exp/nnet2_online/nnet_a_gpu_baseline/decode_bd_tgpr_dev93/wer_11
 %WER 5.10 [ 288 / 5643, 43 ins, 22 del, 223 sub ] exp/nnet2_online/nnet_a_gpu_baseline/decode_bd_tgpr_eval92/wer_11
 %WER 10.51 [ 865 / 8234, 177 ins, 95 del, 593 sub ] exp/nnet2_online/nnet_a_gpu_baseline/decode_tgpr_dev93/wer_11
 %WER 7.34 [ 414 / 5643, 88 ins, 25 del, 301 sub ] exp/nnet2_online/nnet_a_gpu_baseline/decode_tgpr_eval92/wer_13

# Next, truly-online decoding.
# The results below are not quite as good as those in nnet_a_gpu, but I believe
# the difference is that in this setup we're not using config files, and the
# default beams/lattice-beams in the scripts are slightly different: 15.0/8.0
# above, and 13.0/6.0 below.
for x in exp/nnet2_online/nnet_a_gpu_online/decode_*; do grep WER $x/wer_* | utils/best_wer.sh; done | grep -v utt
%WER 7.53 [ 620 / 8234, 74 ins, 97 del, 449 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_bd_tgpr_dev93/wer_11
%WER 4.45 [ 251 / 5643, 35 ins, 19 del, 197 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_bd_tgpr_eval92/wer_12
%WER 10.02 [ 825 / 8234, 166 ins, 88 del, 571 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_tgpr_dev93/wer_12
%WER 6.91 [ 390 / 5643, 103 ins, 15 del, 272 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_tgpr_eval92/wer_10

# Below is as above, but decoding each utterance separately.  It actualy seems slightly better,
# which is counterintuitive.
for x in exp/nnet2_online/nnet_a_gpu_online/decode_*; do grep WER $x/wer_* | utils/best_wer.sh; done | grep  utt
%WER 7.55 [ 622 / 8234, 57 ins, 109 del, 456 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_bd_tgpr_dev93_utt/wer_13
%WER 4.43 [ 250 / 5643, 27 ins, 21 del, 202 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_bd_tgpr_eval92_utt/wer_13
%WER 9.98 [ 822 / 8234, 179 ins, 80 del, 563 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_tgpr_dev93_utt/wer_11
%WER 7.12 [ 402 / 5643, 98 ins, 18 del, 286 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_tgpr_eval92_utt/wer_12
