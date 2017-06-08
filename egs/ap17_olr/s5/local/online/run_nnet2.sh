#!/bin/bash

# this is our online-nnet2 build.  it's a "multi-splice" system (i.e. we have
# splicing at various layers), with p-norm nonlinearities.  We use the "accel2"
# script which uses between 2 and 14 GPUs depending how far through training it
# is.  You can safely reduce the --num-jobs-final to however many GPUs you have
# on your system.

# For joint training with RM, this script is run using the following command line,
# and note that the --stage 8 option is only needed in case you already ran the
# earlier stages.
# local/online/run_nnet2.sh --stage 8 --dir exp/nnet2_online/nnet_ms_a_partial --exit-train-stage 15

. cmd.sh


stage=0
train_stage=-10
use_gpu=true
dir=exp/nnet2_online/nnet_ms_a
exit_train_stage=-100
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
  parallel_opts="--gpu 1" 
  num_threads=1
  minibatch_size=512
  # the _a is in case I want to change the parameters.
else
  num_threads=16
  minibatch_size=128
  parallel_opts="--num-threads $num_threads" 
fi

local/online/run_nnet2_common.sh --stage $stage || exit 1;

if [ $stage -le 8 ]; then
  # last splicing was instead: layer3/-4:2" 
  steps/nnet2/train_multisplice_accel2.sh --stage $train_stage \
    --exit-stage $exit_train_stage \
    --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 14 \
    --num-hidden-layers 4 \
    --splice-indexes "layer0/-1:0:1 layer1/-2:1 layer2/-4:2" \
    --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors_train_si284 \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "--max-jobs-run 12" \
    --initial-effective-lrate 0.005 --final-effective-lrate 0.0005 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 2000 \
    --pnorm-output-dim 250 \
    --mix-up 12000 \
    data/train_si284_hires data/lang exp/tri4b_ali_si284 $dir  || exit 1;
fi

if [ $stage -le 9 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  iter_opt=
  [ $exit_train_stage -gt 0 ] && iter_opt="--iter $exit_train_stage"
  steps/online/nnet2/prepare_online_decoding.sh $iter_opt --mfcc-config conf/mfcc_hires.conf \
    data/lang exp/nnet2_online/extractor "$dir" ${dir}_online || exit 1;
fi

if [ $exit_train_stage -gt 0 ]; then
  echo "$0: not testing since you only ran partial training (presumably in preparation"
  echo " for multilingual training"
  exit 0;
fi

if [ $stage -le 10 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
  for lm_suffix in tgpr bd_tgpr; do
    graph_dir=exp/tri4b/graph_${lm_suffix}
    # use already-built graphs.
    for year in eval92 dev93; do
      steps/nnet2/decode.sh --nj 8 --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet2_online/ivectors_test_$year \
         $graph_dir data/test_${year}_hires $dir/decode_${lm_suffix}_${year} || exit 1;
    done
  done
fi

if [ $stage -le 11 ]; then
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

if [ $stage -le 12 ]; then
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

if [ $stage -le 13 ]; then
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

if [ $stage -le 14 ]; then
  # this does offline decoding, as stage 10, except we estimate the iVectors per
  # speaker, excluding silence (based on alignments from a GMM decoding), with a
  # different script.  This is just to demonstrate that script.
  # the --sub-speaker-frames is optional; if provided, it will divide each speaker
  # up into "sub-speakers" of at least that many frames... can be useful if
  # acoustic conditions drift over time within the speaker's data.
  rm exp/nnet2_online/.error 2>/dev/null
  for year in eval92 dev93; do
    steps/online/nnet2/extract_ivectors.sh --cmd "$train_cmd" --nj 8 \
      --sub-speaker-frames 1500 \
      data/test_${year}_hires data/lang exp/nnet2_online/extractor \
      exp/tri4b/decode_tgpr_$year exp/nnet2_online/ivectors_spk_test_${year} || touch exp/nnet2_online/.error &
  done
  wait
  [ -f exp/nnet2_online/.error ] && echo "$0: Error getting iVectors" && exit 1;

  for lm_suffix in bd_tgpr; do # just use the bd decoding, to avoid wasting time.
    graph_dir=exp/tri4b/graph_${lm_suffix}
    # use already-built graphs.
    for year in eval92 dev93; do
      steps/nnet2/decode.sh --nj 8 --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet2_online/ivectors_spk_test_$year \
         $graph_dir data/test_${year}_hires $dir/decode_${lm_suffix}_${year}_spk || touch exp/nnet2_online/.error &
    done
  done
  wait
  [ -f exp/nnet2_online/.error ] && echo "$0: Error decoding" && exit 1;
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
