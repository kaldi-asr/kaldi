#!/usr/bin/env bash

. ./cmd.sh


stage=1
train_stage=-10
use_gpu=true
dir=exp/nnet2_online/nnet_perturbed


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
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="--num-threads $num_threads"
fi


if [ $stage -le 1 ]; then
  # Note: if you've already run run_online_decoding_nnet2.sh you can
  # skip this stage.
  mkdir -p exp/nnet2_online
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 10 --num-frames 200000 \
    data/train 256 exp/tri3b exp/nnet2_online/diag_ubm
fi

if [ $stage -le 2 ]; then
  # Note: if you've already run run_online_decoding_nnet2.sh you can
  # skip this stage.
  # use a smaller iVector dim (50) than the default (100) because RM has a very
  # small amount of data.
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 4 \
    --ivector-dim 50 \
   data/train exp/nnet2_online/diag_ubm exp/nnet2_online/extractor || exit 1;
fi

if [ $stage -le 3 ]; then
  # Dump perturbed versions of the features.
  # store them in a sub-directory of the experimental directory.
  featdir=exp/perturbed_mfcc/feats; mkdir -p $featdir
  if [ $USER == dpovey ]; then # this shows how you can split across multiple file-systems.
    utils/create_split_dir.pl /export/b0{1,2,3,4}/dpovey/kaldi-online/egs/rm/s5/$featdir $featdir/storage
  fi
  # We can afford to run 80 jobs as we have 4 separate machines for storage.
  steps/nnet2/get_perturbed_feats.sh --cmd "$train_cmd" --feature-type mfcc --nj 80 \
    conf/mfcc.conf "$featdir" exp/perturbed_mfcc data/train data/train_perturbed_mfcc
fi


if [ $stage -le 4 ]; then
  # Align the perturbed features.
  steps/align_fmllr.sh --nj 80 --cmd "$train_cmd" \
    data/train_perturbed_mfcc data/lang exp/tri3b exp/tri3b_ali_perturbed_mfcc
fi

ivectordir=exp/nnet2_online/ivectors_perturbed_mfcc
if [ $stage -le 5 ]; then
  # Extract iVectors for the perturbed features.
  if [ $USER == dpovey ]; then # this shows how you can split across multiple file-systems.
    utils/create_split_dir.pl /export/b0{1,2,3,4}/dpovey/kaldi-online/egs/rm/s5/$ivectordir $ivectordir/storage
  fi
  # Below, setting --utts-per-spk-max to a noninteger helps to randomize the division
  # of speakers into "fake-speakers" with about 2 utterances each, by randomly making
  # some have 2 and some 3 utterances... this randomness will be different in different
  # copies of the data.
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2.5 data/train_perturbed_mfcc \
     data/train_perturbed_mfcc_max2.5

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/train_perturbed_mfcc_max2.5 exp/nnet2_online/extractor $ivectordir || exit 1;
fi


if [ $stage -le 6 ]; then
  if [ $USER == dpovey ]; then # this shows how you can split across multiple file-systems.
    # dir is the neural-net training dir.
    utils/create_split_dir.pl /export/b0{1,2,3,4}/dpovey/kaldi-online/egs/rm/s5/$dir/egs $dir/egs/storage
  fi
  # the --max-jobs-run 15 allows more of the dump_egs jobs than the default (5), since we
  # have 4 filesystems to access.  We reduce the number of epochs since we have
  # more data and we don't want so slow down the training too much, and we also
  # reduce the final learning rate (when we have a lot of data we like a ratio of 10
  # between the initial and final learning rate).  I also have --add-layers-period 2
  # which is typical when we have enough data, and increase the number of hidden layers
  # and pnorm dimentions vs. run_online_decoding_nnet2.sh since we have more data.
  steps/nnet2/train_pnorm_fast.sh --stage $train_stage \
    --splice-width 7 \
    --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors_perturbed_mfcc \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "--max-jobs-run 15" \
    --num-jobs-nnet 4 \
    --num-epochs 5 --num-epochs-extra 2 \
    --add-layers-period 2 \
    --num-hidden-layers 3 \
    --mix-up 4000 \
    --initial-learning-rate 0.02 --final-learning-rate 0.002 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 1200 \
    --pnorm-output-dim 200 \
    data/train_perturbed_mfcc data/lang exp/tri3b_ali_perturbed_mfcc $dir  || exit 1;
fi

# This time we don't bother testing with offline decoding, only with online.

if [ $stage -le 7 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh data/lang exp/nnet2_online/extractor \
    "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 8 ]; then
  # do the actual online decoding with iVectors.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph data/test ${dir}_online/decode &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph_ug data/test ${dir}_online/decode_ug || exit 1;
  wait
fi

if [ $stage -le 9 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    --per-utt true \
    exp/tri3b/graph data/test ${dir}_online/decode_per_utt &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    --per-utt true \
    exp/tri3b/graph_ug data/test ${dir}_online/decode_ug_per_utt || exit 1;
  wait
fi

exit 0;

# the experiment (with GPU)

# per-speaker (carrying adaptation info forward):
#for x in exp/nnet2_online/nnet_gpu_perturbed_online/decode*; do grep WER $x/wer_* | utils/best_wer.sh ; done
#%WER 1.62 [ 203 / 12533, 20 ins, 41 del, 142 sub ] exp/nnet2_online/nnet_gpu_perturbed_online/decode/wer_5
#%WER 8.97 [ 1124 / 12533, 87 ins, 204 del, 833 sub ] exp/nnet2_online/nnet_gpu_perturbed_online/decode_ug/wer_11

 # Note, this is the baseline with no perturbing of features, from ./run_nnet2.sh
 # [different hidden-layer configuration though.]
 #%WER 2.20 [ 276 / 12533, 25 ins, 69 del, 182 sub ] exp/nnet2_online/nnet_gpu_online/decode/wer_8
 #%WER 10.14 [ 1271 / 12533, 127 ins, 198 del, 946 sub ] exp/nnet2_online/nnet_gpu_online/decode_ug/wer_11


# per-utterance:
#%WER 1.85 [ 232 / 12533, 23 ins, 45 del, 164 sub ] exp/nnet2_online/nnet_gpu_perturbed_online/decode_per_utt/wer_5
#%WER 9.17 [ 1149 / 12533, 118 ins, 174 del, 857 sub ] exp/nnet2_online/nnet_gpu_perturbed_online/decode_ug_per_utt/wer_9

 # this is the per-utterance baseline with no perturbing of features, from ./run_nnet2.sh
 # [different hidden-layer configuration though]
 #%WER 2.21 [ 277 / 12533, 45 ins, 48 del, 184 sub ] exp/nnet2_online/nnet_gpu_online/decode_per_utt/wer_4
 #%WER 10.27 [ 1287 / 12533, 142 ins, 186 del, 959 sub ] exp/nnet2_online/nnet_gpu_online/decode_ug_per_utt/wer_10
