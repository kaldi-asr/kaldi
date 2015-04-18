#!/bin/bash -u

. ./cmd.sh
. ./path.sh

# DNN training. This script is based on egs/swbd/s5b/local/run_dnn.sh
# Shinji Watanabe

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s <mic condition(ihm|sdm|mdm)>\n\n" `basename $0`
  exit 1;
fi

mic=$1
final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

for lm_suffix in $LM; do
# Config:
gmmdir=exp/$mic/tri4a
graph_dir=exp/$mic/tri4a/graph_${lm_suffix}
data_fmllr=data-fmllr-tri4
stage=0 # resume training with --stage=N
# End of config.
. utils/parse_options.sh || exit 1;
#

if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,
  # test
  dir=$data_fmllr/$mic/eval
  steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_eval_${lm_suffix} \
     $dir data/$mic/eval $gmmdir $dir/log $dir/data || exit 1
  # dev
  dir=$data_fmllr/$mic/dev
  steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_dev_${lm_suffix} \
     $dir data/$mic/dev $gmmdir $dir/log $dir/data || exit 1
  # train
  dir=$data_fmllr/$mic/train
  steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali \
     $dir data/$mic/train $gmmdir $dir/log $dir/data || exit 1
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs
  dir=exp/$mic/dnn4_pretrain-dbn
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --rbm-iter 1 $data_fmllr/$mic/train $dir || exit 1;
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/$mic/dnn4_pretrain-dbn_dnn
  ali=${gmmdir}_ali
  feature_transform=exp/$mic/dnn4_pretrain-dbn/final.feature_transform
  dbn=exp/$mic/dnn4_pretrain-dbn/6.dbn
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/$mic/train_tr90 $data_fmllr/$mic/train_cv10 data/lang $ali $ali $dir || exit 1;
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.conf --acwt 0.1 \
    --num-threads 3 --parallel-opts "-pe smp 4" \
    $graph_dir $data_fmllr/$mic/dev $dir/decode_dev_${lm_suffix} || exit 1;
  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.conf --acwt 0.1 \
    --num-threads 3 --parallel-opts "-pe smp 4" \
    $graph_dir $data_fmllr/$mic/eval $dir/decode_eval_${lm_suffix} || exit 1;
fi

# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. We use usually good acwt 0.1
# Lattices are re-generated after 1st epoch, to get faster convergence.
dir=exp/$mic/dnn4_pretrain-dbn_dnn_smbr
srcdir=exp/$mic/dnn4_pretrain-dbn_dnn
acwt=0.1

if [ $stage -le 3 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 6 --cmd "$train_cmd" \
    $data_fmllr/$mic/train data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.conf \
    --acwt $acwt $data_fmllr/$mic/train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 4 ]; then
  # Re-train the DNN by 1 iteration of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt --do-smbr true \
    $data_fmllr/$mic/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode (reuse HCLG graph)
  for ITER in 1; do
    steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.conf \
      --num-threads 3 --parallel-opts "-pe smp 4" \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $graph_dir $data_fmllr/$mic/dev $dir/decode_dev_${lm_suffix} || exit 1;
    steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.conf \
      --num-threads 3 --parallel-opts "-pe smp 4" \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $graph_dir $data_fmllr/$mic/eval $dir/decode_eval_${lm_suffix} || exit 1;
  done 
fi

# Re-generate lattices, run 4 more sMBR iterations
dir=exp/$mic/dnn4_pretrain-dbn_dnn_smbr_i1lats
srcdir=exp/$mic/dnn4_pretrain-dbn_dnn_smbr
acwt=0.1

if [ $stage -le 5 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 6 --cmd "$train_cmd" \
    $data_fmllr/$mic/train data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.conf \
    --acwt $acwt $data_fmllr/$mic/train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 6 ]; then
  # Re-train the DNN by 1 iteration of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt --do-smbr true \
    $data_fmllr/$mic/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode (reuse HCLG graph)
  for ITER in 1 2 3 4; do
    steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.conf \
      --num-threads 3 --parallel-opts "-pe smp 4" \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $graph_dir $data_fmllr/$mic/dev $dir/decode_dev_${lm_suffix}_$ITER || exit 1;
    steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.conf \
      --num-threads 3 --parallel-opts "-pe smp 4" \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $graph_dir $data_fmllr/$mic/eval $dir/decode_eval_${lm_suffix}_$ITER || exit 1;
  done 
fi

done
# Getting results [see RESULTS file]
# for x in exp/$mic/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done

