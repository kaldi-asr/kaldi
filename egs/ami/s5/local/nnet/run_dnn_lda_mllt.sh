#!/bin/bash -u

. ./cmd.sh
. ./path.sh

# DNN training. This script is based on egs/swbd/s5b/local/run_dnn.sh
# Shinji Watanabe, Karel Vesely,

# Config:
nj=80

stage=0 # resume training with --stage=N
. utils/parse_options.sh || exit 1;
#

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s [opts] <mic condition(ihm|sdm|mdm)>\n\n" `basename $0`
  exit 1;
fi
mic=$1

gmmdir=exp/$mic/tri3a

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$gmmdir/graph_${LM}

# Set bash to 'debug' mode, it will exit on : 
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
set -x

nj_dev=$(cat data/$mic/dev/spk2utt | wc -l)
nj_eval=$(cat data/$mic/eval/spk2utt | wc -l)

# split the data : 90% train 10% cross-validation (held-out),
if [ $stage -le 0 ]; then
  dir=data/$mic/train
  [ ! -e ${dir}_tr90 ] && utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10
fi

# Pre-train DBN, i.e. a stack of RBMs,
if [ $stage -le 1 ]; then
  dir=exp/$mic/dnn4noSAT_pretrain-dbn; [ ! -e $dir ] && mkdir -p $dir

  # Make feature_transform prototype : feats -> splice -> LDA+MLLT -> splice,
  # - re-use LDA+MLLT matrix from GMM system,
  # - re-use CMVN options,
  feat_dim=$(feat-to-dim scp:data/$mic/train/feats.scp -)
  cmvn_opts=$(cat $gmmdir/cmvn_opts)
  [ -z $cmvn_opts ] && cmvn_opts="--norm-means=true --norm-vars=false" # GMM default,
  {
    echo "<Splice> <InputDim> $feat_dim <OutputDim> $((feat_dim*7)) <ReadVector> [ -3 -2 -1 0 1 2 3 ]"
    echo "<LinearTransform> <InputDim> $((feat_dim*7)) <OutputDim> 40 <ReadMatrix> $gmmdir/final.mat"
    echo "<Splice> <InputDim> 40 <OutputDim> 520 <BuildVector> -10 -5:5 10 </BuildVector>"
  } > $dir/feature_transform.proto

  # Pre-training,
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --rbm-iter 1 \
      --cmvn-opts "$cmvn_opts" \
      --feature-transform-proto $dir/feature_transform.proto \
      data/$mic/train $dir
fi

# Train the DNN optimizing per-frame cross-entropy,
if [ $stage -le 2 ]; then
  dir=exp/$mic/dnn4noSAT_pretrain-dbn_dnn
  ali=${gmmdir}_ali
  feature_transform=exp/$mic/dnn4noSAT_pretrain-dbn/final.feature_transform
  dbn=exp/$mic/dnn4noSAT_pretrain-dbn/6.dbn
  # Train,
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    data/$mic/train_tr90 data/$mic/train_cv10 data/lang $ali $ali $dir
  # Decode (reuse HCLG graph),
  steps/nnet/decode.sh --nj $nj_dev --cmd "$decode_cmd" --config conf/decode_dnn.conf --acwt 0.1 \
    --num-threads 3 \
    $graph_dir data/$mic/dev $dir/decode_dev_${LM}
  steps/nnet/decode.sh --nj $nj_eval --cmd "$decode_cmd" --config conf/decode_dnn.conf --acwt 0.1 \
    --num-threads 3 \
    $graph_dir data/$mic/eval $dir/decode_eval_${LM}
fi


# Sequence training using sMBR criterion, we do Stochastic-GD with 
# per-utterance updates. We use usually good acwt 0.1.
# Lattices are not regenerated (it is faster).

dir=exp/$mic/dnn4noSAT_pretrain-dbn_dnn_smbr
srcdir=exp/$mic/dnn4noSAT_pretrain-dbn_dnn
acwt=0.1

# Generate lattices and alignments,
if [ $stage -le 3 ]; then
  steps/nnet/align.sh --nj $nj --cmd "$train_cmd" \
    data/$mic/train data/lang $srcdir ${srcdir}_ali
  steps/nnet/make_denlats.sh --nj $nj --cmd "$decode_cmd" --config conf/decode_dnn.conf \
    --acwt $acwt data/$mic/train data/lang $srcdir ${srcdir}_denlats
fi

# Re-train the DNN by 4 epochs of sMBR,
if [ $stage -le 4 ]; then
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt --do-smbr true \
    data/$mic/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir
  # Decode (reuse HCLG graph)
  for ITER in 4 1; do
    steps/nnet/decode.sh --nj $nj_dev --cmd "$decode_cmd" --config conf/decode_dnn.conf \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $graph_dir data/$mic/dev $dir/decode_dev_${LM}_it${ITER}
    steps/nnet/decode.sh --nj $nj_eval --cmd "$decode_cmd" --config conf/decode_dnn.conf \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $graph_dir data/$mic/eval $dir/decode_eval_${LM}_it${ITER}
  done
fi

# Getting results [see RESULTS file]
# for x in exp/$mic/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done

