#!/usr/bin/env bash
#Copyright 2016  Tsinghua University (Author: Dong Wang, Xuewei Zhang).  Apache 2.0.

#run from ../..
#DNN training, both xent and MPE


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

stage=0
nj=8

. utils/parse_options.sh || exit 1;

gmmdir=$1
alidir=$2
alidir_cv=$3

#generate fbanks
if [ $stage -le 0 ]; then
  echo "DNN training: stage 0: feature generation"
  rm -rf data/fbank && mkdir -p data/fbank &&  cp -R data/{train,dev,test,test_phone} data/fbank || exit 1;
  for x in train dev test; do
    echo "producing fbank for $x"
    #fbank generation
    steps/make_fbank.sh --nj $nj --cmd "$train_cmd" data/fbank/$x exp/make_fbank/$x fbank/$x || exit 1
    #ompute cmvn
    steps/compute_cmvn_stats.sh data/fbank/$x exp/fbank_cmvn/$x fbank/$x || exit 1
  done

  echo "producing test_fbank_phone"
  cp data/fbank/test/feats.scp data/fbank/test_phone && cp data/fbank/test/cmvn.scp data/fbank/test_phone || exit 1;

fi


#xEnt training
if [ $stage -le 1 ]; then
  outdir=exp/tri4b_dnn
  #NN training
  (tail --pid=$$ -F $outdir/log/train_nnet.log 2>/dev/null)& # forward log
  $cuda_cmd $outdir/log/train_nnet.log \
    steps/nnet/train.sh --copy_feats false --cmvn-opts "--norm-means=true --norm-vars=false" --hid-layers 4 --hid-dim 1024 \
    --learn-rate 0.008 data/fbank/train data/fbank/dev data/lang $alidir $alidir_cv $outdir || exit 1;
  #Decode (reuse HCLG graph in gmmdir)
  (
    steps/nnet/decode.sh --nj $nj --cmd "$decode_cmd" --srcdir $outdir --config conf/decode_dnn.config --acwt 0.1 \
      $gmmdir/graph_word data/fbank/test $outdir/decode_test_word || exit 1;
  )&
  (
   steps/nnet/decode.sh --nj $nj --cmd "$decode_cmd" --srcdir $outdir --config conf/decode_dnn.config --acwt 0.1 \
     $gmmdir/graph_phone data/fbank/test_phone $outdir/decode_test_phone || exit 1;
  )&

fi

#MPE training

srcdir=exp/tri4b_dnn
acwt=0.1

if [ $stage -le 2 ]; then
  # generate lattices and alignments
  steps/nnet/align.sh --nj $nj --cmd "$train_cmd" \
    data/fbank/train data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj $nj --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    data/fbank/train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 3 ]; then
  outdir=exp/tri4b_dnn_mpe
  #Re-train the DNN by 3 iteration of MPE
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 3 --acwt $acwt --do-smbr false \
    data/fbank/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $outdir || exit 1
  #Decode (reuse HCLG graph)
  for ITER in 3 2 1; do
   (
    steps/nnet/decode.sh --nj $nj --cmd "$decode_cmd" --nnet $outdir/${ITER}.nnet --config conf/decode_dnn.config --acwt $acwt \
      $gmmdir/graph_word data/fbank/test $outdir/decode_test_word_it${ITER} || exit 1;
   )&
   (
   steps/nnet/decode.sh --nj $nj --cmd "$decode_cmd" --nnet $outdir/${ITER}.nnet --config conf/decode_dnn.config --acwt $acwt \
     $gmmdir/graph_phone data/fbank/test_phone $outdir/decode_test_phone_it${ITER} || exit 1;
   )&
  done
fi

