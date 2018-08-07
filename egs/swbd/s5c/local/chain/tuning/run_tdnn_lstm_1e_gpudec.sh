#!/bin/bash

# configs for 'chain' gpu decoding
# NOTICE: 1. we need CUDA9.0 installed with correct driver version
#         2. a GPU not earlier than K20
#         3. as libkaldi-decoder.so needs the CUDA runtime libraries, other libraries using
#            libkaldi-decoder.so also needs them.

stage=19
nj=25
data=data/eval2000_hires

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

dir=tmp/tdnn_lstm_1e_sp/
model_dir=exp/chain/tdnn_lstm_1e_sp

# model training
if [ $stage -le 19 ]; then
  local/chain/tuning/run_tdnn_lstm_1e.sh || exit 1;
fi

# acoustic inference
if [ $stage -le 20 ]; then
  mkdir -p $dir/log
  nnet3-am-copy --raw=true  $model_dir/final.mdl $dir/final.raw
  
  sdata=$data/split$nj;
  [[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
  echo $nj > $dir/num_jobs


  queue.pl  JOB=1:$nj $dir/log/post.JOB.log \
  nnet3-compute --use-gpu=no \
     --online-ivectors=scp:exp/nnet3/ivectors_eval2000/ivector_online.scp --online-ivector-period=10 --frame-subsampling-factor=3 --frames-per-chunk=140 --extra-left-context=50 --extra-right-context=0 --extra-left-context-initial=0 --extra-right-context-final=0 \
    $dir/final.raw \
    "ark,s,cs:apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |" \
     ark,scp:$dir/post.JOB.ark,$dir/post.JOB.scp
fi

# GPU decoding
if [ $stage -le 21 ]; then
  graphdir=$model_dir/graph_sw1_tg
  decdir=$dir/decode_`basename $graphdir`/
  mkdir -p $decdir
  subset=1
  latgen-faster-mapped-cuda --cuda-verbose=0 --verbose=0 --gpu-fraction=0.1 \
    --determinize-lattice=false --beam=13 --acoustic-scale=1.0 --allow-partial=true \
    --word-symbol-table=$graphdir/words.txt $model_dir/final.mdl $graphdir/HCLG.fst scp:$dir/post.$subset.scp "ark:|lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >  $decdir/lat.$subset.gz" ark,t:$decdir/ali.$subset.txt
fi

