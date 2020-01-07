#!/bin/bash

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

cmd=run.pl
nj=10

acwt=1.0 # change from 0.1 to 1.0 for chain model (by fangjun)
beam=12.0
lattice_beam=4.0
max_active=7000 # limit of active tokens
max_mem=50000000 # approx. limit to memory consumption during minimization in bytes
min_active=200
num_threads=20
post_decode_acwt=10  # can be used in 'chain' systems to scale acoustics by 10

. ./path.sh

. parse_options.sh || exit 1

if [ $# != 4 ]; then
    echo "Usage: $0 [options] <graph-dir> <trans_model> <nnet_output_scp> <decode-dir>"
    exit 1
fi

graphdir=$1
trans_model=$2
nnet_output_scp=$3
dir=$4

if [[ ! -d $graphdir ]]; then
  echo "graphdir $graphdir does not exist"
  exit 1
fi

if [[ ! -f $trans_model ]]; then
  echo "trans model $trans_model does not exist"
  exit 1
fi

if [[ ! -f $nnet_output_scp ]]; then
  echo "nnet_output_scp $nnet_output_scp does not exist"
  exit 1
fi

mkdir -p $dir

for i in $(seq $nj); do
  utils/split_scp.pl -j $nj $[$i - 1] $nnet_output_scp $dir/nnet_output.$i.scp
done

lat_wspecifier="ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >$dir/lat.JOB.gz"

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

$cmd --num-threads $num_threads JOB=1:$nj $dir/log/decode.JOB.log \
  latgen-faster-mapped$thread_string  \
    --acoustic-scale=$acwt \
    --allow-partial=true \
    --beam=$beam \
    --lattice-beam=$lattice_beam \
    --max-active=$max_active \
    --max-mem=$max_mem \
    --min-active=$min_active \
    --num-threads=$num_threads \
    --word-symbol-table=$graphdir/words.txt \
    $trans_model $graphdir/HCLG.fst scp:$dir/nnet_output.JOB.scp "$lat_wspecifier" || exit 1
