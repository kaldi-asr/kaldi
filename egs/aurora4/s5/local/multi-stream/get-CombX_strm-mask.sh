#!/bin/bash

. ./cmd.sh
. ./path.sh 

# fbank features,
test=data-fbank/test_eval92_new

# mstrm options
strm_indices="0:48:96:144:192:210"
comb_num=
mask_dir=

. utils/parse_options.sh

##############################
# number of stream combinations
num_streams=`echo $strm_indices | awk -F ":" '{print NF-1}'`
num_stream_combns=`echo 2^$num_streams -1|bc`

mkdir -p $mask_dir
copy-feats scp:$test/feats.scp ark:- | \
  python utils/multi-stream/pm_utils/mstrm_combn_wts_new.py \
    --nstrms=$num_streams --comb-num=$comb_num | \
  copy-feats ark:- ark,scp:$mask_dir/feats.ark,$mask_dir/feats.scp || exit 1;

exit 0;


