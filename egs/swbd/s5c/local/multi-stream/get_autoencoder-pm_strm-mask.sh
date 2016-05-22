#!/bin/bash

. ./cmd.sh
. ./path.sh 

# fbank features,
test=data-fbank/test_eval92_new

# 
tandem_transf_dir=
aann_dir=
mask_dir=

# mstrm options
strm_indices="0:48:96:144:192:210"

# mstrm decode opts
alpha=1.0
topN=-1

#decode opts
njdec=60
. utils/parse_options.sh

##############################
# number of stream combinations
num_streams=`echo $strm_indices | awk -F ":" '{print NF-1}'`
all_stream_combn=`echo 2^$num_streams -1|bc`


sdata=$test/split$njdec
[[ -d $sdata && $test/feats.scp -ot $sdata ]] || split_data.sh $test $njdec || exit 1;


logdir=$mask_dir/log; mkdir -p $logdir
aann_scores_dir=$mask_dir/aann_scores; mkdir -p $aann_scores_dir

cmd="${fast_queue_cmd}"
$cmd JOB=1:$njdec $logdir/get-AE-best-stream-combn.JOB.log \
  local/multi-stream/get-AE-best-stream-combn.sh --topN 1 \
    $sdata/JOB "$strm_indices" $tandem_transf_dir $aann_dir/ $logdir/JOB $aann_scores_dir/JOB || exit 1;

(
for ((n=1; n<=$njdec; n++)); do
  cat $aann_scores_dir/${n}/strm_mask.scp
done 
) >$mask_dir/feats.scp
    
exit 0;

