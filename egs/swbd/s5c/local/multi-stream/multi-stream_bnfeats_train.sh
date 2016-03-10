#!/bin/bash

. ./cmd.sh
. ./path.sh 

# source data,
ali=exp/tri5b_kplpf0_ali
lang=data/lang

# outdir
dir=exp/dnn_bn-featXtractor

bn_nnet_proto=
append_feature_transform=

# bnfeatXtractor opts
bn_splice=5
bn_traps_dct_basis=6

# fbank features,
train=data-fbank/training.seg1

#mstrm opts
strm_indices="0:48:96:144:192:210"
iters_per_epoch=1

. utils/parse_options.sh

##############################
# number of stream combinations
num_streams=`echo $strm_indices | awk -F ":" '{print NF-1}'`
all_stream_combn=`echo 2^$num_streams -1|bc`

# create append_feature_transform
num_fbank=$(feat-to-dim "ark:copy-feats scp:${train}/feats.scp ark:- |" -)

mkdir -p $dir
if [ -z "$append_feature_transform" ]; then
python -c "
num_fbank=${num_fbank}
traps_dct_basis=${bn_traps_dct_basis}
print '<Nnet>'
print '<Copy> '+str(num_fbank*traps_dct_basis)+' '+str(num_fbank*traps_dct_basis)
print '[',
for i in xrange(0, num_fbank):
  for j in xrange(0, traps_dct_basis):
    print (i+num_fbank*j)+1,
print ']'
print '</Nnet>'
" >$dir/append_rearrange_subband-group_hamm_dct_${num_fbank}Fbank_${bn_traps_dct_basis}dctbasis.nnet 

append_feature_transform="$dir/append_rearrange_subband-group_hamm_dct_${num_fbank}Fbank_${bn_traps_dct_basis}dctbasis.nnet"
fi

# BNfeats neural network
utils/subset_data_dir_tr_cv.sh ${train} ${train}_tr90 ${train}_cv10 || exit 1;

$cuda_cmd $dir/log/train_nnet.log \
steps/multi-stream-nnet/train.sh \
  --scheduler-opts "--iters-per-epoch $iters_per_epoch" \
  --cmvn-opts "--norm-means=true --norm-vars=false" \
  --feat-type traps --splice $bn_splice --traps-dct-basis $bn_traps_dct_basis --learn-rate 0.008 \
  ${bn_nnet_proto:+ --nnet-proto $bn_nnet_proto} ${append_feature_transform:+ --append-feature-transform "$append_feature_transform"} \
  --proto-opts "--bottleneck-before-last-affine" --bn-dim 40 --hid-dim 1500 --hid-layers 3 \
  ${train}_tr90 ${train}_cv10 $lang $ali $ali $strm_indices $dir || exit 1

exit 0;


