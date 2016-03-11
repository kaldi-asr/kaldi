#!/bin/bash

. ./cmd.sh
. ./path.sh 

stage=0
scratch=/mnt/scratch04/tmp/$USER

lang=data/lang

train=data-fbank/train_si84_multi

# mstrm options
strm_indices="0:48:96:144:192:210"
tandem_dim=120

splice=0
splice_step=1

nnet_dir=
aann_dir=

nj=60
. utils/parse_options.sh

##############################
# number of stream combinations
num_streams=`echo $strm_indices | awk -F ":" '{print NF-1}'`
all_stream_combn=`echo 2^$num_streams -1|bc`

###########################################
###########################################
tandem_feats_dir=tandem_feats/$(basename $nnet_dir)_tandem_dim${tandem_dim}
pca_transf_dir=$tandem_feats_dir/pca_transf; mkdir -p $pca_transf_dir
transf_nnet_out_opts="--apply-logit=true"

#Extract comb31 LogitPcaPhonePost
multi_stream_opts="--cross-validate=true --stream-combination=$all_stream_combn $strm_indices"
if [ $stage -le 0 ]; then

  steps/multi-stream-nnet/estimate_pca_mlpfeats.sh --nj ${nj} --cmd "${train_cmd}" \
    --est-pca-opts "--dim=${tandem_dim}" --remove-last-components 0 \
    --transf-nnet-out-opts "$transf_nnet_out_opts" \
    $train "$multi_stream_opts" $nnet_dir $pca_transf_dir || exit 1
  rm $pca_transf_dir/data/pca_acc.*
fi

if [ $stage -le 1 ]; then  
  local/make_symlink_dir.sh --tmp-root $scratch $nnet_feats_dir/$(basename $train)/data
  steps/multi-stream-nnet/make_pca_transf_mlpfeats.sh --nj ${nj} --cmd "${train_cmd}" \
    $tandem_feats_dir/$(basename $train) $train "$multi_stream_opts" $pca_transf_dir $tandem_feats_dir/$(basename $train)/log $tandem_feats_dir/$(basename $train)/data || exit 1;
  steps/compute_cmvn_stats.sh $tandem_feats_dir/$(basename $train)/ $tandem_feats_dir/$(basename $train)/log $tandem_feats_dir/$(basename $train)/data || exit 1;

fi

###########################################
###########################################
#Train AE
if [ $stage -le 2 ]; then

dir=$aann_dir/aann_dbn
mkdir -p $dir
(tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)&
$cuda_cmd $dir/log/pretrain_dbn.log \
    steps/multi-stream-nnet/pretrain_dbn.sh \
    --splice $splice --splice-step $splice_step --hid-dim "512:512:24:512:512" \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    $tandem_feats_dir/$(basename $train) $dir || exit 1;

dbn=$dir/5.dbn

utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $tandem_feats_dir/$(basename $train) $tandem_feats_dir/$(basename $train)_tr90 $tandem_feats_dir/$(basename $train)_cv10

dir=$aann_dir/aann
mkdir -p $dir
(tail --pid=$$ -F $dir/log/train_aann.log 2>/dev/null)&
$cuda_cmd $dir/log/train_aann.log \
    steps/multi-stream-nnet/train_aann.sh \
    --splice $splice --splice-step $splice_step --train-opts "--max-iters 30" \
    --proto-opts "--no-softmax --activation-type=<Sigmoid>" \
    --hid-layers 0 --dbn $dbn --learn-rate 0.0008 \
    --copy-feats "false" --skip-cuda-check "true" \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    $tandem_feats_dir/$(basename $train)_tr90 $tandem_feats_dir/$(basename $train)_cv10 $dir || exit 1;
fi 
###########################################
###########################################
exit 0;



