#!/bin/bash

. ./cmd.sh
. ./path.sh 

stage=1
scratch=/export/a06/$USER/tmp/
njdec=30

graph_src=exp/tri2b_multi/graph_tgpr_5k

exp="fbank-traps_mstrm_9strms-2BarkPerStrm_CMN_bnfeats_splice5_traps_dct_basis6_iters-per-epoch5" # same as run-dnn-mstrm-bnfeats_train

# fbank features,
test_id=test_eval92

# mstrm options
strm_indices="0:30:60:90:120:150:186:216:252:378"
comb_num=

. utils/parse_options.sh

##############################
# number of stream combinations
num_streams=`echo $strm_indices | awk -F ":" '{print NF-1}'`
num_stream_combns=`echo 2^$num_streams -1|bc`

if [ $stage -le 0 ]; then
  c="$test_id"
  mkdir -p data-multistream-fbank/${c}; 
  cp data/${c}/{spk2utt,spk2gender,text,utt2spk,wav.scp} data-multistream-fbank/${c}/
  steps/make_fbank.sh --fbank-config data-multistream-fbank/conf/fbank_multistream.conf \
    data-multistream-fbank/$c data-multistream-fbank/$c/log data-multistream-fbank/$c/data || exit 1
  steps/compute_cmvn_stats.sh \
    data-multistream-fbank/$c data-multistream-fbank/$c/log data-multistream-fbank/$c/data || exit 1

fi

test=data-multistream-fbank/${test_id}

nnet_dir=exp/dnn8a_bn-feat_${exp}
mask_dir="strm-mask/Comb${comb_num}_$(basename $nnet_dir)_strm-masks/$(basename $test)"; mkdir -p $mask_dir

if [ $stage -le 1 ]; then

copy-feats scp:$test/feats.scp ark:- | \
  python utils/multi-stream/pm_utils/mstrm_combn_wts_new.py \
    --nstrms=$num_streams --comb-num=$comb_num | \
  copy-feats ark:- ark,scp:$mask_dir/feats.ark,$mask_dir/feats.scp || exit 1;

dir=$nnet_dir
graph=$graph_src

multi_stream_opts="--cross-validate=true --stream-mask=scp:${mask_dir}/feats.scp $strm_indices"
steps/multi-stream-nnet/decode.sh --nj $njdec --cmd "${decode_cmd}" --num-threads 3 --scoring-opts "--min-lmwt  4 --max-lmwt 15" \
    $graph $test "$multi_stream_opts" $dir/decode_$(basename $test)_Comb${comb_num} || exit 1;

fi

test_bn=data-fbank-bn-${exp}/$(basename $test)_strm-mask_Comb${comb_num}

if [ $stage -le 2 ]; then

local/make_symlink_dir.sh --tmp-root $scratch $test_bn/data
steps/multi-stream-nnet/make_bn_feats.sh --nj $njdec --cmd "$decode_cmd" \
  --remove-last-components 2 \
  $test_bn $test "--cross-validate=true --stream-mask=scp:${mask_dir}/feats.scp $strm_indices" $nnet_dir $test_bn/log $test_bn/data || exit 1

steps/compute_cmvn_stats.sh $test_bn $test_bn/log $test_bn/data || exit 1;

fi

exit 0;



