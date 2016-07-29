#!/bin/bash

. ./cmd.sh
. ./path.sh 

stage=1
scratch=/export/a07/$USER/tmp/
nj=40

lang=data/lang
ali_src=exp/tri2b_multi_ali_si84
graph_src=exp/tri2b_multi/graph_tgpr_5k

exp="fbank-traps_mstrm_9strms-2BarkPerStrm_CMN_bnfeats_splice5_traps_dct_basis6_iters-per-epoch5"

# fbank features,
train=data-multistream-fbank/train_si84_multi
test=data-multistream-fbank/test_eval92_new

# mstrm options
strm_indices="0:30:60:90:120:150:186:216:252:378"
logit_pca_dim=120
splice=0
splice_step=1


# non-repeating stuff
autoencoder_pm_train_scores_dir=
mdelta_pm_train_scores_dir=
mdelta_stats_dir=

# mstrm decode opts
topN=1
alpha=1.0

pm_suffix=

#decode opts
njdec=60

. utils/parse_options.sh

##############################
# number of stream combinations
num_streams=`echo $strm_indices | awk -F ":" '{print NF-1}'`
all_stream_combn=`echo 2^$num_streams -1|bc`


###########################################
###########################################

nnet_dir=exp/dnn8a_bn-feat_${exp}
ali=$ali_src

nnet_feats_dir=logit-pca-state-post_${logit_pca_dim}D/$(basename $nnet_dir)
pca_transf_dir=$nnet_feats_dir/pca_transf; mkdir -p $pca_transf_dir
transf_nnet_out_opts="--apply-logit=true"

#Extract comb31 LogitPcaPhonePost
if [ $stage -le 1 ]; then

  multi_stream_opts="--cross-validate=true --stream-combination=$all_stream_combn $strm_indices"
  steps/multi-stream-nnet/estimate_pca_mlpfeats.sh --nj ${nj} --cmd "${train_cmd}" \
    --est-pca-opts "--dim=${logit_pca_dim}" --remove-last-components 0 \
    --transf-nnet-out-opts "$transf_nnet_out_opts" \
    $train "$multi_stream_opts" $nnet_dir $pca_transf_dir || exit 1
  
  local/make_symlink_dir.sh --tmp-root $scratch $nnet_feats_dir/$(basename $train)/data
  steps/multi-stream-nnet/make_pca_transf_mlpfeats.sh --nj ${nj} --cmd "${train_cmd}" \
    $nnet_feats_dir/$(basename $train) $train "$multi_stream_opts" $pca_transf_dir $nnet_feats_dir/$(basename $train)/log $nnet_feats_dir/$(basename $train)/data || exit 1;
  steps/compute_cmvn_stats.sh $nnet_feats_dir/$(basename $train)/ $nnet_feats_dir/$(basename $train)/log $nnet_feats_dir/$(basename $train)/data || exit 1;

fi
###########################################
###########################################

aann_dir=exp_aann/aann_logit-pca-statepost_${logit_pca_dim}D_$(basename $nnet_dir)_splice${splice}_splice-step${splice_step}/
#Train AE
if [ $stage -le 2 ]; then

dir=$aann_dir/aann_dbn
mkdir -p $dir
(tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)&
$cuda_cmd $dir/log/pretrain_dbn.log \
    steps/multi-stream-nnet/pretrain_dbn.sh \
    --splice $splice --splice-step $splice_step --hid-dim "512:512:24:512:512" \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    $nnet_feats_dir/$(basename $train) $dir || exit 1;

dbn=$dir/5.dbn

utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $nnet_feats_dir/$(basename $train) $nnet_feats_dir/$(basename $train)_tr90 $nnet_feats_dir/$(basename $train)_cv10

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
    $nnet_feats_dir/$(basename $train)_tr90 $nnet_feats_dir/$(basename $train)_cv10 $dir || exit 1;
fi 
###########################################
###########################################

mask_dir="strm-mask/autoencoder-mdeltaPM_$(basename $nnet_dir)_strm-masks${pm_suffix:+_$pm_suffix}/$(basename $test)"
pm_scores_dir=$mask_dir/autoencoder-mdeltaPM_scores; mkdir -p $pm_scores_dir
logdir=$mask_dir/log; mkdir -p $logdir

[ -z $autoencoder_pm_train_scores_dir ] && autoencoder_pm_train_scores_dir=$pm_scores_dir/autoencoder_$(basename $train)_scores
[ -z $mdelta_stats_dir ] && mdelta_stats_dir=$pm_scores_dir/MDelta_stats; mkdir -p $mdelta_stats_dir
[ -z $mdelta_pm_train_scores_dir ] && mdelta_pm_train_scores_dir=$pm_scores_dir/mdelta_$(basename $train)_scores

if [ $stage -le 3 ]; then 

# get autoencoder training data scores
autoencoder_pm_train_scores_pklz=$autoencoder_pm_train_scores_dir/comb.${all_stream_combn}.pklz
if [ ! -f $autoencoder_pm_train_scores_pklz ]; then
  multi_stream_opts="--cross-validate=true --stream-combination=$all_stream_combn $strm_indices"
  steps/multi-stream-nnet/combine_make_pca_transf_mlpfeats_compute_mse_aann.sh \
    --cmd "${decode_cmd}" --nj ${nj} \
    $autoencoder_pm_train_scores_dir/mse $train "$multi_stream_opts" $pca_transf_dir $aann_dir/aann $autoencoder_pm_train_scores_dir/mse/log $autoencoder_pm_train_scores_dir/mse/data || exit 1;

  python utils/multi-stream/pm_utils/mse_to_pkl_scores.py $autoencoder_pm_train_scores_dir/mse/feats.scp $autoencoder_pm_train_scores_dir/comb.${all_stream_combn}.pklz || exit 1;
else
  echo "autoencoder_pm_train_scores_pklz = $autoencoder_pm_train_scores_pklz exits ......... using that"
  sleep 3
fi

if [ ! -f $mdelta_stats_dir/pri ]; then 
  $decode_cmd $mdelta_stats_dir/compute_mdelta_stats.log \
  local/multi-stream/run_preprocess_mdelta.sh \
    --lang $lang --ali-dir $ali \
    --mdelta-stats-dir $mdelta_stats_dir || exit 1;
else
  echo "mdelta_stats_dir/pri = $mdelta_stats_dir/pri exists ......... using that"
  sleep 3
fi

mdelta_pm_train_scores_pklz=$mdelta_pm_train_scores_dir/comb.${all_stream_combn}.pklz

if [ ! -f $mdelta_pm_train_scores_pklz ]; then
  multi_stream_opts="--cross-validate=true --stream-combination=$all_stream_combn $strm_indices"
  steps/multi-stream-nnet/compute_mdelta_scores.sh \
  --cmd "${decode_cmd}" --nj ${nj} \
  $train "$multi_stream_opts" $nnet_dir $mdelta_stats_dir $mdelta_pm_train_scores_dir/log $mdelta_pm_train_scores_dir/data

  (cd $mdelta_pm_train_scores_dir; ln -s data/mdelta_scores.$(basename $train).pklz comb.${all_stream_combn}.pklz; cd -)
else
  echo "mdelta_pm_train_scores_pklz = $mdelta_pm_train_scores_pklz exists ......... using that"
  sleep 3
fi 

fi

##########
if [ $stage -le 4 ]; then

sdata=$test/split$njdec
[[ -d $sdata && $test/feats.scp -ot $sdata ]] || split_data.sh $test $njdec || exit 1;

run.pl --max-jobs-run $njdec JOB=1:$njdec $logdir/get-AE_mdelta-best-stream-combn.JOB.log \
  steps/multi-stream-nnet/get-AE_mdelta-best-stream-combn_new.sh --cmd "${fast_queue_cmd}" \
    --tmproot /mnt/data/$USER/kaldi.XXXXXXX \
    --alpha ${alpha} --topN ${topN} --pms-combn-weights "0.5:0.5" \
    --pms-train-stats "$autoencoder_pm_train_scores_dir/comb.${all_stream_combn}.pklz:$mdelta_pm_train_scores_dir/comb.${all_stream_combn}.pklz" \
    $sdata/JOB "$strm_indices" $pca_transf_dir $aann_dir/aann $mdelta_stats_dir $logdir/JOB $pm_scores_dir/JOB || exit 1;

(
for ((n=1; n<=$njdec; n++)); do
  cat $pm_scores_dir/${n}/strm_mask.scp
done 
) >$mask_dir/feats.scp
    
fi

###########################################
###########################################

#Run decode
mask_dir="strm-mask/autoencoder-mdeltaPM_$(basename $nnet_dir)_strm-masks${pm_suffix:+_$pm_suffix}/$(basename $test)"
if [ $stage -le 5 ]; then
  dir=$nnet_dir
  #Direct decode
  ali=$ali_src
  graph=$graph_src

  multi_stream_opts="--cross-validate=true --stream-mask=scp:${mask_dir}/feats.scp $strm_indices"
  steps/multi-stream-nnet/decode.sh --nj $njdec --cmd "${decode_cmd}" --num-threads 3 \
    $graph $test "$multi_stream_opts" $dir/decode_$(basename $test)_$(basename $graph)_autoencoder-mdeltaPM${pm_suffix:+_$pm_suffix} || exit 1;

fi

test_bn=data-fbank-bn-${exp}/$(basename $test)_strm-mask_autoencoder-mdeltaPM${pm_suffix:+_$pm_suffix}
if [ $stage -le 6 ]; then

  local/make_symlink_dir.sh --tmp-root $scratch $test_bn/data
  steps/multi-stream-nnet/make_bn_feats.sh --nj $njdec --cmd "$decode_cmd" \
    --remove-last-components 2 \
    $test_bn $test "--cross-validate=true --stream-mask=scp:${mask_dir}/feats.scp $strm_indices" $nnet_dir $test_bn/log $test_bn/data || exit 1
  steps/compute_cmvn_stats.sh $test_bn $test_bn/log $test_bn/data || exit 1;

fi


exit 0;



