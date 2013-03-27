#!/bin/bash
# Caution- what we call "eval" here is really the full dev-set.

. conf/common_vars.sh
. ./lang.conf || exit 1;


#Preparing eval.pem directories

if [[ ! -z $eval_data_list ]] ; then
    echo "Subsetting the EVAL set"

    local/make_corpus_subset.sh $eval_data_dir $eval_data_list ./data/raw_eval_data || exit 1
    eval_data_dir=`readlink -f ./data/raw_eval_data`


    nj_max=`cat $eval_data_list | wc -l`
    if [[ "$nj_max" -lt "$decode_nj" ]] ; then
        echo "The maximum reasonable number of jobs is $nj_max -- you have $decode_nj! (The training and decoding process has file-granularity)"
        decode_nj=$nj_max
    fi
    eval_data_dir=`readlink -f ./data/raw_eval_data`
fi


echo --------------------------------------------------------------------
echo "Preparing eval data lists in data/eval.pem on" `date`
echo --------------------------------------------------------------------
mkdir -p data/eval.pem
local/prepare_acoustic_training_data.pl \
    --fragmentMarkers \-\*\~ \
    $eval_data_dir data/eval.pem > data/eval.pem/skipped_utts.log || exit 1

echo -------------------------------------------------------------------
echo "Preparing eval stm files in data/eval.pem on" `date`
echo -------------------------------------------------------------------
local/prepare_stm.pl --fragmentMarkers \-\*\~ data/eval.pem || exit 1
cp $glmFile data/eval.pem/glm

echo -------------------------------------------------------------------
echo "Preparing keyword spotting system on" `date`
echo -------------------------------------------------------------------
if [[ $subset_ecf ]] ; then
  [ ! -f $eval_data_list ] && echo "empty variable eval_data_list" && exit 1;
  local/kws_setup.sh --case-insensitive $case_insensitive --subset-ecf $eval_data_list \
      $ecf_file $kwlist_file $rttm_file data/lang data/eval.pem || exit 1
else
  local/kws_setup.sh --case-insensitive $case_insensitive \
     $ecf_file $kwlist_file $rttm_file data/lang data/eval.pem || exit 1
fi


if [ ! -f data/eval.pem/feats.scp ]; then
  echo -------------------------------------------------------------------
  echo "Preparing eval stm files in data/eval.pem on" `date`
  echo -------------------------------------------------------------------
  steps/make_plp.sh \
    --cmd "$train_cmd" --nj $decode_nj \
    data/eval.pem exp/make_plp/eval.pem plp || exit 1
  steps/compute_cmvn_stats.sh \
    data/eval.pem exp/make_plp/eval.pem plp || exit 1
  # In case plp extraction failed on some utterance, delist them
  utils/fix_data_dir.sh data/eval.pem
fi

if [[ -d exp/tri5 ]] ; then 
  n=5
else
  n=4
fi
    
echo --------------------------------------------------------------------------
echo "Starting eval.pem decoding using exp/tri$n on" `date`
echo --------------------------------------------------------------------------
mkdir -p exp/tri5/decode_eval.pem
steps/decode_fmllr.sh --nj $decode_nj \
  --cmd "queue.pl -l arch='*64*' -l mem_free=2G,ram_free=0.5g" \
  --num-threads 6 --parallel-opts "-pe smp 5" \
  exp/tri$n/graph data/eval.pem exp/tri$n/decode_eval.pem &> exp/tri$n/decode_eval.pem.log 

echo --------------------------------------------------------------------------
echo "Starting eval.pem keyword spotting using exp/tri$n on" `date`
echo --------------------------------------------------------------------------
local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
  data/lang data/eval.pem exp/tri$n/decode_eval.pem &> exp/tri$n/kws_eval.pem.log
    
echo --------------------------------------------------------------------------
echo "Starting eval.pem decoding using exp/sgmm5 on" `date`
echo --------------------------------------------------------------------------
steps/decode_sgmm2.sh --use-fmllr true \
  --cmd "queue.pl -l arch='*64*' -l mem_free=2G,ram_free=0.6g" \
  --num-threads 6 --parallel-opts "-pe smp 5" \
  exp/sgmm5/graph data/eval.pem exp/sgmm5/decode_fmllr_eval.pem &> exp/sgmm5/decode_fmllr_eval.pem.log

echo --------------------------------------------------------------------------
echo "Starting eval.pem keyword spotting using exp/sgmm5 on" `date`
echo --------------------------------------------------------------------------
local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
  data/lang data/eval.pem exp/sgmm5/decode_fmllr_eval.pem &> exp/sgmm5/kws_eval.pem.fmllr.log

echo --------------------------------------------------------------------------
echo "Starting exp/sgmm5_mmi_b0.1/decode[_fmllr] and keyword spotting on" `date`
echo --------------------------------------------------------------------------
for iter in 1 2 3 4; do
  steps/decode_sgmm2_rescore.sh \
    --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode_eval.pem \
    data/lang data/eval.pem exp/sgmm5/decode_fmllr_eval.pem exp/sgmm5_mmi_b0.1/decode_fmllr_eval.pem_it$iter
  
  local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
    data/lang data/eval.pem exp/sgmm5_mmi_b0.1/decode_fmllr_eval.pem_it$iter &> exp/sgmm5_mmi_b0.1/kws_fmllr_eval.pem.log
done


echo -----------------------------------------------------
echo "Finished successfully on" `date`
echo -----------------------------------------------------

exit 0
