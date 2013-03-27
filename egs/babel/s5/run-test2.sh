#!/bin/bash 
set -e
set -o pipefail

echo "$0 $@"  # Print the command line for logging

decode_test=false #Test is decoded as a part of shadow.uem, so typically is not necessary to decode it on its own
decode_eval=true
decode_shadow=true
decode_nnet=true

[ -f ./path.sh ] && . ./path.sh; # source the path.
[ -f ./cmd.sh ] && . ./cmd.sh; # source the path.
. parse_options.sh || exit 1;

. $1

if [[ `pwd` != *full* || $1 != *full* ]] ; then
  echo "Config file or working directory probably belongs to limitedLP setup"
  exit 1
fi


[ ! -f $test_data_cmudb_shadow ] && echo "The CMU db does not exist... " && exit 1
[ ! -f $test_data_list ] && echo "The test data list does not exist... " && exit 1
[ ! -d $test_data_dir ] && echo "The test data directory does not exist... " && exit 1

[ -z $cer ] && cer=0

#####################################################################
#
# test.uem and shadow.uem directory preparation
#
#####################################################################
if false ; then
if  $decode_shadow || $decode_test ; then
  local/cmu_uem2kaldi_dir.sh --filelist $test_data_list $test_data_cmudb_shadow  $test_data_dir data/test.uem
  steps/make_plp.sh --cmd "$train_cmd" --nj $decode_nj data/test.uem exp/make_plp/test.uem plp

  steps/compute_cmvn_stats.sh data/test.uem exp/make_plp/test.uem plp
  utils/fix_data_dir.sh data/test.uem

  local/kws_setup.sh --case-insensitive $case_insensitive $test_data_ecf $test_data_kwlist data/lang data/test.uem/

  local/create_shadow_dataset.sh data/shadow.uem data/dev data/test.uem
  local/kws_data_prep.sh --case-insensitive $case_insensitive data/lang data/shadow.uem data/shadow.uem/kws || exit 1
  utils/fix_data_dir.sh data/shadow.uem
fi

#####################################################################
#
# eval.uem directory preparation
#
#####################################################################
if $decode_eval ; then
  local/cmu_uem2kaldi_dir.sh --filelist $eval_data_list $test_data_cmudb_shadow  $eval_data_dir data/eval.uem
  cp data/eval.pem/stm data/eval.uem
  cp data/eval.pem/glm data/eval.uem
  if [ $cer == 1 ] ; then
    [ ! -f data/eval.pem/char.stm ] && echo "CER=1 and file eval.pem/char.stm not present" && exit 1
    cp data/eval.pem/char.stm data/eval.uem
  fi
  steps/make_plp.sh --cmd "$train_cmd" --nj $decode_nj data/eval.uem exp/make_plp/eval.uem plp
  steps/compute_cmvn_stats.sh data/eval.uem exp/make_plp/eval.uem plp
  utils/fix_data_dir.sh data/eval.uem

  local/kws_setup.sh --case-insensitive $case_insensitive $ecf_file $test_data_kwlist $rttm_file data/lang data/eval.uem
fi

fi

####################################################################
##
## FMLLR decoding of shadow.uem
##
####################################################################
if $decode_shadow ; then
  steps/decode_fmllr_extra.sh --skip-scoring true --nj 64 --cmd "$decode_cmd" \
    --num-threads 6 --parallel-opts "-pe smp 6 -l ram_free=0.5G" \
    exp/tri5/graph data/shadow.uem exp/tri5/decode_shadow.uem  | tee  exp/tri5/decode_shadow.uem.log
  #-local/lattice_to_ctm.sh --cmd "$decode_cmd" data/shadow.uem data/lang exp/tri5/decode_shadow.uem.si
  #-local/lattice_to_ctm.sh --cmd "$decode_cmd" data/shadow.uem data/lang exp/tri5/decode_shadow.uem
  #-split_ctms exp/tri5/decode_shadow.uem.si data/dev data/test.uem
  #-split_ctms exp/tri5/decode_shadow.uem data/dev data/test.uem
  #-local/shadow_set_kws_search.sh --cmd "$decode_cmd" data/shadow.uem  data/lang exp/tri5/decode_shadow.uem.si data/dev data/test.uem
  #-local/shadow_set_kws_search.sh --cmd "$decode_cmd" data/shadow.uem  data/lang exp/tri5/decode_shadow.uem data/dev data/test.uem
fi

####################################################################
##
## FMLLR decoding of test.uem
##
####################################################################
if $decode_test ; then
  steps/decode_fmllr_extra.sh --skip-scoring true --nj 64 --cmd "$decode_cmd" \
    --num-threads 6 --parallel-opts "-pe smp 6 -l ram_free=0.5G" \
    exp/tri5/graph data/test.uem exp/tri5/decode_test.uem  | tee  exp/tri5/decode_test.uem.log
  #-local/lattice_to_ctm.sh --cmd "$decode_cmd" data/test.uem data/lang exp/tri5/decode_test.uem.si
  #-local/lattice_to_ctm.sh --cmd "$decode_cmd" data/test.uem data/lang exp/tri5/decode_test.uem
  #-local/kws_search.sh --skip-scoring true --cmd "$decode_cmd" data/lang data/test.uem/ exp/tri5/decode_test.uem.si
  #-local/kws_search.sh --skip-scoring true --cmd "$decode_cmd" data/lang data/test.uem/ exp/tri5/decode_test.uem
fi 

####################################################################
##
## FMLLR decoding of test.uem
##
####################################################################
if $decode_eval ; then
  steps/decode_fmllr_extra.sh --skip-scoring true --nj 64 --cmd "$decode_cmd" \
    --num-threads 6 --parallel-opts "-pe smp 6 -l ram_free=0.5G" \
    exp/tri5/graph data/eval.uem exp/tri5/decode_eval.uem |tee exp/tri5/decode_eval.uem.log
  #-local/lattice_to_ctm.sh --cmd "$decode_cmd" data/eval.uem data/lang exp/tri5/decode_eval.uem.si
  #-local/lattice_to_ctm.sh --cmd "$decode_cmd" data/eval.uem data/lang exp/tri5/decode_eval.uem
  #-local/kws_search.sh --skip-scoring true --cmd "$decode_cmd" data/lang data/eval.uem/ exp/tri5/decode_eval.uem.si
  #-local/kws_search.sh --skip-scoring true --cmd "$decode_cmd" data/lang data/eval.uem/ exp/tri5/decode_eval.uem
fi

####################################################################
##
## SGMM2 decoding of shadow.uem
##
####################################################################
if $decode_shadow ; then
  #steps/decode_sgmm2.sh --skip-scoring true --nj 64 --cmd "$decode_cmd" --transform-dir exp/tri5/decode_shadow.uem \
  #        exp/sgmm5/graph data/shadow.uem exp/sgmm5/decode_shadow.uem  | tee  exp/sgmm5/decode_shadow.uem.log
  steps/decode_sgmm2.sh --skip-scoring true --use-fmllr true --nj 64 --cmd "$decode_cmd" --transform-dir exp/tri5/decode_shadow.uem \
    --num-threads 6 --parallel-opts "-pe smp 6 -l ram_free=0.5G" \
    exp/sgmm5/graph data/shadow.uem exp/sgmm5/decode_fmllr_shadow.uem  | tee  exp/sgmm5/decode_fmllr_shadow.uem.log
  
  #-local/lattice_to_ctm.sh --cmd "$decode_cmd" data/shadow.uem data/lang exp/sgmm5/decode_shadow.uem 
  #-local/lattice_to_ctm.sh --cmd "$decode_cmd" data/shadow.uem data/lang exp/sgmm5/decode_fmllr_shadow.uem 
  #-split_ctms exp/sgmm5/decode_shadow.uem data/dev data/test.uem
  #-split_ctms exp/sgmm5/decode_fmllr_shadow.uem data/dev data/test.uem
  #-local/shadow_set_kws_search.sh --cmd "$decode_cmd" data/shadow.uem data/lang exp/sgmm5/decode_shadow.uem data/dev data/test.uem
  #-local/shadow_set_kws_search.sh --cmd "$decode_cmd" data/shadow.uem data/lang exp/sgmm5/decode_fmllr_shadow.uem data/dev data/test.uem
fi

####################################################################
##
## SGMM2 decoding of test.uem
##
####################################################################
if $decode_test ; then
  #steps/decode_sgmm2.sh --skip-scoring true --nj 64 --cmd "$decode_cmd" --transform-dir exp/tri5/decode_test.uem \
  #  exp/sgmm5/graph data/test.uem exp/sgmm5/decode_test.uem  | tee  exp/sgmm5/decode_test.uem.log
  steps/decode_sgmm2.sh --skip-scoring true --use-fmllr true --nj 64 --cmd "$decode_cmd" --transform-dir exp/tri5/decode_test.uem \
    --num-threads 6 --parallel-opts "-pe smp 6 -l ram_free=0.5G" \
    exp/sgmm5/graph data/test.uem exp/sgmm5/decode_fmllr_test.uem  | tee  exp/sgmm5/decode_fmllr_test.uem.log

  #-local/lattice_to_ctm.sh --cmd "$decode_cmd" data/test.uem data/lang exp/sgmm5/decode_test.uem 
  #-local/lattice_to_ctm.sh --cmd "$decode_cmd" data/test.uem data/lang exp/sgmm5/decode_fmllr_test.uem 
  #-local/kws_search.sh --skip-scoring true --cmd "$decode_cmd" data/lang data/test.uem/ exp/sgmm5/decode_test.uem 
  #-local/kws_search.sh --skip-scoring true --cmd "$decode_cmd" data/lang data/test.uem/ exp/sgmm5/decode_fmllr_test.uem 
fi

####################################################################
##
## SGMM2 decoding of eval.uem
##
####################################################################
if $decode_eval ; then
  steps/decode_sgmm2.sh --skip-scoring true --use-fmllr true --nj 64 --cmd "$decode_cmd" --transform-dir exp/tri5/decode_eval.uem \
    --num-threads 6 --parallel-opts "-pe smp 6 -l ram_free=0.5G" \
    exp/sgmm5/graph data/eval.uem exp/sgmm5/decode_fmllr_eval.uem |tee  exp/sgmm5/decode_fmllr_eval.uem.log
  #-steps/decode_sgmm2.sh --skip-scoring true --nj 64 --cmd "$decode_cmd" --transform-dir exp/tri5/decode_eval.uem \
  #-  exp/sgmm5/graph data/eval.uem exp/sgmm5/decode_eval.uem |tee exp/sgmm5/decode_eval.uem.log
  #-local/lattice_to_ctm.sh --cmd "$decode_cmd" data/eval.uem data/lang exp/sgmm5/decode_eval.uem
  #-local/lattice_to_ctm.sh --cmd "$decode_cmd" data/eval.uem data/lang exp/sgmm5/decode_fmllr_eval.uem
  #-local/kws_search.sh --skip-scoring true --cmd "$decode_cmd" data/lang data/eval.uem/ exp/sgmm5/decode_eval.uem
  #-local/kws_search.sh --skip-scoring true --cmd "$decode_cmd" data/lang data/eval.uem/ exp/sgmm5/decode_fmllr_eval.uem
fi

####################################################################
##
## SGMM_MMI rescoring of the shadow.uem
##
####################################################################
if $decode_shadow ; then
  for iter in 1 2 3 4; do
    #steps/decode_sgmm2_rescore.sh --skip_scoring true \
    #  --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode_shadow.uem \
    #  data/lang data/shadow.uem exp/sgmm5/decode_shadow.uem exp/sgmm5_mmi_b0.1/decode_shadow.uem_it$iter
  
    steps/decode_sgmm2_rescore.sh --skip_scoring true \
      --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode_shadow.uem \
      data/lang data/shadow.uem exp/sgmm5/decode_fmllr_shadow.uem exp/sgmm5_mmi_b0.1/decode_fmllr_shadow.uem_it$iter
   
    (

      #local/lattice_to_ctm.sh --cmd "$decode_cmd" --model exp/sgmm5_mmi_b0.1/$iter.mdl \
      #  data/shadow.uem data/lang exp/sgmm5_mmi_b0.1/decode_shadow.uem_it$iter
      local/lattice_to_ctm.sh --cmd "$decode_cmd" \
        --model exp/sgmm5_mmi_b0.1/$iter.mdl --word-ins-penalty 0.5 \
        data/shadow.uem data/lang exp/sgmm5_mmi_b0.1/decode_fmllr_shadow.uem_it$iter
      
      #local/split_ctms.sh data/shadow.pem exp/sgmm5_mmi_b0.1/decode_shadow.uem_it$iter \
      #  data/dev data/test.uem
      local/split_ctms.sh data/shadow.uem exp/sgmm5_mmi_b0.1/decode_fmllr_shadow.uem_it$iter \
        data/dev data/test.uem
    ) &

    #local/shadow_set_kws_search.sh --cmd "$decode_cmd" --model exp/sgmm5_mmi_b0.1/$iter.mdl \
    #  data/shadow.uem data/lang exp/sgmm5_mmi_b0.1/decode_shadow.uem_it$iter 
    #  data/dev data/test.uem
    local/shadow_set_kws_search.sh --cmd "$decode_cmd" \
      --model exp/sgmm5_mmi_b0.1/$iter.mdl  --max-states 150000 \
      data/shadow.uem data/lang exp/sgmm5_mmi_b0.1/decode_fmllr_shadow.uem_it$iter \
      data/dev data/test.uem
  done
fi


####################################################################
##
## SGMM_MMI rescoring of the test.uem
##
####################################################################
if $decode_test ; then
  for iter in 1 2 3 4; do
    #steps/decode_sgmm2_rescore.sh --skip-scoring true\
    #    --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode_test.uem \
    #    data/lang data/test.uem exp/sgmm5/decode_test.uem exp/sgmm5_mmi_b0.1/decode_test.uem_it$iter

    steps/decode_sgmm2_rescore.sh --skip-scoring true \
      --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode_test.uem \
      data/lang data/test.uem exp/sgmm5/decode_fmllr_test.uem exp/sgmm5_mmi_b0.1/decode_fmllr_test.uem_it$iter
    
    #local/lattice_to_ctm.sh --cmd "$decode_cmd" --model exp/sgmm5_mmi_b0.1/$iter.mdl \
    #  data/test.uem data/lang exp/sgmm5_mmi_b0.1/decode_test.uem_it$iter &
    local/lattice_to_ctm.sh --cmd "$decode_cmd" \
      --model exp/sgmm5_mmi_b0.1/$iter.mdl --word-ins-penalty 0.5 \
      data/test.uem data/lang exp/sgmm5_mmi_b0.1/decode_fmllr_test.uem_it$iter &

    #local/kws_search.sh --skip-scoring true --cmd "$decode_cmd" --model exp/sgmm5_mmi_b0.1/$iter.mdl \
    #  data/lang data/test.uem/ exp/sgmm5_mmi_b0.1/decode_test.uem_it$iter
    local/kws_search.sh --skip-scoring true --cmd "$decode_cmd" \
      --model exp/sgmm5_mmi_b0.1/$iter.mdl --max-states 150000 \
      data/lang data/test.uem/ exp/sgmm5_mmi_b0.1/decode_fmllr_test.uem_it$iter
  done
fi

####################################################################
##
## SGMM_MMI rescoring of the eval.uem
##
####################################################################
if $decode_eval ; then
  for iter in 1 2 3 4; do
    #steps/decode_sgmm2_rescore.sh --skip-scoring true\
    #  --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode_eval.uem \
    #  data/lang data/eval.uem exp/sgmm5/decode_eval.uem exp/sgmm5_mmi_b0.1/decode_eval.uem_it$iter

    steps/decode_sgmm2_rescore.sh --skip-scoring true \
      --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode_eval.uem \
      data/lang data/eval.uem exp/sgmm5/decode_fmllr_eval.uem exp/sgmm5_mmi_b0.1/decode_fmllr_eval.uem_it$iter
      
    (
      #local/lattice_to_ctm.sh --cmd "$decode_cmd" --model exp/sgmm5_mmi_b0.1/$iter.mdl \
      #  data/eval.uem data/lang exp/sgmm5_mmi_b0.1/decode_eval.uem_it$iter &
      local/lattice_to_ctm.sh --cmd "$decode_cmd" \
        --model exp/sgmm5_mmi_b0.1/$iter.mdl  --word-ins-penalty 0.5 \
        data/eval.uem data/lang exp/sgmm5_mmi_b0.1/decode_fmllr_eval.uem_it$iter 

      local/score_scm.sh --cer $cer --cmd "$decode_cmd" data/eval.uem/ \
        data/lang exp/sgmm5_mmi_b0.1/decode_fmllr_eval.uem_it$iter
    ) &

    #local/kws_search.sh --cmd "$decode_cmd" --model exp/sgmm5_mmi_b0.1/$iter.mdl \
    #  data/lang data/eval.uem/ exp/sgmm5_mmi_b0.1/decode_eval.uem_it$iter
    local/kws_search.sh --cmd "$decode_cmd" \
      --model exp/sgmm5_mmi_b0.1/$iter.mdl --max-states 150000 \
      data/lang data/eval.uem/ exp/sgmm5_mmi_b0.1/decode_fmllr_eval.uem_it$iter
  done
fi

if $decode_nnet ; then
  decode_nj=64

  if $decode_eval ; then
    steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj $decode_nj \
      --num-threads 6 --parallel-opts "-pe smp 6 -l ram_free=0.5G" \
      --transform-dir exp/tri5/decode_eval.uem/ \
      exp/tri5/graph data/eval.uem exp/tri6_nnet/decode_eval.uem
    
    local/lattice_to_ctm.sh --cmd "$decode_cmd" \
      --word-ins-penalty 0.5 \
      data/eval.uem data/lang exp/tri6_nnet/decode_eval.uem 

    local/score_scm.sh --cmd "$decode_cmd" \
      data/eval.uem data/lang exp/tri6_nnet/decode_eval.uem
      
    local/kws_search.sh --cmd "$decode_cmd" \
      --max-states 150000 \
      data/lang data/eval.uem exp/tri6_nnet/decode_eval.uem
  fi

  if $decode_test ; then
    steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj $decode_nj \
      --num-threads 6 --parallel-opts "-pe smp 6 -l ram_free=0.5G" \
      --transform-dir exp/tri5/decode_test.uem/ \
      exp/tri5/graph data/test.uem exp/tri6_nnet/decode_test.uem
    
    local/lattice_to_ctm.sh --cmd "$decode_cmd" \
      --word-ins-penalty 0.5\
      data/test.uem data/lang exp/tri6_nnet/decode_test.uem 

    local/kws_search.sh --cmd "$decode_cmd" \
      --skip-scoring true --max-states 150000 \
      data/lang data/test.uem exp/tri6_nnet/decode_test.uem
  fi



  if $decode_shadow ; then
    steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj $decode_nj \
      --num-threads 6 --parallel-opts "-pe smp 6 -l ram_free=0.5G" \
      --transform-dir exp/tri5/decode_shadow.uem/ \
      exp/tri5/graph data/shadow.uem exp/tri6_nnet/decode_shadow.uem

    local/lattice_to_ctm.sh --cmd "$decode_cmd" \
      --word-ins-penalty 0.5 \
      data/shadow.uem data/lang exp/tri6_nnet/decode_shadow.uem 

    local/split_ctms.sh data/shadow.uem exp/tri6_nnet/decode_shadow.uem \
      data/dev data/test.uem

    local/shadow_set_kws_search.sh --cmd "$decode_cmd" \
      --max-states 150000 \
      data/shadow.uem data/lang exp/tri6_nnet/decode_shadow.uem data/dev data/test.uem
  fi
fi


echo "Done, waiting until the background tasks finish"
wait
echo "Everything looks fine"

