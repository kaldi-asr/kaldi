#!/bin/bash 
set -e
set -o pipefail

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;



. conf/parse_options.sh     

if [ $# -ne 0 ]; then
  echo "Usage: $(basename $0) [--type (dev10h|dev2h|eval15h|shadow)]"
echo

if [ "$type" != "dev10h" && "$type" != "dev2h" && "$type" != "eval15h" && "$type" != "shadow" ]; then
  echo "Error: invalid variable type=$type, valid values are dev10h|dev2h|eval|shadow"
fi

# Yenda: please check if this logic is right.  Not sure what to do about shadow, and about
# kws scoring versus CTM scoring.
# Also, I might not be passing this variable in everywhere I need to.
if [[ "$type" == "dev2h" || "$type" == "dev10h" ]]; then
  skip_scoring=false
else
  skip_scoring=true
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
if [[ $type == shadow || $type == eval15h ]]; then
  if [ ! -f data/eval15h.uem/.done ]; then
    local/cmu_uem2kaldi_dir.sh --filelist $eval15h_data_list $eval15h_data_cmudb_shadow  $eval15h_data_dir data/eval15h.uem 
    steps/make_plp.sh --cmd "$train_cmd" --nj $decode_nj data/eval15h.uem exp/make_plp/eval15h.uem plp

    steps/compute_cmvn_stats.sh data/eval15h.uem exp/make_plp/eval15h.uem plp
    utils/fix_data_dir.sh data/eval15h.uem
    touch data/eval15h.uem/.done
  fi

  # we expect that the dev2h directory will already exist.
  if [ ! -f data/shadow.uem/.done ]; then
    if [ ! -f data/dev2h/.done ]; then
      echo "Error: data/dev2h/.done does not exist."
    fi

    local/kws_setup.sh --case-insensitive $case_insensitive $eval15h_data_ecf $eval15h_data_kwlist data/lang data/eval15h.uem/

    local/create_shadow_dataset.sh data/shadow.uem data/dev2h data/eval15h.uem
    local/kws_data_prep.sh --case-insensitive $case_insensitive data/lang data/shadow.uem data/shadow.uem/kws || exit 1
    utils/fix_data_dir.sh data/shadow.uem
    touch data/shadow.uem/.done 
  fi
fi

#####################################################################
#
# dev10h.uem directory preparation
#
#####################################################################
if [ $type == dev10h ]; then
  if [ ! -f data/dev10h.uem/.done ]; then
    local/cmu_uem2kaldi_dir.sh --filelist $dev10h_data_list $dev10h_data_cmudb_shadow  $dev10h_data_dir data/dev10h.uem
    cp data/eval.pem/stm data/dev10h.uem
    cp data/eval.pem/glm data/dev10h.uem
    if [ $cer == 1 ] ; then
      [ ! -f data/eval.pem/char.stm ] && echo "CER=1 and file eval.pem/char.stm not present" && exit 1
      cp data/eval.pem/char.stm data/dev10h.uem
    fi
    steps/make_plp.sh --cmd "$train_cmd" --nj $decode_nj data/dev10h.uem exp/make_plp/dev10h.uem plp
    steps/compute_cmvn_stats.sh data/dev10h.uem exp/make_plp/dev10h.uem plp
    utils/fix_data_dir.sh data/dev10h.uem

    local/kws_setup.sh --case-insensitive $case_insensitive $dev10h_ecf_file $dev10h_data_kwlist $dev10h_rttm_file data/lang data/dev10h.uem
    touch data/dev10h.uem/.done
  fi
fi

fi

####################################################################
##
## FMLLR decoding 
##
####################################################################



if [ ! -f exp/tri5/decode_${type}.uem/.done ]; then
  steps/decode_fmllr_extra.sh --skip-scoring "$skip_scoring" --nj 64 --cmd "$decode_cmd" \
    --num-threads 6 --parallel-opts "-pe smp 6 -l ram_free=0.5G" \
    exp/tri5/graph data/${type}.uem exp/tri5/decode_${type}.uem  | tee  exp/tri5/decode_${type}.uem.log
  touch exp/tri5/decode_${type}.uem/.done
fi


if false; then # Disable the scoring for now...
  if [ ! -f exp/tri5/decode_${type}.uem/.kws.done ]; then
    local/lattice_to_ctm.sh --cmd "$decode_cmd" data/${type}.uem data/lang exp/tri5/decode_${type}.uem.si
    local/lattice_to_ctm.sh --cmd "$decode_cmd" data/${type}.uem data/lang exp/tri5/decode_${type}.uem
    if [ $type == shadow ]; then
      split_ctms exp/tri5/decode_shadow.uem.si data/dev2h data/eval15h.uem
      split_ctms exp/tri5/decode_shadow.uem data/dev2h data/eval15h.uem
      
      local/shadow_set_kws_search.sh --cmd "$decode_cmd" data/${type}.uem data/lang \
        exp/tri/decode_${type}.uem.si data/dev2h data/eval15h.uem
      local/shadow_set_kws_search.sh --cmd "$decode_cmd" data/${type}.uem data/lang \
        exp/tri5/decode_${type}.uem data/dev2h data/eval15h.uem
    else
      local/kws_search.sh --skip-scoring "$skip_scoring" --cmd "$decode_cmd" data/lang \
        data/${type}.uem/ exp/tri5/decode_${type}.uem.si
      local/kws_search.sh --skip-scoring "$skip_scoring" --cmd "$decode_cmd" data/lang \
        data/${type}.uem/ exp/tri5/decode_${type}.uem
    fi
    touch exp/tri5/decode_${type}.uem/.kws.done
  fi
fi

####################################################################
## SGMM2 decoding 
####################################################################
if [ ! -f exp/sgmm5/decode_fmllr_${type}.uem/.done ]; then
  steps/decode_sgmm2.sh --skip-scoring "$skip_scoring" --use-fmllr true --nj 64 \
    --cmd "$decode_cmd" --transform-dir exp/tri5/decode_${type}.uem \
    --num-threads 6 --parallel-opts "-pe smp 6 -l ram_free=0.5G" \
    exp/sgmm5/graph data/${type}.uem exp/sgmm5/decode_fmllr_${type}.uem | tee exp/sgmm5/decode_fmllr_${type}.uem.log
fi  

if [ ! -f exp/sgmm5/decode_fmllr_${type}.uem/.kws.done ]; then
  local/lattice_to_ctm.sh --cmd "$decode_cmd" data/shadow.uem data/lang exp/sgmm5/decode_fmllr_shadow.uem 
  if [ $type == shadow ]; then
    split_ctms exp/sgmm5/decode_fmllr_shadow.uem data/dev2h data/eval15h.uem
    local/shadow_set_kws_search.sh --cmd "$decode_cmd" data/shadow.uem data/lang \
      exp/sgmm5/decode_fmllr_shadow.uem data/dev2h data/eval15h.uem
  else
    local/kws_search.sh --skip-scoring "$skip_scoring" --cmd "$decode_cmd" data/lang \
      data/${type}.uem/ exp/sgmm5/decode_fmllr_${type}.uem
  fi
  touch exp/sgmm5/decode_fmllr_${type}.uem/.kws.done
fi

####################################################################
##
## SGMM_MMI rescoring and KWS
##
####################################################################

for iter in 1 2 3 4; do
  # Decode SGMM+MMI (via rescoring).
  if [ ! -f exp/sgmm5_mmi_b0.1/decode_fmllr_${type}.uem_it$iter/.done ]; then

    steps/decode_sgmm2_rescore.sh  --skip-scoring "$skip_scoring" \
      --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode_${type}.uem \
      data/lang data/${type}.uem exp/sgmm5/decode_fmllr_${type}.uem exp/sgmm5_mmi_b0.1/decode_fmllr_${type}.uem_it$iter
   
    touch exp/sgmm5_mmi_b0.1/decode_fmllr_${type}.uem_it$iter/.done
  fi
  if [ ! exp/sgmm5_mmi_b0.1/decode_fmllr_${type}.uem_it$iter/.kws.done ]; then
    local/lattice_to_ctm.sh --cmd "$decode_cmd" --word-ins-penalty 0.5 \
      data/${type}.uem data/lang exp/sgmm5_mmi_b0.1/decode_fmllr_${type}.uem_it$iter

    if [ $type == shadow ]; then
      local/split_ctms.sh data/shadow.uem exp/sgmm5_mmi_b0.1/decode_fmllr_shadow.uem_it$iter \
        data/dev2h data/eval15h.uem
      local/shadow_set_kws_search.sh --cmd "$decode_cmd" --max-states 150000 \
        data/shadow.uem data/lang exp/sgmm5_mmi_b0.1/decode_fmllr_shadow.uem_it$iter \
        data/dev2h data/eval15h.uem
    else
      local/kws_search.sh --skip-scoring "$skip_scoring" --cmd "$decode_cmd" data/lang \
        data/${type}.uem/ exp/sgmm5_mmi_b0.1/decode_fmllr_${type}.uem_it$iter
    fi
    touch exp/sgmm5_mmi_b0.1/decode_fmllr_${type}.uem_it$iter/.kws.done
  fi
done


if [[ ! -f exp/tri6_nnet/decode_${type}.uem/.done && -f exp/tri6_nnet/final.mdl ]]; then

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj $decode_nj \
    --skip-scoring "$skip_scoring" \
    --num-threads 6 --parallel-opts "-pe smp 6 -l ram_free=0.5G" \
    --transform-dir exp/tri5/decode_${type}.uem/ \
    exp/tri5/graph data/${type}.uem exp/tri6_nnet/decode_${type}.uem
    
  touch exp/tri6_nnet/decode_${type}.uem/.done
fi

if [[ ! -f exp/tri6_nnet/decode_${type}.uem/.kws.done && -f exp/tri6_nnet/final.mdl ]]; then

  local/lattice_to_ctm.sh --cmd "$decode_cmd" --word-ins-penalty 0.5 \
    data/${type}.uem data/lang exp/tri6_nnet/decode_${type}.uem 

  if [ "$type" != eval15h ]; then
    local/score_stm.sh --cmd "$decode_cmd" \
      data/${type}.uem data/lang exp/tri6_nnet/decode_${type}.uem
  fi

  if [ $type == shadow ]; then
    local/split_ctms.sh data/shadow.uem exp/tri6_nnet/decode_${type}.uem \
      data/dev2h data/eval15h.uem
    local/shadow_set_kws_search.sh --cmd "$decode_cmd" --max-states 150000 \
      data/shadow.uem data/lang exp/tri6_nnet/decode_${type}.uem \
      data/dev2h data/eval15h.uem
  else
    local/kws_search.sh --skip-scoring "$skip_scoring" --cmd "$decode_cmd" data/lang \
      data/${type}.uem/ exp/tri6_nnet/decode_${type}.uem
  fi
  touch exp/tri6_nnet/decode_${type}.uem/.kws.done
fi


echo "Everything looks fine"
