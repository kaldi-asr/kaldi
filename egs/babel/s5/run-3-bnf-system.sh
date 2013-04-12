#!/bin/bash -v

# This script builds the SGMM system on top of the bottleneck features.
# It comes after run-BNF.sh.
# you might want to make the director ./plp_processed somewhere fast with a lot of space before
# running the script.

. conf/common_vars.sh
. ./lang.conf
[ -f local.conf ] && . ./local.conf

set -e 
set -o pipefail
set -u

sopt="--min-lmwt 25 --max-lmwt 40"
sopt2="--min-lmwt 15 --max-lmwt 30"

#In the cloud I did this:
#tmpdir=`pwd | sed s:/home/:/export/tmp7/:`
#mkdir -p $tmpdir/plp_processed
#ln -s $tmpdir/plp_processed .


if [ ! -f data/bnf_train/.done ]; then
  steps_BNF/make_bnf_feat.sh --stage 0 --nj $train_nj \
    --cmd "$train_cmd" --transform_dir exp/tri5_ali \
    data/train data/bnf_train exp_BNF/bnf_dnn exp/tri5_ali exp_BNF/make_bnf 
  touch data/bnf_train/.done
fi

if [ ! -f data/bnf_dev2h/.done ]; then
  steps_BNF/make_bnf_feat.sh --stage 0 --nj $decode_nj \
    --cmd "$train_cmd" --transform_dir exp/tri5/decode_dev2h/ \
    data/dev2h/ data/bnf_dev2h exp_BNF/bnf_dnn exp/tri5_ali exp_BNF/make_bnf 
  touch data/bnf_dev2h/.done
fi

if [ ! -f data/sat_train/.done ]; then
  steps/make_fmllr_feats.sh --cmd "$train_cmd -tc 10" \
    --nj $train_nj --transform-dir exp/tri5_ali  data/sat_train data/train \
    exp/tri5_ali exp_BNF/make_fmllr_feats/log plp_processed/ 
  touch data/sat_train/.done
fi

if [ ! -f data/sat_dev2h/.done ]; then
  steps/make_fmllr_feats.sh --cmd "$train_cmd -tc 10" \
    --nj $decode_nj --transform-dir exp/tri5/decode_dev2h  \
    data/sat_dev2h data/dev2h \
    exp/tri5 exp_BNF/make_fmllr_feats/log plp_processed/ 
  touch data/sat_dev2h/.done
fi

if [ ! -f data/app_train/.done ]; then
  steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
    data/bnf_train data/sat_train data/app_train exp_BNF/append_feats/log plp_processed/ 
  steps/compute_cmvn_stats.sh --fake \
    data/app_train exp/make_plp/app_train plp_processed/ 
  touch data/app_train/.done
fi

if [ ! -f data/app_dev2h/.done ]; then
  steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
    data/bnf_dev2h data/sat_dev2h data/app_dev2h exp_BNF/append_feats/log plp_processed/  
  steps/compute_cmvn_stats.sh --fake \
    data/app_dev2h exp/make_plp/app_dev2h plp_processed/ 
  ln -s `pwd`/data/dev2h/kws data/app_dev2h/kws
  touch data/app_dev2h/.done
fi


decode_lda_mllt() { 
  dir=$1
  if [ ! -f $dir/decode_dev2h/.done ]; then
    mkdir -p $dir/graph
    utils/mkgraph.sh \
      data/lang $dir $dir/graph &> $dir/mkgraph.log
    mkdir -p $dir/decode_dev2h
    
    steps/decode.sh --nj $decode_nj --acwt 0.0333 --scoring-opts "$sopt" --cmd "$decode_cmd" \
      $dir/graph data/app_dev2h $dir/decode_dev2h &> $dir/decode_dev2h.log 
  fi
  if [ ! -f $dir/decode_dev2h/kws/.done ];then
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
      data/lang data/app_dev2h $dir/decode_dev2h 
    touch $dir/decode_dev2h/kws/.done
  fi
}


if [ ! -f exp/tri5_ali_10/.done ]; then
  steps/align_fmllr.sh --boost-silence 1.5 --nj 10 --cmd "$train_cmd" \
    data/train data/lang exp/tri5_ali exp/tri5_ali_10  
  touch exp/tri5_ali_10/.done
fi


if [ ! -f exp_BNF/tri5/.done ]; then
  steps/train_lda_mllt.sh --splice-opts "--left-context=1 --right-context=1" \
    --dim 60 --boost-silence 1.5 --cmd "$train_cmd" \
    $numLeavesMLLT $numGaussMLLT data/app_train data/lang exp/tri5_ali_10 exp_BNF/tri5 ;
  touch exp_BNF/tri5/.done
fi

decode_lda_mllt exp_BNF/tri5  &


if [ ! -f exp_BNF/tri5ali_20/.done ]; then
  steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
    data/app_train data/lang exp_BNF/tri5 exp_BNF/tri5ali_20  
  touch exp_BNF/tri5ali_20/.done
fi

if [ ! -f exp_BNF/tri6/.done ]; then
  steps/train_sat.sh --boost-silence 1.5 --cmd "$train_cmd" \
    $numLeavesSAT $numGaussSAT data/app_train data/lang exp_BNF/tri5ali_20 exp_BNF/tri6 
  touch exp_BNF/tri6/.done
fi

( 
  echo "Spawning decoding job for tri6"
  dir=exp_BNF/tri6
  mkdir -p $dir/graph
  utils/mkgraph.sh \
    data/lang $dir $dir/graph &> $dir/mkgraph.log
  mkdir -p $dir/decode_dev2h
  if [ ! -f $dir/decode_dev2h/.done ]; then
    exit 1
    steps/decode_fmllr.sh --nj $decode_nj --acwt 0.0333 --scoring-opts "$sopt" \
      --cmd "$train_cmd" "${decode_extra_opts[@]}" \
      $dir/graph data/app_dev2h $dir/decode_dev2h &> $dir/decode_dev2h.log ;
    touch $dir/decode_dev2h/.done
  fi
  if [ ! -f $dir/decode_dev2h/kws/.done ];then
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
      data/lang data/app_dev2h $dir/decode_dev2h
    touch $dir/decode_dev2h/kws/.done
  fi
) || exit 1


echo ---------------------------------------------------------------------
echo "Starting exp_BNF/ubm7 on" `date`
echo ---------------------------------------------------------------------
[ -f exp_BNF/tri6_ali/trans.1 ] && touch exp_BNF/tri6_ali/.done ## TEMP, remove this line
if [ ! -f exp_BNF/tri6_ali/.done ]; then
  steps/align_fmllr.sh \
    --boost-silence 1.5 --nj $train_nj --cmd "$train_cmd" \
    data/app_train data/lang exp_BNF/tri6 exp_BNF/tri6_ali 
  touch exp_BNF/tri6_ali/.done
fi

[ -f exp_BNF/ubm7/final.ubm ] && touch exp_BNF/ubm7/.done ## TEMP, remove this line
if [ ! -f exp_BNF/ubm7/.done ]; then
  steps/train_ubm.sh --cmd "$train_cmd" \
    $numGaussUBM data/app_train data/lang exp_BNF/tri6_ali exp_BNF/ubm7 
  touch exp_BNF/ubm7/.done
fi

[ -f exp_BNF/sgmm7/final.alimdl ] && touch exp_BNF/sgmm7/.done ## TEMP, remove this line
if [ ! -f exp_BNF/sgmm7/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp_BNF/sgmm7 on" `date`
  echo ---------------------------------------------------------------------
  steps/train_sgmm2_group.sh \
    --cmd "$train_cmd" "${sgmm_group_extra_opts[@]}"\
    $numLeavesSGMM $numGaussSGMM data/app_train data/lang \
    exp_BNF/tri6_ali exp_BNF/ubm7/final.ubm exp_BNF/sgmm7 
  touch exp_BNF/sgmm7/.done
fi

################################################################################
# Ready to decode with SGMM2 models
################################################################################
echo "Waiting for finishing tri6 decode"
echo ---------------------------------------------------------------------
echo "Spawning exp_BNF/sgmm7/decode_dev2h[_fmllr] on" `date`
echo ---------------------------------------------------------------------
(
    mkdir -p exp_BNF/sgmm7/graph
    utils/mkgraph.sh \
        data/lang exp_BNF/sgmm7 exp_BNF/sgmm7/graph &> exp_BNF/sgmm7/mkgraph.log

    mkdir -p exp_BNF/sgmm7/decode_fmllr_dev2h
    
    if [ ! -f exp_BNF/sgmm7/decode_fmllr_dev2h/.done ]; then
      steps/decode_sgmm2.sh --use-fmllr true --nj $decode_nj \
        --transform-dir exp_BNF/tri6/decode_dev2h \
        --cmd "$train_cmd" "${decode_extra_opts[@]}"\
        --acwt 0.05 --scoring-opts "$sopt2" \
        exp_BNF/sgmm7/graph data/app_dev2h exp_BNF/sgmm7/decode_fmllr_dev2h 
      touch exp_BNF/sgmm7/decode_fmllr_dev2h/.done;
    fi
    
    if [ ! -f exp_BNF/sgmm7/decode_fmllr_dev2h/kws/.done ]; then
      local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/app_dev2h exp_BNF/sgmm7/decode_fmllr_dev2h
      touch exp_BNF/sgmm7/decode_fmllr_dev2h/kws/.done
    fi
) 


if [ ! -f exp_BNF/sgmm7_ali/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp_BNF/sgmm7_ali on" `date`
  echo ---------------------------------------------------------------------
  steps/align_sgmm2.sh \
     --nj $train_nj --cmd "$train_cmd" --transform-dir exp_BNF/tri6_ali --use-graphs true \
    data/app_train data/lang exp_BNF/sgmm7 exp_BNF/sgmm7_ali 
  touch exp_BNF/sgmm7_ali/.done
fi

if [ ! -f exp_BNF/sgmm7_denlats/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp_BNF/sgmm5_denlats on" `date`
  echo ---------------------------------------------------------------------
  steps/make_denlats_sgmm2.sh \
    --nj $train_nj --sub-split $train_nj "${sgmm_denlats_extra_opts[@]}"
    --transform-dir exp_BNF/tri6_ali --beam 10.0 --acwt 0.06 --lattice-beam 6 \
     data/app_train data/lang exp_BNF/sgmm7_ali exp_BNF/sgmm7_denlats 
  touch exp_BNF/sgmm7_denlats/.done
fi


if [ ! -f exp_BNF/sgmm7_mmi_b0.1/.done ]; then
  steps/train_mmi_sgmm2.sh \
    --cmd "$train_cmd" --acwt 0.06 \
    --transform-dir exp_BNF/tri6_ali --boost 0.1 --zero-if-disjoint true \
    data/app_train data/lang exp_BNF/sgmm7_ali exp_BNF/sgmm7_denlats \
    exp_BNF/sgmm7_mmi_b0.1 
  touch exp_BNF/sgmm7_mmi_b0.1/.done;
fi

for iter in 1 2 3 4; do
  if [ ! -f exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_dev2h_it$iter/.done ]; then
    steps/decode_sgmm2_rescore.sh --scoring-opts "$sopt2" \
        --cmd "$decode_cmd" --iter $iter --transform-dir exp_BNF/tri6/decode_dev2h \
        data/lang data/app_dev2h exp_BNF/sgmm7/decode_fmllr_dev2h exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_dev2h_it$iter 
    touch exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_dev2h_it$iter/.done 
  fi
  if [ ! -f exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_dev2h_it$iter/kws/.done ]; then
    local/kws_search.sh $sopt2 --cmd "$decode_cmd" --duptime $duptime \
      data/lang data/app_dev2h exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_dev2h_it$iter
    touch exp_BNF/sgmm7/decode_fmllr_dev2h/kws/.done
    touch exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_dev2h_it$iter/kws/.done 
  fi
done


echo ---------------------------------------------------------------------
echo "Finished successfully on" `date`
echo ---------------------------------------------------------------------

exit 0
