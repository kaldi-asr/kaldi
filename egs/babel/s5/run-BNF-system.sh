#!/bin/bash -v

# This script builds the SGMM system on top of the bottleneck features.
# It comes after run-BNF.sh.

. conf/common_vars.sh
. ./lang.conf
sopt="--min-lmwt 25 --max-lmwt 40"
sopt2="--min-lmwt 15 --max-lmwt 30"
tmpdir=`pwd | sed s:/home/:/export/tmp7/:`
mkdir -p $tmpdir/plp_processed
ln -s $tmpdir/plp_processed .
if [ -d exp/tri5 ]; then
  srcdir=exp/tri5 # fullLP
else
  srcdir=exp/tri4
fi

if [ ! -f data/train_bnf/.done ]; then
  steps_BNF/make_bnf_feat.sh --stage 0 --nj $train_nj --cmd "$train_cmd" --transform_dir ${srcdir}_ali \
    data/train data/train_bnf exp_BNF/bnf_dnn ${srcdir}_ali exp_BNF/make_bnf || exit 1
  touch data/train_bnf/.done
fi

if [ ! -f data/dev_bnf/.done ]; then
  steps_BNF/make_bnf_feat.sh --stage 0 --nj $decode_nj --cmd "$train_cmd" --transform_dir ${srcdir}/decode \
    data/dev data/dev_bnf exp_BNF/bnf_dnn ${srcdir}_ali exp_BNF/make_bnf || exit 1
  touch data/dev_bnf/.done
fi

if [ ! -f data/train_sat/.done ]; then
  steps/make_fmllr_feats.sh --cmd "$train_cmd -tc 10" \
    --nj $train_nj --transform-dir $srcdir  data/train_sat data/train \
    $srcdir exp_BNF/make_fmllr_feats/log plp_processed/ || exit 1;
  touch data/train_sat/.done
fi

if [ ! -f data/dev_sat/.done ]; then
  steps/make_fmllr_feats.sh --cmd "$train_cmd -tc 10" \
    --nj $decode_nj --transform-dir $srcdir/decode/  data/dev_sat data/dev \
    $srcdir exp_BNF/make_fmllr_feats/log plp_processed/ || exit 1
  touch data/dev_sat/.done
fi

if [ ! -f data/train_app/.done ]; then
  steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
    data/train_bnf data/train_sat data/train_app exp_BNF/append_feats/log plp_processed/ || exit 1
  steps/compute_cmvn_stats.sh --fake \
    data/train_app exp/make_plp/train_app plp_processed/ || exit 1
  touch data/train_app/.done
fi

if [ ! -f data/dev_app/.done ]; then
  steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
    data/dev_bnf data/dev_sat data/dev_app exp_BNF/append_feats/log plp_processed/  || exit 1
  steps/compute_cmvn_stats.sh --fake \
    data/dev_app exp/make_plp/dev_app plp_processed/ || exit 1
  ln -s `pwd`/data/dev/kws data/dev_app/kws
  touch data/dev_app/.done
fi


decode_lda_mllt() { 
  dir=$1
  if [ ! -f $dir/decode/.done ]; then
    mkdir -p $dir/graph
    utils/mkgraph.sh \
      data/lang $dir $dir/graph &> $dir/mkgraph.log
    mkdir -p $dir/decode
    
    steps/decode.sh --nj $decode_nj --acwt 0.0333 --scoring-opts "$sopt" --cmd "$decode_cmd" \
      $dir/graph data/dev_app $dir/decode &> $dir/decode.log || exit 1;
  fi
  if [ ! -f $dir/decode/kws/.done ];then
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
      data/lang data/dev_app $dir/decode || exit 1;
    touch $dir/decode/kws/.done
  fi
}


if [ ! -f ${srcdir}_ali_10/.done ]; then
  steps/align_fmllr.sh --boost-silence 1.5 --nj 10 --cmd "$train_cmd" \
    data/train data/lang $srcdir ${srcdir}_ali_10  || exit 1
  touch ${srcdir}_ali_10/.done
fi


if [ ! -f exp_BNF/tri5/.done ]; then
  steps/train_lda_mllt.sh --splice-opts "--left-context=1 --right-context=1" \
    --dim 60 --boost-silence 1.5 --cmd "$train_cmd" \
    $numLeavesMLLT $numGaussMLLT data/train_app data/lang ${srcdir}_ali_10 exp_BNF/tri5 || exit 1;
  touch exp_BNF/tri5/.done
fi

decode_lda_mllt exp_BNF/tri5  &


if [ ! -f exp_BNF/tri5ali_20/.done ]; then
  steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
    data/train_app data/lang exp_BNF/tri5 exp_BNF/tri5ali_20  || exit 1;
  touch exp_BNF/tri5ali_20/.done
fi

if [ ! -f exp_BNF/tri6/.done ]; then
  steps/train_sat.sh --boost-silence 1.5 --cmd "$train_cmd" \
    $numLeavesSAT $numGaussSAT data/train_app data/lang exp_BNF/tri5ali_20 exp_BNF/tri6 || exit 1;
  touch exp_BNF/tri6/.done
fi

( 
  echo "Spawning decoding job for tri6"
  dir=exp_BNF/tri6
  mkdir -p $dir/graph
  utils/mkgraph.sh \
    data/lang $dir $dir/graph &> $dir/mkgraph.log
  mkdir -p $dir/decode
  if [ ! -f $dir/decode/.done ]; then
    steps/decode_fmllr.sh --nj $decode_nj --acwt 0.0333 --scoring-opts "$sopt" \
      --cmd "queue.pl -l mem_free=2G,ram_free=0.5G" --num-threads 6 --parallel-opts "-pe smp 6" \
      $dir/graph data/dev_app $dir/decode &> $dir/decode.log || exit 1;
    touch $dir/decode/.done
  fi
  if [ ! -f $dir/decode/kws/.done ];then
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
      data/lang data/dev_app $dir/decode
    touch $dir/decode/kws/.done
  fi
) &


echo ---------------------------------------------------------------------
echo "Starting exp_BNF/ubm7 on" `date`
echo ---------------------------------------------------------------------
[ -f exp_BNF/tri6_ali/trans.1 ] && touch exp_BNF/tri6_ali/.done ## TEMP, remove this line
if [ ! -f exp_BNF/tri6_ali/.done ]; then
  steps/align_fmllr.sh \
    --boost-silence 1.5 --nj 30 --cmd "$train_cmd" \
    data/train_app data/lang exp_BNF/tri6 exp_BNF/tri6_ali || exit 1
  touch exp_BNF/tri6_ali/.done
fi

[ -f exp_BNF/ubm7/final.ubm ] && touch exp_BNF/ubm7/.done ## TEMP, remove this line
if [ ! -f exp_BNF/ubm7/.done ]; then
  steps/train_ubm.sh --cmd "$train_cmd" \
    $numGaussUBM data/train_app data/lang exp_BNF/tri6_ali exp_BNF/ubm7 || exit 1
  touch exp_BNF/ubm7/.done
fi

[ -f exp_BNF/sgmm7/final.alimdl ] && touch exp_BNF/sgmm7/.done ## TEMP, remove this line
if [ ! -f exp_BNF/sgmm7/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp_BNF/sgmm7 on" `date`
  echo ---------------------------------------------------------------------
  steps/train_sgmm2_group.sh --cmd "queue.pl -l mem_free=2.0G,ram_free=2.0G" \
    --parallel-opts "-pe smp 3 -l mem_free=6.0G,ram_free=2.0G" --group 3 \
    $numLeavesSGMM $numGaussSGMM data/train_app data/lang \
    exp_BNF/tri6_ali exp_BNF/ubm7/final.ubm exp_BNF/sgmm7 || exit 1
  touch exp_BNF/sgmm7/.done
fi

################################################################################
# Ready to decode with SGMM2 models
################################################################################

echo ---------------------------------------------------------------------
echo "Spawning exp_BNF/sgmm7/decode[_fmllr] on" `date`
echo ---------------------------------------------------------------------
(
    mkdir -p exp_BNF/sgmm7/graph
    utils/mkgraph.sh \
        data/lang exp_BNF/sgmm7 exp_BNF/sgmm7/graph &> exp_BNF/sgmm7/mkgraph.log

    mkdir -p exp_BNF/sgmm7/decode_fmllr
    
    if [ ! -f exp_BNF/sgmm7/decode_fmllr/.done ]; then
      steps/decode_sgmm2.sh --use-fmllr true --nj $decode_nj \
        --transform-dir exp_BNF/tri6/decode \
        --cmd "queue.pl -l mem_free=3G,ram_free=3G" --num-threads 6 \
        --parallel-opts "-pe smp 6 -l mem_free=3G,ram_free=0.6G" \
        --acwt 0.05 --scoring-opts "$sopt2" \
        exp_BNF/sgmm7/graph data/dev_app exp_BNF/sgmm7/decode_fmllr || exit 1;
      touch exp_BNF/sgmm7/decode_fmllr/.done;
    fi
    
    if [ ! -f exp_BNF/sgmm7/decode_fmllr/kws/.done ]; then
      local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev_app exp_BNF/sgmm7/decode_fmllr
      touch exp_BNF/sgmm7/decode_fmllr/kws/.done
    fi
) 


if [ ! -f exp_BNF/sgmm7_ali/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp_BNF/sgmm7_ali on" `date`
  echo ---------------------------------------------------------------------
  steps/align_sgmm2.sh \
     --nj 30 --cmd "$train_cmd" --transform-dir exp_BNF/tri6_ali --use-graphs true \
    data/train_app data/lang exp_BNF/sgmm7 exp_BNF/sgmm7_ali || exit 1
  touch exp_BNF/sgmm7_ali/.done
fi

if [ ! -f exp_BNF/sgmm7_denlats/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp_BNF/sgmm5_denlats on" `date`
  echo ---------------------------------------------------------------------
  steps/make_denlats_sgmm2.sh \
    --nj 30 --sub-split 30 \
    --num-threads 4 --parallel-opts "-pe smp 4" --cmd "queue.pl -l mem_free=2G,ram_free=0.8G" \
    --transform-dir exp_BNF/tri6_ali --beam 10.0 --acwt 0.06 --lattice-beam 6 \
     data/train_app data/lang exp_BNF/sgmm7_ali exp_BNF/sgmm7_denlats || exit 1
  touch exp_BNF/sgmm7_denlats/.done
fi


if [ ! -f exp_BNF/sgmm7_mmi_b0.1/.done ]; then
  steps/train_mmi_sgmm2.sh \
    --cmd "queue.pl -l mem_free=5G,ram_free=6.5G" --acwt 0.06 \
    --transform-dir exp_BNF/tri6_ali --boost 0.1 --zero-if-disjoint true \
    data/train_app data/lang exp_BNF/sgmm7_ali exp_BNF/sgmm7_denlats \
    exp_BNF/sgmm7_mmi_b0.1 || exit 1
  touch exp_BNF/sgmm7_mmi_b0.1/.done;
fi

for iter in 1 2 3 4; do
  if [ ! -f exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_it$iter/.done ]; then
    steps/decode_sgmm2_rescore.sh --scoring-opts "$sopt2" \
        --cmd "$decode_cmd" --iter $iter --transform-dir exp_BNF/tri6/decode \
        data/lang data/dev_app exp_BNF/sgmm7/decode_fmllr exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_it$iter || exit 1;
    touch exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_it$iter/.done 
  fi
  if [ ! -f exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_it$iter/kws/.done ]; then
    local/kws_search.sh $sopt2 --cmd "$decode_cmd" --duptime $duptime \
      data/lang data/dev_app exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_it$iter
    touch exp_BNF/sgmm7/decode_fmllr/kws/.done
    touch exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_it$iter/kws/.done 
  fi
done



# HERE.


echo ---------------------------------------------------------------------
echo "Finished successfully on" `date`
echo ---------------------------------------------------------------------

exit 0
