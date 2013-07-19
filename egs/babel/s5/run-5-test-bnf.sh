#!/bin/bash 
set -e
set -o pipefail
set -u
set -v

cer=0
type=shadow # "type" may be "dev10h", "dev2h", or "eval", or "shadow" which means eval + dev2h. 
dev2shadow=dev10h.uem
sopt="--min-lmwt 25 --max-lmwt 40"
sopt2="--min-lmwt 15 --max-lmwt 30"

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

. utils/parse_options.sh     

if [ $# -ne 0 ]; then
  echo "Usage: $(basename $0) [options]"
fi

tmpdir=`pwd`

[ ! -d data/${type}.uem ] && echo "No such directory data/${type}.uem" && exit 1;
[ ! -d exp/tri5/decode_${type}.uem ] && echo "No such directory exp/tri5/decode_${type}.uem" && exit 1;

# Set decode_nj; typically 64.
decode_nj=`cat exp/tri5/decode_${type}.uem/num_jobs` || exit 1;


if [ ! -f data/bnf_$type.uem/.done ]; then
  [ ! -d data/bnf_$type.uem ] && mkdir -p $tmpdir/data/bnf_$type.uem && ln -s $tmpdir/data/bnf_$type.uem data/bnf_$type.uem
  steps_BNF/make_bnf_feat.sh --stage 0 --nj $decode_nj --cmd "$train_cmd" --transform_dir exp/tri5/decode_${type}.uem/ \
    data/${type}.uem data/bnf_$type.uem exp_BNF/bnf_dnn exp/tri5_ali exp_BNF/make_bnf_${type} || exit 1
  touch data/bnf_$type.uem/.done
fi

if [ ! -f data/sat_$type.uem/.done ]; then
  [ ! -d data/sat_$type.uem ] && mkdir -p $tmpdir/data/sat_$type.uem && ln -s $tmpdir/data/sat_$type.uem data/sat_$type.uem
  steps/make_fmllr_feats.sh --cmd "$train_cmd -tc 10" --nj $decode_nj --transform-dir exp/tri5/decode_${type}.uem  \
    data/sat_$type.uem data/${type}.uem exp/tri5 exp_BNF/make_fmllr_feats_${type}/log plp_processed || exit 1
  touch data/sat_$type.uem/.done
fi

if [ ! -f data/app_$type.uem/.done ]; then
  [ ! -d data/app_$type.uem ] && mkdir -p $tmpdir/data/app_$type.uem && ln -s $tmpdir/data/app_$type.uem data/app_$type.uem
  steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
    data/bnf_$type.uem data/sat_$type.uem data/app_$type.uem exp_BNF/append_feats/log plp_processed/  || exit 1
  steps/compute_cmvn_stats.sh --fake \
    data/app_$type.uem exp/make_plp/app_$type.uem plp_processed/ || exit 1
  ln -s `pwd`/data/${type}.uem/kws data/app_$type.uem/kws
  touch data/app_$type.uem/.done
fi


decode_lda_mllt() { 
  data=$1
  dir=$2
  decode=$3
  if [ ! -f $dir/graph/.done ]; then
    utils/mkgraph.sh data/lang $dir $dir/graph
    touch $dir/graph/.done
  fi
  if [ ! -f $decode/.done ]; then
    steps/decode.sh --nj $decode_nj --acwt 0.0333 
      --scoring-opts "$sopt" --skip-scoring true \
      --cmd "$decode_cmd" "${decode_extra_opts[@]}"\
      $dir/graph $data $decode |tee $decode/decode.log || exit 1;
    touch $decode/.done
  fi
}

#decode_lda_mllt data/app_$type.uem/ exp_BNF/tri5  exp_BNF/tri5/decode_${type}.uem


decode_sat() { 
  echo "Spawning decoding job for tri6"
  data=$1
  dir=$2
  decode=$3
  mkdir -p $decode
  if [ ! -f $dir/graph/.done ]; then
    utils/mkgraph.sh data/lang $dir $dir/graph
    touch $dir/graph/.done
  fi
  if [ ! -f $decode/.done ]; then
    steps/decode_fmllr_extra.sh --nj $decode_nj --acwt 0.0333  --skip-scoring true\
      --cmd "$decode_cmd" "${decode_extra_opts[@]}"\
      $dir/graph $data $decode |tee $decode/decode.log || exit 1;
    touch $decode/.done
  fi
}

decode_sat data/app_$type.uem exp_BNF/tri6  exp_BNF/tri6/decode_${type}.uem


echo ---------------------------------------------------------------------
echo "Spawning exp_BNF/sgmm7/decode[_fmllr] on" `date`
echo ---------------------------------------------------------------------
decode_sgmm2() { 
  data=$1
  dir=$2
  decode=$3
  transforms=$4
  mkdir -p $decode
  if [ ! -f $dir/graph/.done ]; then
    utils/mkgraph.sh data/lang $dir $dir/graph
    touch $dir/graph/.done
  fi
  if [ ! -f $decode/.done ]; then
    steps/decode_sgmm2.sh --use-fmllr true --nj $decode_nj \
      --beam 15 --lat-beam 8 \
      --transform-dir $transforms \
      --acwt 0.05 --skip-scoring true \
      --cmd "$decode_cmd" "${decode_extra_opts[@]}"\
      $dir/graph $data $decode | tee $decode/decode.log || exit 1;
    touch $decode/.done;
  fi
}

decode_sgmm2 data/app_$type.uem   exp_BNF/sgmm7 exp_BNF/sgmm7/decode_fmllr_${type}.uem   exp_BNF/tri6/decode_${type}.uem


for iter in 1 2 3 4; do
  decode=exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_${type}.uem_it$iter

  if [ ! -f $decode/.done ]; then
    
    steps/decode_sgmm2_rescore.sh --skip-scoring true \
        --cmd "$decode_cmd" --iter $iter --transform-dir exp_BNF/tri6/decode_${type}.uem \
        data/lang data/app_$type.uem/ exp_BNF/sgmm7/decode_fmllr_${type}.uem $decode || exit 1;
  
    touch $decode/.done 
  fi
done


# scoring and keyword search:
for iter in 1 2 3 4; do
  decode=exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_${type}.uem_it$iter

  if [ ! -f $decode/.score.done ]; then
    (   
      local/lattice_to_ctm.sh --cmd "$decode_cmd" "${lmwt_bnf_extra_opts[@]}" \
        --word-ins-penalty 0.5 \
        data/app_${type}.uem data/lang $decode || exit 1

      if [[ $type == shadow ]]; then
        local/split_ctms.sh --cer $cer --cmd "$decode_cmd"  "${lmwt_bnf_extra_opts[@]}" \
          data/app_shadow.uem $decode data/$dev2shadow data/eval.uem || exit 1
      else
        local/score_stm.sh --cer $cer --cmd "$decode_cmd" "${lmwt_bnf_extra_opts[@]}" \
          data/app_${type}.uem data/lang $decode || exit 1
      fi

      touch $decode/.score.done
   ) &
  fi
  
  mkdir -p $decode/kws
  (
  if [ ! -f $decode/.kws.done ]; then
    if [ $type == shadow ]; then # shadow data
      local/shadow_set_kws_search.sh --cmd "$decode_cmd" \
        "${lmwt_bnf_extra_opts[@]}" --max-states 150000 \
        data/app_shadow.uem data/lang $decode data/$dev2shadow data/eval.uem || exit 1
    else 
      local/kws_search.sh "${lmwt_bnf_extra_opts[@]}"  --cmd "$decode_cmd" \
        --duptime $duptime --max-states 150000 \
        data/lang data/app_$type.uem $decode || exit 1
    fi
    touch $decode/.kws.done 
  fi
  ) &
  wait
done


jobs

wait
echo ---------------------------------------------------------------------
echo "Finished successfully on" `date`
echo ---------------------------------------------------------------------


exit 0
