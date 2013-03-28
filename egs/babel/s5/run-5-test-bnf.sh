#!/bin/bash 
set -e
set -o pipefail
set -x

cer=0
type=shadow # "type" may be "dev10h", "dev2h", or "eval15h", or "shadow" which means eval15h + dev2h. 
sopt="--min-lmwt 25 --max-lmwt 40"
sopt2="--min-lmwt 15 --max-lmwt 30"

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

. conf/parse_options.sh     

if [ $# -ne 0 ]; then
  echo "Usage: $(basename $0) [options]"
fi



[ ! -f data/${type}.uem ] && echo "No such directory data/${type}.uem" && exit 1;
[ ! -d exp/tri5/decode_${type}.uem ] && echo "No such directory exp/tri5/decode_${type}.uem" && exit 1;

# Set decode_nj; typically 64.
decode_nj=`cat exp/tri5/decode_${type}.uem/num_jobs` || exit 1;


if [ ! -f data/${type}_bnf.uem/.done ]; then
  [ ! -d data/${type}_bnf.uem ] && mkdir -p $tmpdir/data/${type}_bnf.uem && ln -s $tmpdir/data/${type}_bnf.uem data/${type}_bnf.uem
  steps_BNF/make_bnf_feat.sh --stage 0 --nj $decode_nj --cmd "$train_cmd" --transform_dir exp/tri5/decode_${type}.uem/ \
    data/${type}.uem data/${type}_bnf.uem exp_BNF/bnf_dnn exp/tri5_ali exp_BNF/make_bnf_${type} || exit 1
  touch data/${type}_bnf.uem/.done
fi

if [ ! -f data/${type}_sat.uem/.done ]; then
  [ ! -d data/${type}_sat.uem ] && mkdir -p $tmpdir/data/${type}_sat.uem && ln -s $tmpdir/data/${type}_sat.uem data/${type}_sat.uem
  steps/make_fmllr_feats.sh --cmd "$train_cmd -tc 10" --nj $decode_nj --transform-dir exp/tri5/decode_${type}.uem  \
    data/${type}_sat.uem data/${type}.uem exp/tri5 exp_BNF/make_fmllr_feats_${type}/log plp_processed || exit 1
  touch data/${type}_sat.uem/.done
fi

if [ ! -f data/${type}_app.uem/.done ]; then
  [ ! -d data/${type}_app.uem ] && mkdir -p $tmpdir/data/${type}_app.uem && ln -s $tmpdir/data/${type}_app.uem data/${type}_app.uem
  steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
    data/${type}_bnf.uem data/${type}_sat.uem data/${type}_app.uem exp_BNF/append_feats/log plp_processed/  || exit 1
  steps/compute_cmvn_stats.sh --fake \
    data/${type}_app.uem exp/make_plp/${type}_app.uem plp_processed/ || exit 1
  ln -s `pwd`/data/${type}.uem/kws data/${type}_app.uem/kws
  touch data/${type}_app.uem/.done
fi


decode_lda_mllt() { 
  data=$1
  dir=$2
  decode=$3
  if [ ! -f $decode/.done ]; then
    steps/decode.sh --nj $decode_nj --acwt 0.0333 --scoring-opts "$sopt" \
      --cmd "$decode_cmd" --skip-scoring true \
      $dir/graph $data $decode |tee $decode/decode.log || exit 1;
    touch $decode/.done
  fi
}

#decode_lda_mllt data/${type}_app.uem/ exp_BNF/tri5  exp_BNF/tri5/decode_${type}.uem


decode_sat() { 
  echo "Spawning decoding job for tri6"
  data=$1
  dir=$2
  decode=$3
  mkdir -p $decode
  if [ ! -f $decode/.done ]; then
    steps/decode_fmllr_extra.sh --nj $decode_nj --acwt 0.0333  --skip-scoring true\
      --cmd "queue.pl -l mem_free=2G,ram_free=0.5G" --num-threads 6 --parallel-opts "-pe smp 6" \
      $dir/graph $data $decode |tee $decode/decode.log || exit 1;
    touch $decode/.done
  fi
}

decode_sat data/${type}_app.uem exp_BNF/tri6  exp_BNF/tri6/decode_${type}.uem


echo ---------------------------------------------------------------------
echo "Spawning exp_BNF/sgmm7/decode[_fmllr] on" `date`
echo ---------------------------------------------------------------------
decode_sgmm2() { 
  data=$1
  dir=$2
  decode=$3
  transforms=$4
  mkdir -p $decode
  if [ ! -f $decode/.done ]; then
    steps/decode_sgmm2.sh --use-fmllr true --nj $decode_nj \
      --beam 15 --lat-beam 8 \
      --transform-dir $transforms \
      --cmd "queue.pl -l mem_free=3G,ram_free=3G" \
      --num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=3G,ram_free=0.6G" \
      --acwt 0.05 --scoring-opts "$sopt2" --skip-scoring true \
      $dir/graph $data $decode | tee $decode/decode.log || exit 1;
    touch $decode/.done;
  fi
}

decode_sgmm2 data/${type}_app.uem   exp_BNF/sgmm7 exp_BNF/sgmm7/decode_fmllr_${type}.uem   exp_BNF/tri6/decode_${type}.uem


for iter in 1 2 3 4; do
  decode=exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_${type}.uem_it$iter

  if [ ! -f $decode/.done ]; then
    
    steps/decode_sgmm2_rescore.sh --scoring-opts "$sopt2 --cer $cer" --skip-scoring true\
        --cmd "$decode_cmd" --iter $iter --transform-dir exp_BNF/tri6/decode_${type}.uem \
        data/lang data/${type}_app.uem/ exp_BNF/sgmm7/decode_fmllr_${type}.uem $decode || exit 1;
  
    touch $decode/.done 
  fi

  decode=exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_shadow.uem_it$iter
  if [ ! -f $decode/.done ]; then
    
    steps/decode_sgmm2_rescore.sh --scoring-opts "$sopt2 --cer $cer" --skip-scoring true\
        --cmd "$decode_cmd" --iter $iter --transform-dir exp_BNF/tri6/decode_shadow.uem \
        data/lang data/shadow_app.uem/ exp_BNF/sgmm7/decode_fmllr_shadow.uem $decode || exit 1;
  
    touch $decode/.done 
  fi

done

# scoring and keyword search:
for iter in 1 2 3 4; do
  decode=exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_${type}.uem_it$iter

  if [ ! -f $decode/scoring/.done ]; then
    (   
      local/lattice_to_ctm.sh --cmd "$decode_cmd" \
        $sopt2 --word-ins-penalty 0.5 \
        data/eval_app.uem data/lang $decode || exit 1

      if [[ $type == dev10h || $type == dev2h ]]; then
        local/score_stm.sh --cer $cer --cmd "$decode_cmd" $sopt2 \
          data/eval_app.uem data/lang $decode || exit 1
      fi

      touch $decode/scoring/.done
   ) &
  fi
  
  mkdir -p $decode/kws
  if [ ! -f $decode/kws/.done ]; then
    if [ $type != "shadow" ]; then
      local/kws_search.sh $sopt2 --cmd "$decode_cmd" \
        --duptime $duptime --max-states 150000 \
        data/lang data/${type}_app.uem $decode || exit 1

    else # shadow data.
      local/split_ctms.sh $sopt2 data/shadow_app.uem $decode data/dev data/test.uem || exit 1

      local/shadow_set_kws_search.sh --cmd "$decode_cmd" \
        $sopt2 --max-states 150000 \
        data/shadow_app.uem data/lang $decode data/dev data/test.uem || exit 1
    fi
    touch $decode/kws/.done 
  fi
done


jobs

wait
echo ---------------------------------------------------------------------
echo "Finished successfully on" `date`
echo ---------------------------------------------------------------------


exit 0
