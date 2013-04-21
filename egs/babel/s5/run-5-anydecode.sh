#!/bin/bash 
set -e
set -o pipefail

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;


type=""
first_stage="tri5"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $(basename $0) --type (dev10h|dev2h|eval)"
fi

if [[ "$type" != "dev10h" && "$type" != "dev2h" && "$type" != "eval" && "$type" != "shadow" ]] ; then
  echo "Warning: invalid variable type=$type, valid values are dev10h|dev2h|eval"
  echo "Hope you know what your ar doing!"
fi

eval my_data_dir=\$${type}_data_dir
eval my_data_list=\$${type}_data_list
if [ "$type" == dev2h ] ; then
  my_nj=$decode_nj

  eval my_ecf_file=$ecf_file 
  eval my_subset_ecf=$subset_ecf 
  eval my_kwlist_file=$kwlist_file 
  eval my_rttm_file=$rttm_file
elif [ "$type" == dev10h ] ; then
  eval my_nj=\$${type}_nj

  eval my_ecf_file=$ecf_file 
  eval my_subset_ecf=false
  eval my_kwlist_file=$kwlist_file 
  eval my_rttm_file=$rttm_file
else 
  eval my_nj=\$${type}_nj

  eval my_ecf_file=\$${type}_ecf_file 
  eval my_subset_ecf=\$${type}_subset_ecf 
  eval my_kwlist_file=\$${type}_kwlist_file 
  eval my_rttm_file=\$${type}_rttm_file
fi

if [ ! -d data/raw_${type}_data ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting the ${type} set"
  echo ---------------------------------------------------------------------
  
  local/make_corpus_subset.sh "$my_data_dir" "$my_data_list" ./data/raw_${type}_data || exit 1
fi
my_data_dir=`readlink -f ./data/raw_${type}_data`

nj_max=`cat $my_data_list | wc -l`
if [[ "$nj_max" -lt "$my_nj" ]] ; then
  echo "The maximum reasonable number of jobs is $nj_max -- you have $my_nj! (The training and decoding process has file-granularity)"
  exit 1
  my_nj=$nj_max
fi

if [[ ! -f data/${type}/wav.scp || data/${type}/wav.scp -ot "$dev2h_data_dir" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing ${type} data lists in data/${type} on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p data/${type}
  local/prepare_acoustic_training_data.pl \
    --fragmentMarkers \-\*\~ \
    $dev2h_data_dir data/${type} > data/${type}/skipped_utts.log || exit 1
fi


if [[ ! -f data/${type}/glm || data/${type}/glm -ot "$glmFile" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing ${type} stm files in data/${type} on" `date`
  echo ---------------------------------------------------------------------
  local/prepare_stm.pl --fragmentMarkers \-\*\~ data/${type} || exit 1
  cp $glmFile data/${type}/glm
fi

if [ ! -f data/${type}/.kws.done ]; then
  icu_opt=()
  if [ ! -z $icu_transform ] ; then
    icu_opt=(--use-icu true --icu-transfrom $icu_transform)
  fi
  if [[ $my_subset_ecf ]] ; then
    local/kws_setup.sh --case-insensitive $case_insensitive --subset-ecf $my_data_list \
      --rttm-file "$my_rttm_file ${icu_opt[@]}" \
      $my_ecf_file $my_kwlist_file data/lang data/${type}
  else
    local/kws_setup.sh --case-insensitive $case_insensitive                            \
      --rttm-file $my_rttm_file "${icu_opt[@]}" \
      $my_ecf_file $my_kwlist_file data/lang data/${type} 
  fi
  touch data/${type}/.kws.done
fi

if [ ! -f data/${type}/.plp.done ]; then
  steps/make_plp.sh \
    --cmd "$train_cmd" --nj $my_nj \
    data/${type} exp/make_plp/${type} plp || exit 1
  steps/compute_cmvn_stats.sh \
    data/${type} exp/make_plp/${type} plp || exit 1
  # In case plp extraction failed on some utterance, delist them
  utils/fix_data_dir.sh data/${type}
  touch data/${type}/.plp.done
fi

decode_si() { 
  dir=$1
  if [ ! -f $dir/decode_${type}/.done ]; then
    echo ---------------------------------------------------------------
    echo "Spawning decoding with triphone models in $dir on" `date`
    echo ---------------------------------------------------------------
    mkdir -p $dir/graph
    utils/mkgraph.sh data/lang $dir $dir/graph |tee $dir/mkgraph.log || exit 1;
    mkdir -p $dir/decode_${type}
    steps/decode.sh --nj $my_nj --cmd "$decode_cmd" "${decode_extra_opts[@]}" \
       $dir/graph data/${type} $dir/decode_${type} |tee $dir/decode_${type}.log || exit 1;

    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/${type} $dir/decode_${type}
    touch $dir/decode_${type}/.done
    echo "See $dir/mkgraph.log and $dir/decode_${type}.log for decoding outcomes"
  fi
}
  
if [ ! -f exp/tri5/decode_${type}/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Spawning decoding with SAT models  on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p exp/tri5/graph
  utils/mkgraph.sh \
      data/lang exp/tri5 exp/tri5/graph |tee exp/tri5/mkgraph.log
  mkdir -p exp/tri5/decode_${type}
  touch exp/tri5/decode_${type}.started # A signal to the SGMM2 decoding step
  steps/decode_fmllr.sh --nj $my_nj \
    --cmd "$decode_cmd" "${decode_extra_opts[@]}" \
    exp/tri5/graph data/${type} exp/tri5/decode_${type} |tee exp/tri5/decode_${type}.log 
  touch exp/tri5/decode_${type}/.done
fi

if [ ! -f exp/tri5/decode_${type}/kws/.done ]; then
  local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
      data/lang data/${type} exp/tri5/decode_${type}.si
  local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
      data/lang data/${type} exp/tri5/decode_${type}
  touch exp/tri5/decode_${type}/kws/.done 
fi

if [ ! -f exp/sgmm5/decode_${type}_fmllr/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Spawning exp/sgmm5/decode_${type}_fmllr on" `date`
  echo ---------------------------------------------------------------------
  echo "exp/sgmm5/decode_${type} will wait on tri5 decode if necessary"
  mkdir -p exp/sgmm5/graph
  utils/mkgraph.sh \
      data/lang exp/sgmm5 exp/sgmm5/graph |tee exp/sgmm5/mkgraph.log

  steps/decode_sgmm2.sh --use-fmllr true --nj $my_nj \
      --cmd "$decode_cmd" "${decode_extra_opts[@]}" \
      --transform-dir exp/tri5/decode_${type} \
      exp/sgmm5/graph data/${type}/ exp/sgmm5/decode_${type}_fmllr |tee exp/sgmm5/decode_${type}_fmllr.log
  touch exp/sgmm5/decode_${type}_fmllr/.done
fi

if [ ! -f exp/sgmm5/decode_${type}_fmllr/kws/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/sgmm5/decode_${type}_fmllr/kws on" `date`
  echo ---------------------------------------------------------------------
  local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
      data/lang data/${type} exp/sgmm5/decode_${type}_fmllr
  touch exp/sgmm5/decode_${type}_fmllr/kws/.done
fi

echo ---------------------------------------------------------------------
echo "Starting exp/sgmm5_mmi_b0.1/decode_${type}[_fmllr] on" `date`
echo ---------------------------------------------------------------------
for iter in 1 2 3 4; do
  if [ ! -f exp/sgmm5_mmi_b0.1/decode_${type}_fmllr_it$iter/.done ]; then
    echo "Waiting for exp/sgmm5/decode_${type}_fmllr/.done if necessary"
    steps/decode_sgmm2_rescore.sh \
        --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode_${type} \
        data/lang data/${type} exp/sgmm5/decode_${type}_fmllr exp/sgmm5_mmi_b0.1/decode_${type}_fmllr_it$iter
    touch exp/sgmm5_mmi_b0.1/decode_${type}_fmllr_it$iter/.done
  fi
  if [ ! -f exp/sgmm5_mmi_b0.1/decode_${type}_fmllr_it$iter/kws/.done ]; then
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/${type} exp/sgmm5_mmi_b0.1/decode_${type}_fmllr_it$iter
    touch exp/sgmm5_mmi_b0.1/decode_${type}_fmllr_it$iter/kws/.done
  fi
done

if [ -f exp/tri6_nnet/.done ]; then
  if [ ! -f exp/tri6_nnet/decode_$type/.done ]; then
    steps/decode_nnet_cpu.sh --cmd "$decode_cmd" \
      --nj $my_nj "${decode_extra_opts[@]}" \
      --transform-dir exp/tri5/decode_$type \
      exp/tri5/graph data/$type exp/tri6_nnet/decode_$type 
    touch exp/tri6_nnet/decode_${type}/.done
  fi
  if [ ! -f exp/tri6_nnet/decode_$type/kws/.done ]; then
    local/kws_search.sh  --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/$type exp/tri6_nnet/decode_$type
    touch exp/tri6_nnet/decode_$type/kws/.done 
  fi  
fi

echo "Everything looking good...." 
exit 0
