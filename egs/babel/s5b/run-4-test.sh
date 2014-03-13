#!/bin/bash 
set -e
set -o pipefail

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;


type=dev10h
dev2shadow=dev10h.uem
eval2shadow=eval.uem
data_only=false
fast_path=true
skip_kws=false
skip_stt=false
skip_scoring=false
max_states=150000
wip=0.5

echo "run-4-test.sh $@"

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $(basename $0) --type (dev10h|dev2h|eval|shadow)"
  exit 1
fi

#For backward compatibility, we will assume the dirid is "UEM" if no extension used
if [[ "${type%.*}" ==  "$type" ]]; then
  #If no extension had been  used
  datadir=data/${type}.uem
  dirid=${type}.uem
else
  datadir=data/${type}
  dirid=${type}
  type="${type%%.*}"
fi
#Now the $type value will be the dataset name without any UEM extension. 
#Still,we are able to figure out the correct output directory to use when running decoding

if [[ "$type" != "dev10h" && "$type" != "dev2h" && "$type" != "eval" && "$type" != "shadow" ]] ; then
  echo "Warning: invalid variable type=${type}, valid values are dev10h|dev2h|eval"
  echo "Hope you know what your ar doing!"
fi

if [ $type == shadow ] ; then
  shadow_set_extra_opts=(--dev2shadow data/${dev2shadow} --eval2shadow data/${eval2shadow} )
else
  shadow_set_extra_opts=()
fi


function make_plp {
  t=$1
  if $use_pitch; then
    steps/make_plp_pitch.sh --cmd "$decode_cmd" --nj $my_nj data/${t} exp/make_plp_pitch/${t} plp
  else
    steps/make_plp.sh --cmd "$decode_cmd" --nj $my_nj data/${t} exp/make_plp/${t} plp
  fi
  utils/fix_data_dir.sh data/${t}
  steps/compute_cmvn_stats.sh data/${t} exp/make_plp/${t} plp
  utils/fix_data_dir.sh data/${t}
}

#if [ ${type} == shadow ] || [ ${type} == eval ] ; then
if  [[ ${type} == shadow || $type == eval || $type == semitrain || $type == devtrain ]] ; then
  mandatory_variables=""
  optional_variables=""
else
  mandatory_variables="${type}_data_dir ${type}_data_list ${type}_stm_file \
    ${type}_ecf_file ${type}_kwlist_file ${type}_rttm_file ${type}_nj" 
  optional_variables="${type}_subset_ecf "
fi

eval my_data_dir=\$${type}_data_dir
eval my_data_list=\$${type}_data_list
eval my_stm_file=\$${type}_stm_file
eval my_data_cmudb=\$${type}_data_cmudb

eval my_ecf_file=\$${type}_ecf_file 
eval my_subset_ecf=\$${type}_subset_ecf 
eval my_kwlist_file=\$${type}_kwlist_file 
eval my_rttm_file=\$${type}_rttm_file
eval my_nj=\$${type}_nj  #for shadow, this will be re-set when appropriate

declare -A my_more_kwlists
eval my_more_kwlist_keys="\${!${type}_more_kwlists[@]}"
declare -p my_more_kwlist_keys
for key in $my_more_kwlist_keys  # make sure you include the quotes there
do
  echo $key
  eval my_more_kwlist_val="\${${type}_more_kwlists[$key]}"
  echo $my_more_kwlist_val
  my_more_kwlists["$key"]="${my_more_kwlist_val}"
done


for variable in $mandatory_variables ; do
  eval my_variable=\$${variable}
  if [ -z $my_variable ] ; then
    echo "Mandatory variable $variable is not set! " \
         "You should probably set the variable in the config file "
    exit 1
  else
    echo "$variable=$my_variable"
  fi
done

for variable in $option_variables ; do
  eval my_variable=\$${variable}
  echo "$variable=$my_variable"
done

if [[ $type == shadow ]] ; then
  if [ ! -f ${datadir}/.done ]; then
    # we expect that the ${dev2shadow} as well as ${eval2shadow} already exist
    if [ ! -f data/${dev2shadow}/.done ]; then
      echo "Error: data/${dev2shadow}/.done does not exist."
      echo "Create the directory data/${dev2shadow} first, by calling $0 --type $dev2shadow --dataonly"
      exit 1
    fi
    if [ ! -f data/${eval2shadow}/.done ]; then
      echo "Error: data/${eval2shadow}/.done does not exist."
      echo "Create the directory data/${eval2shadow} first, by calling $0 --type $eval2shadow --dataonly"
      exit 1
    fi

    local/create_shadow_dataset.sh ${datadir} data/${dev2shadow} data/${eval2shadow}
    utils/fix_data_dir.sh ${datadir}
    touch ${datadir}/.done
  fi
  my_nj=$eval_nj
elif [[ $type == eval || $type == semitrain || $type == devtrain ]] ; then
  if [ ! -d data/raw_${type}_data ]; then
    echo ---------------------------------------------------------------------
    echo "Subsetting the ${type} set"
    echo ---------------------------------------------------------------------
    
    local/make_corpus_subset.sh --ignore-missing-txt true "$my_data_dir" "$my_data_list" ./data/raw_${type}_data
  fi
  my_data_dir=`readlink -f ./data/raw_${type}_data`

  skip_scoring=true

  nj_max=`cat $my_data_list | wc -l`
  if [[ "$nj_max" -lt "$my_nj" ]] ; then
    echo "The maximum reasonable number of jobs is $nj_max -- you have $my_nj! (The training and decoding process has file-granularity)"
    exit 1
    my_nj=$nj_max
  fi
  
  if [ ! -f ${datadir}/.done ]; then
    if [[ ! -f ${datadir}/wav.scp || ${datadir}/wav.scp -ot "$my_data_cmudb" ]]; then
      echo ---------------------------------------------------------------------
      echo "Preparing ${type} data lists in ${datadir} on" `date`
      echo ---------------------------------------------------------------------
      mkdir -p ${datadir}
      local/cmu_uem2kaldi_dir.sh --filelist $my_data_list \
        $my_data_cmudb  $my_data_dir ${datadir}
    fi

    if [ ! -f ${datadir}/.plp.done ]; then
      echo ---------------------------------------------------------------------
      echo "Preparing ${type} parametrization files in ${datadir} on" `date`
      echo ---------------------------------------------------------------------
      make_plp ${dirid}
      touch ${datadir}/.plp.done
    fi

    touch ${datadir}/.done
  fi
else
  if [ ! -f data/raw_${type}_data/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Subsetting the ${type} set"
    echo ---------------------------------------------------------------------
    
    local/make_corpus_subset.sh "$my_data_dir" "$my_data_list" ./data/raw_${type}_data
    touch data/raw_${type}_data/.done
  fi
  my_data_dir=`readlink -f ./data/raw_${type}_data`

  nj_max=`cat $my_data_list | wc -l`
  if [[ "$nj_max" -lt "$my_nj" ]] ; then
    echo "The maximum reasonable number of jobs is $nj_max -- you have $my_nj! (The training and decoding process has file-granularity)"
    exit 1
    my_nj=$nj_max
  fi

  if [ ! -f ${datadir}/.done ]; then
    if [[ ! -f ${datadir}/wav.scp || ${datadir}/wav.scp -ot "$my_data_cmudb" ]]; then
      echo ---------------------------------------------------------------------
      echo "Preparing ${type} data lists in ${datadir} on" `date`
      echo ---------------------------------------------------------------------
      mkdir -p ${datadir}
      local/cmu_uem2kaldi_dir.sh --filelist $my_data_list \
        $my_data_cmudb  $my_data_dir ${datadir}
    fi

    echo ---------------------------------------------------------------------
    echo "Preparing ${type} stm files in ${datadir} on" `date`
    echo ---------------------------------------------------------------------
    if [ ! -z $my_stm_file ] ; then
      local/augment_original_stm.pl $my_stm_file ${datadir}
    elif [[ $type == shadow || $type == eval ]]; then
      echo "Not doing anything for the STM file!"
    else
      local/prepare_stm.pl --fragmentMarkers \-\*\~ ${datadir}
    fi

    if [ ! -f ${datadir}/.plp.done ]; then
      echo ---------------------------------------------------------------------
      echo "Preparing ${type} parametrization files in ${datadir} on" `date`
      echo ---------------------------------------------------------------------
      make_plp ${dirid}
      touch ${datadir}/.plp.done
    fi

    touch ${datadir}/.done
  fi
fi

#####################################################################
#
# data directory preparation
#
#####################################################################
echo ---------------------------------------------------------------------
echo "Preparing ${type} kws data files in ${datadir} on" `date`
echo ---------------------------------------------------------------------
if ! $skip_kws  && [ ! -f ${datadir}/kws/.done ] ; then
  if [[ $type == shadow ]]; then
    
    # we expect that the ${dev2shadow} as well as ${eval2shadow} already exist
    if [ ! -f data/${dev2shadow}/kws/.done ]; then
      echo "Error: data/${dev2shadow}/kws/.done does not exist."
      echo "Create the directory data/${dev2shadow} first, by calling $0 --type $dev2shadow --dataonly"
      exit 1
    fi
    if [ ! -f data/${eval2shadow}/kws/.done ]; then
      echo "Error: data/${eval2shadow}/kws/.done does not exist."
      echo "Create the directory data/${eval2shadow} first, by calling $0 --type $eval2shadow --dataonly"
      exit 1
    fi


    local/kws_data_prep.sh --case_insensitive $case_insensitive \
      "${icu_opt[@]}" \
      data/lang ${datadir} ${datadir}/kws || exit 1
    utils/fix_data_dir.sh ${datadir}


    touch ${datadir}/kws/.done
  else
    kws_flags=()
    if [ ! -z $my_rttm_file ] ; then
      kws_flags+=(--rttm-file $my_rttm_file )
    fi
    if [ $my_subset_ecf ] ; then
      kws_flags+=(--subset-ecf $my_data_list)
    fi
    
    local/kws_setup.sh --case_insensitive $case_insensitive \
      "${kws_flags[@]}" "${icu_opt[@]}" \
      $my_ecf_file $my_kwlist_file data/lang ${datadir} || exit 1

    if [ ${#my_more_kwlists[@]} -ne 0  ] ; then
      touch $datadir/extra_kws_tasks
      for extraid in "${!my_more_kwlists[@]}" ; do
        kwlist=${my_more_kwlists[$extraid]}
        local/kws_setup.sh --rttm-file ${datadir}/kws/rttm --extraid $extraid \
          --case_insensitive $case_insensitive "${icu_opt[@]}"  \
          ${datadir}/kws/ecf.xml $kwlist data/lang ${datadir} || exit 1
        echo $extraid >> $datadir/extra_kws_tasks;  sort -u $datadir/extra_kws_tasks -o  $datadir/extra_kws_tasks
      done
    fi
    touch ${datadir}/kws/.done
  fi

  langid=`ls -1 data/raw_${type}_data/audio/ | head -n 1| cut -d '_' -f 3`
  local/kws_setup.sh --kwlist-wordlist true --rttm-file $datadir/kws/rttm  --extraid fullvocab  \
    $datadir/kws/ecf.xml <(cat data/lang/words.txt | grep -v -F "<" | grep -v -F "#"  | \
                            awk "{printf \"KWID$langid-FULLVOCAB-%05d %s\\n\", \$2, \$1 }" ) \
    data/lang ${datadir} || exit 1
  echo fullvocab >> $datadir/extra_kws_tasks;  sort -u $datadir/extra_kws_tasks -o  $datadir/extra_kws_tasks

  #local/make_lexicon_subset.sh $data_dir/transcription $lexicon_file $datadir/filtered_lexicon.txt
  #local/kws_setup.sh --kwlist-wordlist true --rttm-file $datadir/kws/rttm  --extraid vocab  \
  #  $datadir/kws/ecf.xml  <(cut -f 1 $datadir/filtered_lexicon.txt  | grep -v -F "<" | grep -v -F "#"  | awk "{printf \"KWID$langid-VOCAB-%05d data/raw_${type}_data/.done %s\\n\", \$2, \$1 }" ) data/lang ${datadir}
fi

if $data_only ; then
  echo "Exiting, as data-only was requested..."
  exit 0;
fi


####################################################################
##
## FMLLR decoding 
##
####################################################################
decode=exp/tri5/decode_${dirid}
if [ ! -f ${decode}/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Spawning decoding with SAT models  on" `date`
  echo ---------------------------------------------------------------------
  utils/mkgraph.sh \
    data/lang exp/tri5 exp/tri5/graph |tee exp/tri5/mkgraph.log

  mkdir -p $decode
  #By default, we do not care about the lattices for this step -- we just want the transforms
  #Therefore, we will reduce the beam sizes, to reduce the decoding times
  steps/decode_fmllr_extra.sh --skip-scoring true --beam 10 --lattice-beam 4\
    --nj $my_nj --cmd "$decode_cmd" "${decode_extra_opts[@]}"\
    exp/tri5/graph ${datadir} ${decode} |tee ${decode}/decode.log
  touch ${decode}/.done
fi

if ! $fast_path ; then
  local/run_kws_stt_task.sh --cer $cer --max-states $max_states --skip-scoring $skip_scoring\
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --wip $wip \
    "${shadow_set_extra_opts[@]}" "${lmwt_plp_extra_opts[@]}" \
    ${datadir} data/lang ${decode}

  local/run_kws_stt_task.sh --cer $cer --max-states $max_states --skip-scoring $skip_scoring\
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --wip $wip \
    "${shadow_set_extra_opts[@]}" "${lmwt_plp_extra_opts[@]}" \
    ${datadir} data/lang ${decode}.si
fi

####################################################################
## SGMM2 decoding 
####################################################################
decode=exp/sgmm5/decode_fmllr_${dirid}
if [ ! -f $decode/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Spawning $decode on" `date`
  echo ---------------------------------------------------------------------
  utils/mkgraph.sh \
    data/lang exp/sgmm5 exp/sgmm5/graph |tee exp/sgmm5/mkgraph.log

  mkdir -p $decode
  steps/decode_sgmm2.sh --skip-scoring true --use-fmllr true --nj $my_nj \
    --cmd "$decode_cmd" --transform-dir exp/tri5/decode_${dirid} "${decode_extra_opts[@]}"\
    exp/sgmm5/graph ${datadir} $decode |tee $decode/decode.log
  touch $decode/.done

  if ! $fast_path ; then
    local/run_kws_stt_task.sh --cer $cer --max-states $max_states --skip-scoring $skip_scoring\
      --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --wip $wip \
      "${shadow_set_extra_opts[@]}" "${lmwt_plp_extra_opts[@]}" \
      ${datadir} data/lang  exp/sgmm5/decode_fmllr_${dirid}
  fi

  ####################################################################
  ##
  ## SGMM_MMI rescoring
  ##
  ####################################################################

  for iter in 1 2 3 4; do
    # Decode SGMM+MMI (via rescoring).
    decode=exp/sgmm5_mmi_b0.1/decode_fmllr_${dirid}_it$iter
    if [ ! -f $decode/.done ]; then

      mkdir -p $decode
      steps/decode_sgmm2_rescore.sh  --skip-scoring true \
        --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode_${dirid} \
        data/lang ${datadir} exp/sgmm5/decode_fmllr_${dirid} $decode | tee ${decode}/decode.log

      touch $decode/.done
    fi
  done

  #We are done -- all lattices has been generated. We have to
  #a)Run MBR decoding
  #b)Run KW search
  for iter in 1 2 3 4; do
    # Decode SGMM+MMI (via rescoring).
    decode=exp/sgmm5_mmi_b0.1/decode_fmllr_${dirid}_it$iter
    local/run_kws_stt_task.sh --cer $cer --max-states $max_states --skip-scoring $skip_scoring\
      --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --wip $wip \
      "${shadow_set_extra_opts[@]}" "${lmwt_plp_extra_opts[@]}" \
      ${datadir} data/lang $decode
  done
fi

####################################################################
##
## DNN ("compatibility") decoding -- also, just decode the "default" net
##
####################################################################
if [ -f exp/tri6_nnet/.done ]; then
  decode=exp/tri6_nnet/decode_${dirid}
  if [ ! -f $decode/.done ]; then
    mkdir -p $decode
    steps/nnet2/decode.sh \
      --minimize $minimize --cmd "$decode_cmd" --nj $my_nj \
      --beam $dnn_beam --lat-beam $dnn_lat_beam \
      --skip-scoring true "${decode_extra_opts[@]}" \
      --transform-dir exp/tri5/decode_${dirid} \
      exp/tri5/graph ${datadir} $decode | tee $decode/decode.log

    touch $decode/.done
  fi

  local/run_kws_stt_task.sh --cer $cer --max-states $max_states --skip-scoring $skip_scoring\
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --wip $wip \
    "${shadow_set_extra_opts[@]}" "${lmwt_dnn_extra_opts[@]}" \
    ${datadir} data/lang $decode
fi

####################################################################
##
## DNN (nextgen DNN) decoding
##
####################################################################
if [ -f exp/tri6a_nnet/.done ]; then
  decode=exp/tri6a_nnet/decode_${dirid}
  if [ ! -f $decode/.done ]; then
    mkdir -p $decode
    steps/nnet2/decode.sh \
      --minimize $minimize --cmd "$decode_cmd" --nj $my_nj \
      --beam $dnn_beam --lat-beam $dnn_lat_beam \
      --skip-scoring true "${decode_extra_opts[@]}" \
      --transform-dir exp/tri5/decode_${dirid} \
      exp/tri5/graph ${datadir} $decode | tee $decode/decode.log

    touch $decode/.done
  fi

  local/run_kws_stt_task.sh --cer $cer --max-states $max_states --skip-scoring $skip_scoring\
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --wip $wip \
    "${shadow_set_extra_opts[@]}" "${lmwt_dnn_extra_opts[@]}" \
    ${datadir} data/lang $decode
fi

####################################################################
##
## DNN (ensemble) decoding
##
####################################################################
if [ -f exp/tri6b_nnet/.done ]; then
  decode=exp/tri6b_nnet/decode_${dirid}
  if [ ! -f $decode/.done ]; then
    mkdir -p $decode
    steps/nnet2/decode.sh \
      --minimize $minimize --cmd "$decode_cmd" --nj $my_nj \
      --beam $dnn_beam --lat-beam $dnn_lat_beam \
      --skip-scoring true "${decode_extra_opts[@]}" \
      --transform-dir exp/tri5/decode_${dirid} \
      exp/tri5/graph ${datadir} $decode | tee $decode/decode.log

    touch $decode/.done
  fi

  local/run_kws_stt_task.sh --cer $cer --max-states $max_states --skip-scoring $skip_scoring\
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --wip $wip \
    "${shadow_set_extra_opts[@]}" "${lmwt_dnn_extra_opts[@]}" \
    ${datadir} data/lang $decode
fi
####################################################################
##
## DNN_MPE decoding
##
####################################################################
if [ -f exp/tri6_nnet_mpe/.done ]; then
  for epoch in 1 2 3 4; do
    decode=exp/tri6_nnet_mpe/decode_${dirid}_epoch$epoch
    if [ ! -f $decode/.done ]; then
      mkdir -p $decode
      steps/nnet2/decode.sh --minimize $minimize \
        --cmd "$decode_cmd" --nj $my_nj --iter epoch$epoch \
        --beam $dnn_beam --lat-beam $dnn_lat_beam \
        --skip-scoring true "${decode_extra_opts[@]}" \
        --transform-dir exp/tri5/decode_${dirid} \
        exp/tri5/graph ${datadir} $decode | tee $decode/decode.log

      touch $decode/.done
    fi

    local/run_kws_stt_task.sh --cer $cer --max-states $max_states --skip-scoring $skip_scoring\
      --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --wip $wip \
      "${shadow_set_extra_opts[@]}" "${lmwt_dnn_extra_opts[@]}" \
      ${datadir} data/lang $decode
  done
fi

echo "Everything looking good...." 
exit 0
