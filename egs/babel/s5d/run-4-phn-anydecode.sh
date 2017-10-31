#!/bin/bash
set -e
set -o pipefail

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;


dir=dev10h.phn.pem
kind=
data_only=false
fast_path=true
skip_kws=false
skip_stt=false
skip_scoring=
extra_kws=true
vocab_kws=false
tri5_only=false
wip=0.5

nnet3_model=nnet3/tdnn_sp
is_rnn=false
extra_left_context=0
extra_right_context=0
frames_per_chunk=0

echo $0 "$@"

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $(basename $0) --type (dev10h.phn|dev2h.phn|eval.phn|shadow.phn)"
  exit 1
fi

#This seems to be the only functioning way how to ensure the comple
#set of scripts will exit when sourcing several of them together
#Otherwise, the CTRL-C just terminates the deepest sourced script ?
# Let shell functions inherit ERR trap.  Same as `set -E'.
set -o errtrace
trap "echo Exited!; exit;" SIGINT SIGTERM

./local/check_tools.sh || exit 1

# Set proxy search parameters for the extended lexicon case.
if [ -f data/.extlex ]; then
  proxy_phone_beam=$extlex_proxy_phone_beam
  proxy_phone_nbest=$extlex_proxy_phone_nbest
  proxy_beam=$extlex_proxy_beam
  proxy_nbest=$extlex_proxy_nbest
fi

dataset_segments=${dir##*.}
dataset_dir=data/$dir
dataset_id=$dir
dataset_type=${dir%%.phn.*}
#By default, we want the script to accept how the dataset should be handled,
#i.e. of  what kind is the dataset
if [ -z ${kind} ] ; then
  if [ "$dataset_type" == "dev2h" ] || [ "$dataset_type" == "dev10h" ]; then
    dataset_kind=supervised
  else
    dataset_kind=unsupervised
  fi
else
  dataset_kind=$kind
fi

if [ -z $dataset_segments ]; then
  echo "You have to specify the segmentation type as well"
  echo "If you are trying to decode the PEM segmentation dir"
  echo "such as data/dev10h, specify dev10h.pem"
  echo "The valid segmentations types are:"
  echo "\tpem   #PEM segmentation"
  echo "\tuem   #UEM segmentation in the CMU database format"
  echo "\tseg   #UEM segmentation (kaldi-native)"
fi

if [ -z "${skip_scoring}" ] ; then
  if [ "$dataset_kind" == "unsupervised" ]; then
    skip_scoring=true
  else
    skip_scoring=false
  fi
fi

#The $dataset_type value will be the dataset name without any extrension
eval my_data_dir=( "\${${dataset_type}_data_dir[@]}" )
eval my_data_list=( "\${${dataset_type}_data_list[@]}" )
if [ -z $my_data_dir ] || [ -z $my_data_list ] ; then
  echo "Error: The dir you specified ($dataset_id) does not have existing config";
  exit 1
fi

eval my_stm_file=\$${dataset_type}_stm_file
eval my_ecf_file=\$${dataset_type}_ecf_file
eval my_rttm_file=\$${dataset_type}_rttm_file
eval my_nj=\$${dataset_type}_nj  #for shadow, this will be re-set when appropriate

if [ -z "$my_nj" ]; then
  echo >&2 "You didn't specify the number of jobs -- variable \"${dataset_type}_nj\" not defined."
  exit 1
fi
my_nj=$(($my_nj * 2))

my_subset_ecf=false
eval ind=\${${dataset_type}_subset_ecf+x}
if [ "$ind" == "x" ] ; then
  eval my_subset_ecf=\$${dataset_type}_subset_ecf
fi

declare -A my_kwlists=()
eval my_kwlists_keys="\${!${dataset_type}_kwlists[@]}"
for key in $my_kwlists_keys  # make sure you include the quotes there
do
  eval my_kwlists_val="\${${dataset_type}_kwlists[$key]}"
  my_kwlists["$key"]="${my_kwlists_val}"
done
declare -p my_kwlists
export my_kwlists

#Just a minor safety precaution to prevent using incorrect settings
#The dataset_* variables should be used.
set -e
set -o pipefail
set -u
unset dir
unset kind

function make_plp {
  target=$1
  logdir=$2
  output=$3
  if $use_pitch; then
    steps/make_plp_pitch.sh --cmd "$decode_cmd" --nj $my_nj $target $logdir $output
  else
    steps/make_plp.sh --cmd "$decode_cmd" --nj $my_nj $target $logdir $output
  fi
  utils/fix_data_dir.sh $target
  steps/compute_cmvn_stats.sh $target $logdir $output
  utils/fix_data_dir.sh $target
}

function check_variables_are_set {
  for variable in $mandatory_variables ; do
    if ! declare -p $variable ; then
      echo "Mandatory variable ${variable/my/$dataset_type} is not set! "
      echo "You should probably set the variable in the config file "
      exit 1
    else
      declare -p $variable
    fi
  done

  if [ ! -z ${optional_variables+x} ] ; then
    for variable in $optional_variables ; do
      eval my_variable=\$${variable}
      echo "$variable=$my_variable"
    done
  fi
}

if [ ! -f data/raw_${dataset_type}_data/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting the ${dataset_type} set"
  echo ---------------------------------------------------------------------

  l1=${#my_data_dir[*]}
  l2=${#my_data_list[*]}
  if [ "$l1" -ne "$l2" ]; then
    echo "Error, the number of source files lists is not the same as the number of source dirs!"
    exit 1
  fi

  resource_string=""
  if [ "$dataset_kind" == "unsupervised" ]; then
    resource_string+=" --ignore-missing-txt true"
  fi

  for i in `seq 0 $(($l1 - 1))`; do
    resource_string+=" ${my_data_dir[$i]} "
    resource_string+=" ${my_data_list[$i]} "
  done
  local/make_corpus_subset.sh $resource_string ./data/raw_${dataset_type}_data
  touch data/raw_${dataset_type}_data/.done
fi
my_data_dir=`utils/make_absolute.sh ./data/raw_${dataset_type}_data`
[ -f $my_data_dir/filelist.list ] && my_data_list=$my_data_dir/filelist.list
nj_max=`cat $my_data_list | wc -l` || nj_max=`ls $my_data_dir/audio | wc -l`

if [ "$nj_max" -lt "$my_nj" ] ; then
  echo "Number of jobs ($my_nj) is too big!"
  echo "The maximum reasonable number of jobs is $nj_max"
  my_nj=$nj_max
fi

#####################################################################
#
# Audio data directory preparation
#
#####################################################################
echo ---------------------------------------------------------------------
echo "Preparing ${dataset_kind} data files in ${dataset_dir} on" `date`
echo ---------------------------------------------------------------------
if [ ! -f  $dataset_dir/.done ] ; then
  if [ "$dataset_kind" == "supervised" ]; then
    if [ "$dataset_segments" == "seg" ]; then
      . ./local/datasets/supervised_seg.sh || exit 1
    elif [ "$dataset_segments" == "uem" ]; then
      . ./local/datasets/supervised_uem.sh || exit 1
    elif [ "$dataset_segments" == "pem" ]; then
      . ./local/datasets/supervised_pem.sh || exit 1
    else
      echo "Unknown type of the dataset: \"$dataset_segments\"!";
      echo "Valid dataset types are: seg, uem, pem";
      exit 1
    fi
  elif [ "$dataset_kind" == "unsupervised" ] ; then
    if [ "$dataset_segments" == "seg" ] ; then
      . ./local/datasets/unsupervised_seg.sh
    elif [ "$dataset_segments" == "uem" ] ; then
      . ./local/datasets/unsupervised_uem.sh
    elif [ "$dataset_segments" == "pem" ] ; then
      ##This combination does not really makes sense,
      ##Because the PEM is that we get the segmentation
      ##and because of the format of the segment files
      ##the transcript as well
      echo "ERROR: $dataset_segments combined with $dataset_type"
      echo "does not really make any sense!"
      exit 1
      #. ./local/datasets/unsupervised_pem.sh
    else
      echo "Unknown type of the dataset: \"$dataset_segments\"!";
      echo "Valid dataset types are: seg, uem, pem";
      exit 1
    fi
  else
    echo "Unknown kind of the dataset: \"$dataset_kind\"!";
    echo "Valid dataset kinds are: supervised, unsupervised, shadow";
    exit 1
  fi

  if [ ! -f ${dataset_dir}/.plp.done ]; then
    echo ---------------------------------------------------------------------
    echo "Preparing ${dataset_kind} parametrization files in ${dataset_dir} on" `date`
    echo ---------------------------------------------------------------------
    make_plp ${dataset_dir} exp/make_plp/${dataset_id} plp
    touch ${dataset_dir}/.plp.done
  fi
  touch $dataset_dir/.done
fi

if [ -f exp/nnet3/extractor/final.ie ] && [ ! -f ${dataset_dir}_hires/.mfcc.done ]; then
  dataset=$(basename $dataset_dir)
  echo ---------------------------------------------------------------------
  echo "Preparing ${dataset_kind} MFCC features in  ${dataset_dir}_hires and corresponding iVectors in exp/nnet3/ivectors_$dataset on" `date`
  echo ---------------------------------------------------------------------
  if [ ! -d ${dataset_dir}_hires ]; then
    utils/copy_data_dir.sh data/${dataset_type}.${dataset_segments}_hires data/${dataset}_hires
  fi
  ln -sf ivectors_${dataset_type}.${dataset_segments} exp/nnet3/ivectors_${dataset} || true
  touch ${dataset_dir}_hires/.done
fi
set -x
ln -sf ivectors_${dataset_type}.${dataset_segments} exp/nnet3/ivectors_${dataset} || true
set +x

#####################################################################
#
# KWS data directory preparation
#
#####################################################################
echo ---------------------------------------------------------------------
echo "Preparing kws data files in ${dataset_dir} on" `date`
echo ---------------------------------------------------------------------
lang=data/lang.phn
if ! $skip_kws ; then
  if  $extra_kws ; then
    L1_lex=data/local/dict.phn/lexiconp.txt
    . ./local/datasets/extra_kws.sh || exit 1
  fi
  if  $vocab_kws ; then
    . ./local/datasets/vocab_kws.sh || exit 1
  fi
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
if [ ! -f data/langp_test.phn//.done ]; then
  ln -sf lang.phn data/langp_test.phn || true
  touch data/langp_test.phn/.done
fi

decode=exp/tri5/decode_${dataset_id}
if [ ! -f ${decode}/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Spawning decoding with SAT models  on" `date`
  echo ---------------------------------------------------------------------
  utils/mkgraph.sh \
    data/langp_test.phn exp/tri5 exp/tri5/graph.phn |tee exp/tri5/mkgraph.phn.log

  mkdir -p $decode
  #By default, we do not care about the lattices for this step -- we just want the transforms
  #Therefore, we will reduce the beam sizes, to reduce the decoding times
  steps/decode_fmllr_extra.sh --skip-scoring true --beam 10 --lattice-beam 4\
    --nj $my_nj --cmd "$decode_cmd" "${decode_extra_opts[@]}"\
    exp/tri5/graph.phn ${dataset_dir} ${decode} |tee ${decode}/decode.log
  touch ${decode}/.done
fi

if ! $fast_path ; then
  local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
    --skip-scoring $skip_scoring --extra-kws $extra_kws --wip $wip \
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt \
    "${lmwt_plp_extra_opts[@]}" \
    ${dataset_dir} data/langp_test.phn ${decode}

  local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
    --skip-scoring $skip_scoring --extra-kws $extra_kws --wip $wip \
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt  \
    "${lmwt_plp_extra_opts[@]}" \
    ${dataset_dir} data/langp_test.phn ${decode}.si
fi

if $tri5_only; then
  echo "--tri5-only is true. So exiting."
  exit 0
fi

####################################################################
## SGMM2 decoding
## We Include the SGMM_MMI inside this, as we might only have the DNN systems
## trained and not PLP system. The DNN systems build only on the top of tri5 stage
####################################################################
if [ -f exp/sgmm5/.done ]; then
  decode=exp/sgmm5/decode_fmllr_${dataset_id}
  if [ ! -f $decode/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Spawning $decode on" `date`
    echo ---------------------------------------------------------------------
    utils/mkgraph.sh \
      data/langp_test.phn exp/sgmm5 exp/sgmm5/graph.phn |tee exp/sgmm5/mkgraph.phn.log

    mkdir -p $decode
    steps/decode_sgmm2.sh --skip-scoring true --use-fmllr true --nj $my_nj \
      --cmd "$decode_cmd" --transform-dir exp/tri5/decode_${dataset_id} "${decode_extra_opts[@]}"\
      exp/sgmm5/graph.phn ${dataset_dir} $decode |tee $decode/decode.log
    touch $decode/.done

    if ! $fast_path ; then
      local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
        --skip-scoring $skip_scoring --extra-kws $extra_kws --wip $wip \
        --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt  \
        "${lmwt_plp_extra_opts[@]}" \
        ${dataset_dir} data/langp_test.phn  exp/sgmm5/decode_fmllr_${dataset_id}
    fi
  fi

  ####################################################################
  ##
  ## SGMM_MMI rescoring
  ##
  ####################################################################

  for iter in 1 2 3 4; do
      # Decode SGMM+MMI (via rescoring).
    decode=exp/sgmm5_mmi_b0.1/decode_fmllr_${dataset_id}_it$iter
    if [ ! -f $decode/.done ]; then

      mkdir -p $decode
      steps/decode_sgmm2_rescore.sh  --skip-scoring true \
        --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode_${dataset_id} \
        data/langp_test.phn ${dataset_dir} exp/sgmm5/decode_fmllr_${dataset_id} $decode | tee ${decode}/decode.log

      touch $decode/.done
    fi
  done

  #We are done -- all lattices has been generated. We have to
  #a)Run MBR decoding
  #b)Run KW search
  for iter in 1 2 3 4; do
    # Decode SGMM+MMI (via rescoring).
    decode=exp/sgmm5_mmi_b0.1/decode_fmllr_${dataset_id}_it$iter
      local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
        --skip-scoring $skip_scoring --extra-kws $extra_kws --wip $wip \
        --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt  \
      "${lmwt_plp_extra_opts[@]}" \
      ${dataset_dir} data/langp_test.phn $decode
  done
fi



####################################################################
##
## DNN ("compatibility") decoding -- also, just decode the "default" net
##
####################################################################
if [ -f exp/tri6_nnet/.done ]; then
  decode=exp/tri6_nnet/decode_${dataset_id}
  if [ ! -f $decode/.done ]; then
    mkdir -p $decode
    steps/nnet2/decode.sh \
      --minimize $minimize --cmd "$decode_cmd" --nj $my_nj \
      --beam $dnn_beam --lattice-beam $dnn_lat_beam \
      --skip-scoring true "${decode_extra_opts[@]}" \
      --transform-dir exp/tri5/decode_${dataset_id} \
      exp/tri5/graph.phn ${dataset_dir} $decode | tee $decode/decode.log

    touch $decode/.done
  fi
  local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
    --skip-scoring $skip_scoring --extra-kws $extra_kws --wip $wip \
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt  \
    "${lmwt_dnn_extra_opts[@]}" \
    ${dataset_dir} data/langp_test.phn $decode
fi

####################################################################
##
## nnet3 model decoding
##
####################################################################
if [ -f exp/nnet3/lstm_bidirectional_sp/.done ]; then
  decode=exp/nnet3/lstm_bidirectional_sp/decode_${dataset_id}.phn
  rnn_opts=" --extra-left-context 40 --extra-right-context 40  --frames-per-chunk 20 "
  decode_script=steps/nnet3/lstm/decode.sh
  if [ ! -f $decode/.done ]; then
    mkdir -p $decode
    $decode_script --nj $my_nj --cmd "$decode_cmd" $rnn_opts \
          --beam $dnn_beam --lattice-beam $dnn_lat_beam \
          --skip-scoring true  \
          --online-ivector-dir exp/nnet3/ivectors_${dataset_id} \
          exp/tri5/graph.phn ${dataset_dir}_hires $decode | tee $decode/decode.log

    touch $decode/.done
  fi

  local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
    --skip-scoring $skip_scoring --extra-kws $extra_kws --wip $wip \
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt  \
    "${lmwt_dnn_extra_opts[@]}" \
    ${dataset_dir} data/langp_test.phn $decode
fi

if [ -f exp/nnet3/lstm_sp/.done ]; then
  decode=exp/nnet3/lstm_sp/decode_${dataset_id}.phn
  rnn_opts=" --extra-left-context 40 --extra-right-context 0  --frames-per-chunk 20 "
  decode_script=steps/nnet3/lstm/decode.sh
  if [ ! -f $decode/.done ]; then
    mkdir -p $decode
    $decode_script --nj $my_nj --cmd "$decode_cmd" $rnn_opts \
          --beam $dnn_beam --lattice-beam $dnn_lat_beam \
          --skip-scoring true  \
          --online-ivector-dir exp/nnet3/ivectors_${dataset_id} \
          exp/tri5/graph.phn ${dataset_dir}_hires $decode | tee $decode/decode.log

    touch $decode/.done
  fi

  local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
    --skip-scoring $skip_scoring --extra-kws $extra_kws --wip $wip \
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt  \
    "${lmwt_dnn_extra_opts[@]}" \
    ${dataset_dir} data/langp_test.phn $decode
fi

if [ -f exp/$nnet3_model/.done ]; then
  decode=exp/$nnet3_model/decode_${dataset_id}.phn
  rnn_opts=
  decode_script=steps/nnet3/decode.sh
  if [ "$is_rnn" == "true" ]; then
    rnn_opts=" --extra-left-context $extra_left_context --extra-right-context $extra_right_context  --frames-per-chunk $frames_per_chunk "
    decode_script=steps/nnet3/lstm/decode.sh
  fi
  if [ ! -f $decode/.done ]; then
    mkdir -p $decode
    $decode_script --nj $my_nj --cmd "$decode_cmd" $rnn_opts \
          --beam $dnn_beam --lattice-beam $dnn_lat_beam \
          --skip-scoring true  \
          --online-ivector-dir exp/nnet3/ivectors_${dataset_id} \
          exp/tri5/graph.phn ${dataset_dir}_hires $decode | tee $decode/decode.log

    touch $decode/.done
  fi

  local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
    --skip-scoring $skip_scoring --extra-kws $extra_kws --wip $wip \
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt  \
    "${lmwt_dnn_extra_opts[@]}" \
    ${dataset_dir} data/langp_test.phn $decode
fi


####################################################################
##
## DNN (nextgen DNN) decoding
##
####################################################################
if [ -f exp/tri6a_nnet/.done ]; then
  decode=exp/tri6a_nnet/decode_${dataset_id}
  if [ ! -f $decode/.done ]; then
    mkdir -p $decode
    steps/nnet2/decode.sh \
      --minimize $minimize --cmd "$decode_cmd" --nj $my_nj \
      --beam $dnn_beam --lattice-beam $dnn_lat_beam \
      --skip-scoring true "${decode_extra_opts[@]}" \
      --transform-dir exp/tri5/decode_${dataset_id} \
      exp/tri5/graph.phn ${dataset_dir} $decode | tee $decode/decode.log

    touch $decode/.done
  fi

  local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
    --skip-scoring $skip_scoring --extra-kws $extra_kws --wip $wip \
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt  \
    "${lmwt_dnn_extra_opts[@]}" \
    ${dataset_dir} data/langp_test.phn $decode
fi


####################################################################
##
## DNN (ensemble) decoding
##
####################################################################
if [ -f exp/tri6b_nnet/.done ]; then
  decode=exp/tri6b_nnet/decode_${dataset_id}
  if [ ! -f $decode/.done ]; then
    mkdir -p $decode
    steps/nnet2/decode.sh \
      --minimize $minimize --cmd "$decode_cmd" --nj $my_nj \
      --beam $dnn_beam --lattice-beam $dnn_lat_beam \
      --skip-scoring true "${decode_extra_opts[@]}" \
      --transform-dir exp/tri5/decode_${dataset_id} \
      exp/tri5/graph.phn ${dataset_dir} $decode | tee $decode/decode.log

    touch $decode/.done
  fi

  local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
    --skip-scoring $skip_scoring --extra-kws $extra_kws --wip $wip \
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt  \
    "${lmwt_dnn_extra_opts[@]}" \
    ${dataset_dir} data/langp_test.phn $decode
fi
####################################################################
##
## DNN_MPE decoding
##
####################################################################
if [ -f exp/tri6_nnet_mpe/.done ]; then
  for epoch in 1 2 3 4; do
    decode=exp/tri6_nnet_mpe/decode_${dataset_id}_epoch$epoch
    if [ ! -f $decode/.done ]; then
      mkdir -p $decode
      steps/nnet2/decode.sh --minimize $minimize \
        --cmd "$decode_cmd" --nj $my_nj --iter epoch$epoch \
        --beam $dnn_beam --lattice-beam $dnn_lat_beam \
        --skip-scoring true "${decode_extra_opts[@]}" \
        --transform-dir exp/tri5/decode_${dataset_id} \
        exp/tri5/graph.phn ${dataset_dir} $decode | tee $decode/decode.log

      touch $decode/.done
    fi

    local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
      --skip-scoring $skip_scoring --extra-kws $extra_kws --wip $wip \
      --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt  \
      "${lmwt_dnn_extra_opts[@]}" \
      ${dataset_dir} data/langp_test.phn $decode
  done
fi

####################################################################
##
## DNN semi-supervised training decoding
##
####################################################################
for dnn in tri6_nnet_semi_supervised tri6_nnet_semi_supervised2 \
          tri6_nnet_supervised_tuning tri6_nnet_supervised_tuning2 ; do
  if [ -f exp/$dnn/.done ]; then
    decode=exp/$dnn/decode_${dataset_id}
    if [ ! -f $decode/.done ]; then
      mkdir -p $decode
      steps/nnet2/decode.sh \
        --minimize $minimize --cmd "$decode_cmd" --nj $my_nj \
        --beam $dnn_beam --lattice-beam $dnn_lat_beam \
        --skip-scoring true "${decode_extra_opts[@]}" \
        --transform-dir exp/tri5/decode_${dataset_id} \
        exp/tri5/graph.phn ${dataset_dir} $decode | tee $decode/decode.log

      touch $decode/.done
    fi

    local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
      --skip-scoring $skip_scoring --extra-kws $extra_kws --wip $wip \
      --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt  \
      "${lmwt_dnn_extra_opts[@]}" \
      ${dataset_dir} data/langp_test.phn $decode
  fi
done
echo "Everything looking good...."
exit 0
