#!/bin/bash 
# Copyright 2014  Pegah Ghahremani
# Apache 2.0

# decode BNF + sgmm_mmi system 
set -e
set -o pipefail

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;


dir=dev10h.pem
kind=
data_only=false
fast_path=true
skip_kws=false
extra_kws=false
skip_stt=false
skip_scoring=false
tmpdir=`pwd`
semisupervised=true
unsup_string=

. utils/parse_options.sh

type=$dir

if [ $# -ne 0 ]; then
  echo "Usage: $(basename $0) --type (dev10h|dev2h|eval|shadow)"
  echo  "--semisupervised<true>  #set to false to skip unsupervised training."
  exit 1
fi

if [ $babel_type == "full" ] && $semisupervised; then
  echo "Error: Using unsupervised training for fullLP is meaningless, use semisupervised=false "
  exit 1
fi

if [ -z "$unsup_string" ] ; then
  if $semisupervised ; then
    unsup_string="_semisup"
  else
    unsup_string=""  #" ": supervised training, _semi_supervised: unsupervised BNF training
  fi
fi

if ! echo {dev10h,dev2h,eval,unsup,shadow}{,.uem,.seg} | grep -w "$type" >/dev/null; then
  # note: echo dev10.uem | grep -w dev10h will produce a match, but this
  # doesn't matter because dev10h is also a valid value.
  echo "Invalid variable type=${type}, valid values are " {dev10h,dev2h,eval,unsup}{,.uem,.seg}
  exit 1;
fi

dataset_segments=${dir##*.}
dataset_dir=data/$dir
dataset_id=$dir
dataset_type=${dir%%.*}
#By default, we want the script to accept how the dataset should be handled,
#i.e. of  what kind is the dataset
if [ -z ${kind} ] ; then
  if [ "$dataset_type" == "dev2h" ] || [ "$dataset_type" == "dev10h" ] ; then
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

if [ "$dataset_kind" == "unsupervised" ]; then
  skip_scoring=true
fi

dirid=${type}
exp_dir=exp_bnf${unsup_string}
data_bnf_dir=data_bnf${unsup_string}
param_bnf_dir=param_bnf${unsup_string}
datadir=$data_bnf_dir/${dirid}    

[ ! -d data/${dirid} ] && echo "No such directory data/${dirid}" && exit 1;
[ ! -d exp/tri5/decode_${dirid} ] && echo "No such directory exp/tri5/decode_${dirid}" && exit 1;

# Set my_nj; typically 64.
my_nj=`cat exp/tri5/decode_${dirid}/num_jobs` || exit 1;


if [ ! $data_bnf_dir/${dirid}_bnf/.done -nt exp/tri5/decode_${dirid}/.done ] || \
   [ ! $data_bnf_dir/${dirid}_bnf/.done -nt $exp_dir/tri6_bnf/.done ]; then
  # put the archives in $param_bnf_dir/.
  steps/nnet2/dump_bottleneck_features.sh --nj $my_nj --cmd "$train_cmd" \
    --transform-dir exp/tri5/decode_${dirid} data/${dirid} $data_bnf_dir/${dirid}_bnf $exp_dir/tri6_bnf $param_bnf_dir $exp_dir/dump_bnf
  touch $data_bnf_dir/${dirid}_bnf/.done
fi

if [ ! $data_bnf_dir/${dirid}/.done -nt $data_bnf_dir/${dirid}_bnf/.done ]; then
  steps/nnet/make_fmllr_feats.sh --cmd "$train_cmd -tc 10" \
    --nj $train_nj --transform-dir exp/tri5/decode_${dirid} $data_bnf_dir/${dirid}_sat data/${dirid} \
    exp/tri5_ali $exp_dir/make_fmllr_feats/log $param_bnf_dir/ 

  steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
    $data_bnf_dir/${dirid}_bnf $data_bnf_dir/${dirid}_sat $data_bnf_dir/${dirid} \
    $exp_dir/append_feats/log $param_bnf_dir/ 
  steps/compute_cmvn_stats.sh --fake $data_bnf_dir/${dirid} $exp_dir/make_fmllr_feats $param_bnf_dir
  rm -r $data_bnf_dir/${dirid}_sat
  if ! $skip_kws ; then
    cp -r data/${dirid}/*kws* $data_bnf_dir/${dirid}/ || true
  fi
  touch $data_bnf_dir/${dirid}/.done
fi
if ! $skip_kws ; then
  rm -rf $data_bnf_dir/${dirid}/*kws*
  cp -r data/${dirid}/*kws* $data_bnf_dir/${dirid}/ || true
fi


if $data_only ; then
  echo "Exiting, as data-only was requested... "
fi

####################################################################
##
## FMLLR decoding 
##
####################################################################
decode=$exp_dir/tri6/decode_${dirid}
if [ ! -f ${decode}/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Decoding with SAT models on top of bottleneck features on" `date`
  echo ---------------------------------------------------------------------
  utils/mkgraph.sh \
    data/lang $exp_dir/tri6 $exp_dir/tri6/graph |tee $exp_dir/tri6/mkgraph.log

  mkdir -p $decode
  #By default, we do not care about the lattices for this step -- we just want the transforms
  #Therefore, we will reduce the beam sizes, to reduce the decoding times
  steps/decode_fmllr_extra.sh --skip-scoring true --beam 10 --lattice-beam 4 \
    --acwt $bnf_decode_acwt \
    --nj $my_nj --cmd "$decode_cmd" "${decode_extra_opts[@]}"\
    $exp_dir/tri6/graph ${datadir} ${decode} |tee ${decode}/decode.log
  touch ${decode}/.done
fi

if ! $fast_path ; then
  local/run_kws_stt_task.sh --cer $cer --max-states $max_states --skip-scoring $skip_scoring\
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --extra-kws $extra_kws --wip $wip\
    "${shadow_set_extra_opts[@]}" "${lmwt_bnf_extra_opts[@]}" \
    ${datadir} data/lang ${decode}

  local/run_kws_stt_task.sh --cer $cer --max-states $max_states --skip-scoring $skip_scoring\
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --extra-kws $extra_kws --wip $wip \
    "${shadow_set_extra_opts[@]}" "${lmwt_bnf_extra_opts[@]}" \
    ${datadir} data/lang  ${decode}.si
fi

####################################################################
## SGMM2 decoding 
####################################################################
decode=$exp_dir/sgmm7/decode_fmllr_${dirid}
if [ ! -f $decode/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Spawning $decode on" `date`
  echo ---------------------------------------------------------------------
  utils/mkgraph.sh \
    data/lang $exp_dir/sgmm7 $exp_dir/sgmm7/graph |tee $exp_dir/sgmm7/mkgraph.log

  mkdir -p $decode
  steps/decode_sgmm2.sh --skip-scoring true --use-fmllr true --nj $my_nj \
    --acwt $bnf_decode_acwt \
    --cmd "$decode_cmd" --transform-dir $exp_dir/tri6/decode_${dirid} "${decode_extra_opts[@]}"\
    $exp_dir/sgmm7/graph ${datadir} $decode |tee $decode/decode.log
  touch $decode/.done
fi

if ! $fast_path ; then
  local/run_kws_stt_task.sh --cer $cer --max-states $max_states --skip-scoring $skip_scoring \
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --extra-kws $extra_kws --wip $wip \
    "${shadow_set_extra_opts[@]}" "${lmwt_bnf_extra_opts[@]}" \
    ${datadir} data/lang  $exp_dir/sgmm7/decode_fmllr_${dirid}
fi

####################################################################
##
## SGMM_MMI rescoring
##
####################################################################

for iter in 1 2 3 4; do
  # Decode SGMM+MMI (via rescoring).
  decode=$exp_dir/sgmm7_mmi_b0.1/decode_fmllr_${dirid}_it$iter
  if [ ! -f $decode/.done ]; then

    mkdir -p $decode
    steps/decode_sgmm2_rescore.sh  --skip-scoring true \
      --cmd "$decode_cmd" --iter $iter --transform-dir $exp_dir/tri6/decode_${dirid} \
      data/lang ${datadir} $exp_dir/sgmm7/decode_fmllr_${dirid} $decode | tee ${decode}/decode.log

    touch $decode/.done
  fi
done

#We are done -- all lattices has been generated. We have to
#a)Run MBR decoding
#b)Run KW search
for iter in 1 2 3 4; do
  # Decode SGMM+MMI (via rescoring).
  decode=$exp_dir/sgmm7_mmi_b0.1/decode_fmllr_${dirid}_it$iter
  local/run_kws_stt_task.sh --cer $cer --max-states $max_states --skip-scoring $skip_scoring\
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --extra-kws $extra_kws --wip $wip \
    "${shadow_set_extra_opts[@]}" "${lmwt_bnf_extra_opts[@]}" \
    ${datadir} data/lang $decode
done


if [ -f $exp_dir/tri7_nnet/.done ] && 
    [[ ( ! $exp_dir/tri7_nnet/decode_${dirid}/.done -nt $datadir/.done)  || \
       (! $exp_dir/tri7_nnet/decode_${dirid}/.done -nt $exp_dir/tri7_nnet/.done ) ]]; then
  
  echo ---------------------------------------------------------------------
  echo "Decoding hybrid system on top of bottleneck features on" `date`
  echo ---------------------------------------------------------------------

  # We use the graph from tri6.
  utils/mkgraph.sh \
    data/lang $exp_dir/tri6 $exp_dir/tri6/graph |tee $exp_dir/tri6/mkgraph.log

  decode=$exp_dir/tri7_nnet/decode_${dirid}
  if [ ! -f $decode/.done ]; then
    mkdir -p $decode
    steps/nnet2/decode.sh --cmd "$decode_cmd" --nj $my_nj \
      --acwt $bnf_decode_acwt \
      --beam $dnn_beam --lattice-beam $dnn_lat_beam \
      --skip-scoring true "${decode_extra_opts[@]}" \
      --feat-type raw \
      $exp_dir/tri6/graph ${datadir} $decode | tee $decode/decode.log

    touch $decode/.done
  fi

  local/run_kws_stt_task.sh --cer $cer --max-states $max_states --skip-scoring $skip_scoring\
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --extra-kws $extra_kws --wip $wip \
    "${shadow_set_extra_opts[@]}" "${lmwt_bnf_extra_opts[@]}" \
    ${datadir} data/lang $decode
fi

echo "$0: Everything looking good...." 
exit 0
