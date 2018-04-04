#!/bin/bash 
set -e
set -o pipefail

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;


type=dev10h
data_only=false
fast_path=true
skip_kws=false
skip_stt=false
tmpdir=`pwd`

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $(basename $0) --type (dev10h|dev2h|eval|shadow)"
  exit 1
fi

if [[ "$type" != "dev10h" && "$type" != "dev2h" && "$type" != "eval" && "$type" != "shadow" ]] ; then
  echo "Warning: invalid variable type=${type}, valid values are dev10h|dev2h|eval"
  echo "Hope you know what your ar doing!"
fi


datadir=data/app_${type}.uem
dirid=${type}.uem

[ ! -d data/${dirid} ] && echo "No such directory data/${dirid}" && exit 1;
[ ! -d exp/tri5/decode_${dirid} ] && echo "No such directory exp/tri5/decode_${dirid}" && exit 1;

# Set my_nj; typically 64.
my_nj=`cat exp/tri5/decode_${dirid}/num_jobs` || exit 1;



if [ ! -f data/bnf_${dirid}/.done ]; then
  [ ! -d data/bnf_${dirid} ] && \
    mkdir -p $tmpdir/data/bnf_${dirid} && \
    ln -s $tmpdir/data/bnf_${dirid} data/bnf_${dirid}

  steps_BNF/make_bnf_feat.sh --nj $my_nj --cmd "$decode_cmd" \
    --transform_dir exp/tri5/decode_${dirid}/ \
    data/${dirid} data/bnf_${dirid} exp_BNF/bnf_dnn \
    exp/tri5_ali exp_BNF/make_bnf_${dirid}

  touch data/bnf_${dirid}/.done
fi

if [ ! -f data/sat_${dirid}/.done ]; then
  [ ! -d data/sat_${dirid} ] && \
    mkdir -p $tmpdir/data/sat_${dirid} && \
    ln -s $tmpdir/data/sat_${dirid} data/sat_${dirid}

  steps/make_fmllr_feats.sh --cmd "$decode_cmd -tc 10" --nj $my_nj \
    --transform-dir exp/tri5/decode_${dirid}  \
    data/sat_${dirid} data/${dirid} exp/tri5 \
    exp_BNF/make_fmllr_feats_${dirid}/log plp_processed

  touch data/sat_${dirid}/.done
fi

if [ ! -f $datadir/.done ]; then
  [ ! -d ${datadir} ] && \
    mkdir -p $tmpdir/${datadir} && \
    ln -s $tmpdir/${datadir} ${datadir}

  steps/append_feats.sh --cmd "$decode_cmd" --nj 4 \
    data/bnf_${dirid} data/sat_${dirid} ${datadir} \
    exp_BNF/append_feats/log plp_processed/

  steps/compute_cmvn_stats.sh --fake \
    ${datadir} exp/make_plp/app_${dirid} plp_processed

  ln -s `pwd`/data/${dirid}/kws ${datadir}/kws
  touch ${datadir}/.done
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
decode=exp_BNF/tri6/decode_${dirid}
if [ ! -f ${decode}/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Spawning decoding with SAT models  on" `date`
  echo ---------------------------------------------------------------------
  utils/mkgraph.sh \
    data/lang exp_BNF/tri6 exp_BNF/tri6/graph |tee exp_BNF/tri6/mkgraph.log

  mkdir -p $decode
  #By default, we do not care about the lattices for this step -- we just want the transforms
  #Therefore, we will reduce the beam sizes, to reduce the decoding times
  steps/decode_fmllr_extra.sh --skip-scoring true --beam 10 --lattice-beam 4\
    --nj $my_nj --cmd "$decode_cmd" "${decode_extra_opts[@]}"\
    exp_BNF/tri6/graph ${datadir} ${decode} |tee ${decode}/decode.log
  touch ${decode}/.done
fi

if ! $fast_path ; then
  local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --wip $wip \
    "${shadow_set_extra_opts[@]}" "${lmwt_bnf_extra_opts[@]}" \
    ${datadir} data/lang ${decode}

  local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --wip $wip \
    "${shadow_set_extra_opts[@]}" "${lmwt_bnf_extra_opts[@]}" \
    ${datadir} data/lang  ${decode}.si
fi

####################################################################
## SGMM2 decoding 
####################################################################
decode=exp_BNF/sgmm7/decode_fmllr_${dirid}
if [ ! -f $decode/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Spawning $decode on" `date`
  echo ---------------------------------------------------------------------
  utils/mkgraph.sh \
    data/lang exp_BNF/sgmm7 exp_BNF/sgmm7/graph |tee exp_BNF/sgmm7/mkgraph.log

  mkdir -p $decode
  steps/decode_sgmm2.sh --skip-scoring true --use-fmllr true --nj $my_nj \
    --cmd "$decode_cmd" --transform-dir exp_BNF/tri6/decode_${dirid} "${decode_extra_opts[@]}"\
    exp_BNF/sgmm7/graph ${datadir} $decode |tee $decode/decode.log
  touch $decode/.done
fi

if ! $fast_path ; then
  local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --wip 0.5 \
    "${shadow_set_extra_opts[@]}" "${lmwt_bnf_extra_opts[@]}" \
    ${datadir} data/lang  exp/sgmm5/decode_fmllr_${dirid}
fi

####################################################################
##
## SGMM_MMI rescoring
##
####################################################################

for iter in 1 2 3 4; do
  # Decode SGMM+MMI (via rescoring).
  decode=exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_${dirid}_it$iter
  if [ ! -f $decode/.done ]; then

    mkdir -p $decode
    steps/decode_sgmm2_rescore.sh  --skip-scoring true \
      --cmd "$decode_cmd" --iter $iter --transform-dir exp_BNF/tri6/decode_${dirid} \
      data/lang ${datadir} exp_BNF/sgmm7/decode_fmllr_${dirid} $decode | tee ${decode}/decode.log

    touch $decode/.done
  fi
done

#We are done -- all lattices has been generated. We have to
#a)Run MBR decoding
#b)Run KW search
for iter in 1 2 3 4; do
  # Decode SGMM+MMI (via rescoring).
  decode=exp_BNF/sgmm7_mmi_b0.1/decode_fmllr_${dirid}_it$iter
  local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --wip 0.5 \
    "${shadow_set_extra_opts[@]}" "${lmwt_bnf_extra_opts[@]}" \
    ${datadir} data/lang $decode
done


echo "Everything looking good...." 
exit 0
