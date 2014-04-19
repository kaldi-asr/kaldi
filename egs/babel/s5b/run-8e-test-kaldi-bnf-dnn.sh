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

if ! echo {dev10h,dev2h,eval}{,.uem} | grep -w "$type" >/dev/null; then
  # note: echo dev10.uem | grep -w dev10h will produce a match, but this
  # doesn't matter because dev10h is also a valid value.
  echo "Invalid variable type=${type}, valid values are " {dev10h,dev2h,eval}{,.uem}
  exit 1;
fi

dirid=${type}
datadir=data_bnf/${dirid}

[ ! -d data/${dirid} ] && echo "No such directory data/${dirid}" && exit 1;
[ ! -d exp/tri5/decode_${dirid} ] && echo "No such directory exp/tri5/decode_${dirid}" && exit 1;

# Set my_nj; typically 64.
my_nj=`cat exp/tri5/decode_${dirid}/num_jobs` || exit 1;


if [ ! data_bnf/${dirid}_bnf/.done -nt exp/tri5/decode_${dirid}/.done ] || \
   [ ! data_bnf/${dirid}_bnf/.done -nt exp_bnf/tri6_bnf/.done ]; then
  # put the archives in plp/.
  steps/nnet2/dump_bottleneck_features.sh --nj $my_nj --cmd "$train_cmd" \
    --transform-dir exp/tri5/decode_${dirid} data/${dirid} data_bnf/${dirid}_bnf exp_bnf/tri6_bnf param_bnf exp_bnf/dump_bnf
  touch data_bnf/${dirid}_bnf/.done
fi

if [ ! data_bnf/${dirid}/.done -nt data_bnf/${dirid}_bnf/.done ]; then
  steps/make_fmllr_feats.sh --cmd "$train_cmd -tc 10" \
    --nj $train_nj --transform-dir exp/tri5/decode_${dirid} data_bnf/${dirid}_sat data/${dirid} \
    exp/tri5_ali exp_bnf/make_fmllr_feats/log param_bnf

  steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
    data_bnf/${dirid}_bnf data_bnf/${dirid}_sat data_bnf/${dirid} \
    exp_bnf/append_feats/log param_bnf
  steps/compute_cmvn_stats.sh --fake data_bnf/${dirid} exp_bnf/make_fmllr_feats param_bnf
  rm -r data_bnf/${dirid}_sat
  cp -r data/${dirid}/kws* data_bnf/${dirid}/
  touch data_bnf/${dirid}/.done
fi

if [ ! exp_bnf/tri7_nnet/decode_${dirid}/.done -nt data_bnf/${dirid}_bnf/.done ] || \
   [ ! exp_bnf/tri7_nnet/decode_${dirid}/.done -nt exp_bnf/tri7_nnet/.done ]; then
  
  echo ---------------------------------------------------------------------
  echo "Decoding hybrid system on top of bottleneck features on" `date`
  echo ---------------------------------------------------------------------

  # We use the graph from tri6.
  utils/mkgraph.sh \
    data/lang exp_bnf/tri6 exp_bnf/tri6/graph |tee exp_bnf/tri6/mkgraph.log

  decode=exp_bnf/tri7_nnet/decode_${dirid}
  if [ ! -f $decode/.done ]; then
    mkdir -p $decode
    steps/nnet2/decode.sh \
      --minimize $minimize --cmd "$decode_cmd" --nj $my_nj \
      --acwt $bnf_decode_acwt \
      --beam $dnn_beam --lat-beam $dnn_lat_beam \
      --skip-scoring true "${decode_extra_opts[@]}" \
      --feat-type raw \
      exp_bnf/tri6/graph ${datadir} $decode | tee $decode/decode.log

    touch $decode/.done
  fi

  local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
    --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --wip $wip \
    "${shadow_set_extra_opts[@]}" "${lmwt_bnf_extra_opts[@]}" \
    ${datadir} data/lang $decode
fi

echo "$0: Everything looking good...." 
exit 0
