#!/usr/bin/env bash
# 2020 Dongji Gao

stage=0

[ -f ./path.sh ] && . ./path.sh
. ./path.sh
. ./cmd.sh
. utils/parse_options.sh

sclite="$KALDI_ROOT/tools/sctk/bin/sclite"
text="data/safe_t_dev1/text"
segments="data/safe_t_dev1/segments"
stm="tmp/stm"
datadir=data/safe_t_dev1
dir=exp/ihm/chain_1a/tdnn_b_bigger_2_aug/
graph=${dir}/graph_3
decode_dir=${dir}/decode_safe_t_dev1_whole

## get stm file
if [ $stage -le 0 ]; then
  ./local/get_stm.py ${text} ${segments} ${stm}
fi

# get ctm file
if [ $stage -le 1 ]; then
  steps/get_ctm_fast.sh --lmwt 8 --cmd "$train_cmd" --frame-shift 0.03 ${datadir}  $graph $decode_dir tmp/
fi

# sclite scoring
if [ $stage -le 2 ]; then
    ${sclite} -f 0 -s -m hyp -r tmp/stm stm -h tmp/ctm ctm > tmp/sclite_scoring_result_orig
fi
