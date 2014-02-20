#!/bin/bash

# This script builds the SGMM system on top of the kaldi internal bottleneck features.
# It comes after run-kaldi-bnf-dnn.sh.


. conf/common_vars.sh
. ./lang.conf
[ -f local.conf ] && . ./local.conf

set -e 
set -o pipefail
set -u


echo ---------------------------------------------------------------------
echo "Starting exp_bnf/ubm7 on" `date`
echo ---------------------------------------------------------------------
if [ ! exp_bnf/ubm7/.done -nt exp_bnf/tri6/.done ]; then
  steps/train_ubm.sh --cmd "$train_cmd" \
    $bnf_num_gauss_ubm data_bnf/train data/lang exp_bnf/tri6 exp_bnf/ubm7 
  touch exp_bnf/ubm7/.done
fi

if [ ! exp_bnf/sgmm7/.done -nt exp_bnf/ubm7/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp_bnf/sgmm7 on" `date`
  echo ---------------------------------------------------------------------
  steps/train_sgmm2.sh \
    --cmd "$train_cmd" \
    $numLeavesSGMM $bnf_num_gauss_sgmm data_bnf/train data/lang \
    exp_bnf/tri6 exp_bnf/ubm7/final.ubm exp_bnf/sgmm7 
  touch exp_bnf/sgmm7/.done
fi

if [ ! exp_bnf/sgmm7_ali/.done -nt exp_bnf/sgmm7/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp_bnf/sgmm7_ali on" `date`
  echo ---------------------------------------------------------------------
  steps/align_sgmm2.sh \
     --nj $train_nj --cmd "$train_cmd" --transform-dir exp_bnf/tri6 --use-graphs true \
    data_bnf/train data/lang exp_bnf/sgmm7 exp_bnf/sgmm7_ali 
  touch exp_bnf/sgmm7_ali/.done
fi

if [ ! exp_bnf/sgmm7_denlats/.done -nt exp_bnf/sgmm7/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp_bnf/sgmm5_denlats on" `date`
  echo ---------------------------------------------------------------------
  steps/make_denlats_sgmm2.sh \
    --nj $train_nj --sub-split $train_nj "${sgmm_denlats_extra_opts[@]}" \
    --transform-dir exp_bnf/tri6 --beam 10.0 --acwt 0.06 --lattice-beam 6 \
     data_bnf/train data/lang exp_bnf/sgmm7_ali exp_bnf/sgmm7_denlats 
  touch exp_bnf/sgmm7_denlats/.done
fi

if [ ! exp_bnf/sgmm7_mmi_b0.1/.done -nt exp_bnf/sgmm7_denlats/.done ]; then
  steps/train_mmi_sgmm2.sh \
    --cmd "$train_cmd" --acwt 0.06 \
    --transform-dir exp_bnf/tri6 --boost 0.1 --drop-frames true \
    data_bnf/train data/lang exp_bnf/sgmm7_ali exp_bnf/sgmm7_denlats \
    exp_bnf/sgmm7_mmi_b0.1 
  touch exp_bnf/sgmm7_mmi_b0.1/.done;
fi


echo ---------------------------------------------------------------------
echo "Finished successfully on" `date`
echo ---------------------------------------------------------------------

exit 0
