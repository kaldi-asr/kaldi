#!/bin/bash

# This script builds the SGMM system on top of the kaldi internal bottleneck features.

. ./cmd.sh

set -e 
set -o pipefail
set -u

# Set my_nj; typically 64.   
numLeaves=2500
numGauss=15000
numLeavesSGMM=10000
bnf_num_gauss_ubm=600
bnf_num_gauss_sgmm=7000
align_dir=exp/tri4b_ali_si284
bnf_decode_acwt=0.0357
sgmm_group_extra_opts=(--group 3 --cmd "queue.pl -l arch=*64 --mem 7G")

if [ ! -d exp_bnf ]; then
  echo "$0: before running this script, please run local/run_bnf.sh"
  exit 1;
fi

echo ---------------------------------------------------------------------
echo "Starting exp_bnf/tri5 on" `date`
echo ---------------------------------------------------------------------
if [ ! exp_bnf/tri5/.done -nt data_bnf/train/.done ]; then
  steps/train_lda_mllt.sh --splice-opts "--left-context=1 --right-context=1" \
    --dim 60  --cmd "$train_cmd" \
    $numLeaves $numGauss data_bnf/train data/lang $align_dir exp_bnf/tri5 ;
  touch exp_bnf/tri5/.done
fi

echo ---------------------------------------------------------------------
echo "Starting exp_bnf/tri6 on" `date`
echo ---------------------------------------------------------------------
if [ ! exp_bnf/tri6/.done -nt exp_bnf/tri5/.done ]; then
  steps/train_sat.sh  --cmd "$train_cmd" \
    $numLeaves $numGauss data_bnf/train data/lang exp_bnf/tri5 exp_bnf/tri6
  touch exp_bnf/tri6/.done
fi
echo ---------------------------------------------------------------------
echo "Decoding with SAT models on top of bottleneck features on" `date`
echo ---------------------------------------------------------------------
decode1=exp_bnf/tri6/decode_bd_tgpr_eval92
decode2=exp_bnf/tri6/decode_bd_tgpr_dev93
utils/mkgraph.sh \
  data/lang_test_bd_tgpr exp_bnf/tri6 exp_bnf/tri6/graph_bd_tgpr |tee exp_bnf/tri6/mkgraph.log

mkdir -p $decode1 $decode2
#By default, we do not care about the lattices for this step -- we just want the transforms
#Therefore, we will reduce the beam sizes, to reduce the decoding times
steps/decode_fmllr_extra.sh --skip-scoring true --beam 10 --lattice-beam 4 \
  --acwt $bnf_decode_acwt \
  exp_bnf/tri6/graph_bd_tgpr data_bnf/eval92 ${decode1} |tee ${decode1}/decode.log
steps/decode_fmllr_extra.sh --skip-scoring true --beam 10 --lattice-beam 4 \
  --acwt $bnf_decode_acwt \
  exp_bnf/tri6/graph_bd_tgpr data_bnf/dev93 ${decode2} |tee ${decode2}/decode.log

echo ---------------------------------------------------------------------
echo "Starting exp_bnf/ubm7 on" `date`
echo ---------------------------------------------------------------------
if [ ! exp_bnf/ubm7/.done -nt exp_bnf/tri6/.done ]; then
  steps/train_ubm.sh \
    $bnf_num_gauss_ubm data_bnf/train data/lang exp_bnf/tri6 exp_bnf/ubm7 
  touch exp_bnf/ubm7/.done
fi

if [ ! exp_bnf/sgmm7/.done -nt exp_bnf/ubm7/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp_bnf/sgmm7 on" `date`
  echo ---------------------------------------------------------------------
  steps/train_sgmm2_group.sh \
    "${sgmm_group_extra_opts[@]}"\
    $numLeavesSGMM $bnf_num_gauss_sgmm data_bnf/train data/lang \
    exp_bnf/tri6 exp_bnf/ubm7/final.ubm exp_bnf/sgmm7 
  touch exp_bnf/sgmm7/.done
fi

## SGMM2 decoding 
decode1=exp_bnf/sgmm7/decode_bd_tgpr_eval92
decode2=exp_bnf/sgmm7/decode_bd_tgpr_dev93
  echo ---------------------------------------------------------------------
  echo "Spawning $decode1 and $decode2 on" `date`
  echo ---------------------------------------------------------------------
  utils/mkgraph.sh \
    data/lang_test_bd_tgpr exp_bnf/sgmm7 exp_bnf/sgmm7/graph_bd_tgpr |tee exp_bnf/sgmm7/mkgraph.log

  mkdir -p $decode1 $decode2
  steps/decode_sgmm2.sh --skip-scoring false --use-fmllr true \
    --acwt $bnf_decode_acwt --scoring-opts "--min-lmwt 20 --max-lmwt 40"  --cmd "$decode_cmd" \
    --transform-dir exp_bnf/tri6/decode_bd_tgpr_eval92 \
    exp_bnf/sgmm7/graph_bd_tgpr data_bnf/eval92 $decode1 |tee $decode1/decode.log
  steps/decode_sgmm2.sh --skip-scoring false --use-fmllr true \
    --acwt $bnf_decode_acwt --scoring-opts "--min-lmwt 20 --max-lmwt 40"  --cmd "$decode_cmd" \
    --transform-dir exp_bnf/tri6/decode_bd_tgpr_dev93 \
    exp_bnf/sgmm7/graph_bd_tgpr data_bnf/dev93 $decode2 |tee $decode2/decode.log

if [ ! exp_bnf/sgmm7_ali/.done -nt exp_bnf/sgmm7/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp_bnf/sgmm7_ali on" `date`
  echo ---------------------------------------------------------------------
  steps/align_sgmm2.sh \
    --transform-dir exp_bnf/tri6 --nj 30 --use-graphs true \
    data_bnf/train data/lang exp_bnf/sgmm7 exp_bnf/sgmm7_ali 
  touch exp_bnf/sgmm7_ali/.done
fi

if [ ! exp_bnf/sgmm7_denlats/.done -nt exp_bnf/sgmm7/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp_bnf/sgmm5_denlats on" `date`
  echo ---------------------------------------------------------------------
  steps/make_denlats_sgmm2.sh \
     "${sgmm_denlats_extra_opts[@]}" \
    --transform-dir exp_bnf/tri6 --nj 30 --beam 14.0 --acwt $bnf_decode_acwt --lattice-beam 8 \
     data_bnf/train data/lang exp_bnf/sgmm7_ali exp_bnf/sgmm7_denlats 
  touch exp_bnf/sgmm7_denlats/.done
fi

if [ ! exp_bnf/sgmm7_mmi_b0.1/.done -nt exp_bnf/sgmm7_denlats/.done ]; then
  steps/train_mmi_sgmm2.sh \
    --acwt $bnf_decode_acwt \
    --transform-dir exp_bnf/tri6 --boost 0.1 --drop-frames true \
    data_bnf/train data/lang exp_bnf/sgmm7_ali exp_bnf/sgmm7_denlats \
    exp_bnf/sgmm7_mmi_b0.1 
  touch exp_bnf/sgmm7_mmi_b0.1/.done;
fi

## SGMM_MMI rescoring
for iter in 1 2 3 4; do
  # Decode SGMM+MMI (via rescoring).
  decode1=exp_bnf/sgmm7_mmi_b0.1/decode_bd_tgpr_eval92_it$iter
  mkdir -p $decode1
  steps/decode_sgmm2_rescore.sh  --skip-scoring false --cmd "$decode_cmd" \
    --iter $iter --transform-dir exp_bnf/tri6/decode_bd_tgpr_eval92 --scoring-opts "--min-lmwt 20 --max-lmwt 40" \
  data/lang_test_bd_tgpr data_bnf/eval92 exp_bnf/sgmm7/decode_bd_tgpr_eval92 $decode1 | tee ${decode1}/decode.log
done

for iter in 1 2 3 4; do
  # Decode SGMM+MMI (via rescoring).
  decode2=exp_bnf/sgmm7_mmi_b0.1/decode_bd_tgpr_dev93_it$iter  
  mkdir -p $decode2
  steps/decode_sgmm2_rescore.sh  --skip-scoring false --cmd "$decode_cmd" \
    --iter $iter --transform-dir exp_bnf/tri6/decode_bd_tgpr_dev93 --scoring-opts "--min-lmwt 20 --max-lmwt 40" \
  data/lang_test_bd_tgpr data_bnf/dev93 exp_bnf/sgmm7/decode_bd_tgpr_dev93 $decode2 | tee ${decode2}/decode.log
done

echo ---------------------------------------------------------------------
echo "Finished successfully on" `date`
echo ---------------------------------------------------------------------

#exit 1
