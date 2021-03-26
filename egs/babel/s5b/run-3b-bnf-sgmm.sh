#!/usr/bin/env bash

# Copyright 2014  Pegah Ghahremani
#           2014  Johns Hopkins (Yenda Trmal)

# Apache 2.0

# This script builds the SGMM system on top of the kaldi internal bottleneck features.
# It comes after run-6-bnf-semisupervised.sh.


. conf/common_vars.sh
. ./lang.conf
[ -f local.conf ] && . ./local.conf

set -e 
set -o pipefail
set -u
semisupervised=true
unsup_string=

. ./utils/parse_options.sh

if [ $babel_type == "full" ] && $semisupervised; then
  echo "Error: Using unsupervised training for fullLP is meaningless, use semisupervised=false "
  exit 1
fi

if [ -z "$unsup_string" ]; then
  if $semisupervised ; then
    unsup_string="_semisup"
  else
    unsup_string=""  #" ": supervised training, _semi_supervised: unsupervised BNF training
  fi
fi
exp_dir=exp_bnf${unsup_string}
data_bnf_dir=data_bnf${unsup_string}
param_bnf_dir=param_bnf${unsup_string}

echo ---------------------------------------------------------------------
echo "Starting $exp_dir/ubm7 on" `date`
echo ---------------------------------------------------------------------
if [ ! $exp_dir/ubm7/.done -nt $exp_dir/tri6/.done ]; then
  steps/train_ubm.sh --cmd "$train_cmd" \
    $bnf_num_gauss_ubm $data_bnf_dir/train data/lang $exp_dir/tri6 $exp_dir/ubm7 
  touch $exp_dir/ubm7/.done
fi

if [ ! $exp_dir/sgmm7/.done -nt $exp_dir/ubm7/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting $exp_dir/sgmm7 on" `date`
  echo ---------------------------------------------------------------------
  #steps/train_sgmm2_group.sh \
  steps/train_sgmm2.sh \
    --cmd "$train_cmd" "${sgmm_train_extra_opts[@]}"\
    $numLeavesSGMM $bnf_num_gauss_sgmm $data_bnf_dir/train data/lang \
    $exp_dir/tri6 $exp_dir/ubm7/final.ubm $exp_dir/sgmm7 
  touch $exp_dir/sgmm7/.done
fi

if [ ! $exp_dir/sgmm7_ali/.done -nt $exp_dir/sgmm7/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting $exp_dir/sgmm7_ali on" `date`
  echo ---------------------------------------------------------------------
  steps/align_sgmm2.sh \
     --nj $train_nj --cmd "$train_cmd" --transform-dir $exp_dir/tri6 --use-graphs true \
    $data_bnf_dir/train data/lang $exp_dir/sgmm7 $exp_dir/sgmm7_ali 
  touch $exp_dir/sgmm7_ali/.done
fi

if [ ! $exp_dir/sgmm7_denlats/.done -nt $exp_dir/sgmm7/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting $exp_dir/sgmm5_denlats on" `date`
  echo ---------------------------------------------------------------------
  steps/make_denlats_sgmm2.sh \
    --nj $train_nj --sub-split $train_nj "${sgmm_denlats_extra_opts[@]}" \
    --transform-dir $exp_dir/tri6 --beam 10.0 --acwt 0.06 --lattice-beam 6 \
     $data_bnf_dir/train data/lang $exp_dir/sgmm7_ali $exp_dir/sgmm7_denlats 
  touch $exp_dir/sgmm7_denlats/.done
fi

if [ ! $exp_dir/sgmm7_mmi_b0.1/.done -nt $exp_dir/sgmm7_denlats/.done ]; then
  steps/train_mmi_sgmm2.sh \
    --cmd "$train_cmd" --acwt 0.06 \
    --transform-dir $exp_dir/tri6 --boost 0.1 --drop-frames true \
    $data_bnf_dir/train data/lang $exp_dir/sgmm7_ali $exp_dir/sgmm7_denlats \
    $exp_dir/sgmm7_mmi_b0.1 
  touch $exp_dir/sgmm7_mmi_b0.1/.done;
fi


echo ---------------------------------------------------------------------
echo "Finished successfully on" `date`
echo "To decode a data-set, use run-4b-anydecode-bnf.sh"
echo ---------------------------------------------------------------------

exit 0
