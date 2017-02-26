#!/bin/bash

# Copyright  2016 Tokyo Institute of Technology (Authors: Tomohiro Tanaka, Takafumi Moriya and Takahiro Shinozaki)
#            2016 Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
# Apache 2.0
# Acknowledgement  This work was supported by JSPS KAKENHI Grant Number 26280055. 

[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh
. cmd.sh

if [ -e data/train_dev ] ;then
    dev_set=train_dev
fi

rnnlm_ver=rnnlm-0.3e

#:<<"#SKIP"

echo h30 Begin
local/csj_train_rnnlms.sh --dict-suffix "_nosp" data/local/rnnlm.h30
sleep 20; # wait till tools compiled.

echo h100 Begin 
local/csj_train_rnnlms.sh --dict-suffix "_nosp" \
    --hidden 100 --nwords 10000 --class 200 \
    --direct 0 data/local/rnnlm.h100

echo h200 Begin
local/csj_train_rnnlms.sh --dict-suffix "_nosp" \
    --hidden 200 --nwords 10000 --class 200 \
    --direct 0 data/local/rnnlm.h200

echo h300 Begin
local/csj_train_rnnlms.sh --dict-suffix "_nosp" \
    --hidden 300 --nwords 10000 --class 200 \
    --direct 0 data/local/rnnlm.h300

echo h400 Begin
local/csj_train_rnnlms.sh --dict-suffix "_nosp" \
    --hidden 400 --nwords 10000 --class 200 \
    --direct 0 data/local/rnnlm.h400

echo h500 Begin
local/csj_train_rnnlms.sh --dict-suffix "_nosp" \
    --hidden 500 --nwords 10000 --class 200 \
    --direct 0 data/local/rnnlm.h500

#SKIP

echo Begin rescoring
sourceresult=dnn5b_pretrain-dbn_dnn_smbr_i1lats
acwt=17

for dict in rnnlm.h30 rnnlm.h100 rnnlm.h200 rnnlm.h300 rnnlm.h400 rnnlm.h500 ;do
  for eval_num in eval1 eval2 eval3 $dev_set ;do
      dir=data/local/$dict
      sourcedir=exp/${sourceresult}/decode_${eval_num}_csj
      resultsdir=${sourcedir}_${dict}

      echo "rnnlm0.5"
      steps/rnnlmrescore.sh --rnnlm_ver $rnnlm_ver \
        --N 100 --cmd "queue -l mem_free=1G" --inv-acwt $acwt 0.5 \
        data/lang_csj_tg $dir data/$eval_num $sourcedir ${resultsdir}_L0.5
      
      rm -rf ${resultsdir}_L0.25
      rm -rf ${resultsdir}_L0.75
      cp -rp ${resultsdir}_L0.5 ${resultsdir}_L0.25
      cp -rp ${resultsdir}_L0.5 ${resultsdir}_L0.75

      echo "rnnlm0.25"
      steps/rnnlmrescore.sh --rnnlm_ver $rnnlm_ver \
        --stage 7 --N 100 --cmd "$decode_cmd -l mem_free=1G" --inv-acwt $acwt 0.25 \
        data/lang_csj_tg $dir data/$eval_num $sourcedir ${resultsdir}_L0.25

      echo "rnnlm0.75"
      steps/rnnlmrescore.sh --rnnlm_ver $rnnlm_ver \
        --stage 7 --N 100 --cmd "$decode_cmd -l mem_free=1G" --inv-acwt $acwt 0.75 \
        data/lang_csj_tg $dir data/$eval_num $sourcedir ${resultsdir}_L0.75
  done
done
