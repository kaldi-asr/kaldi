#!/bin/bash

. cmd.sh

lang_test=$1 # e.g. data/lang_test_4g
rnndir=$2 # e.g. data/local/rnntest
testdir=$3 # e.g. data/test
decode_srcdir=$4 # e.g. exp/tri3b/decode_4g_test1k
suffix=$(basename $rnndir)
#decode_rnndir= $5 # e.g. exp/tri3b/decode_4g_test1k_rnn


 # This step interpolates a small RNNLM (with weight 0.25) with the 4-gram LM.
steps/rnnlmrescore.sh \
  --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
  0.25 $lang_test $rnndir $testdir \
  $decode_srcdir ${decode_srcdir}_${suffix}_0.25  \
  || exit 1;

steps/rnnlmrescore.sh \
  --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
  0.5 $lang_test $rnndir $testdir \
  $decode_srcdir ${decode_srcdir}_${suffix}_0.5  \
  || exit 1;

steps/rnnlmrescore.sh \
  --N 1000 --cmd "$decode_cmd" --inv-acwt 17 \
  1 $lang_test $rnndir $testdir \
  $decode_srcdir ${decode_srcdir}_${suffix}_1_N1000  \
  || exit 1;

steps/rnnlmrescore.sh \
  --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
  0.75 $lang_test $rnndir $testdir \
  $decode_srcdir ${decode_srcdir}_${suffix}_0.75  \
  || exit 1;

steps/rnnlmrescore.sh \
  --N 1000 --cmd "$decode_cmd" --inv-acwt 17 \
  0.5 $lang_test $rnndir $testdir \
  $decode_srcdir ${decode_srcdir}_${suffix}_0.5_N1000  

dir=${decode_srcdir}_${suffix}_0.75_N1000
rm -rf $dir
cp -r ${decode_srcdir}_${suffix}_0.5_N1000 $dir
steps/rnnlmrescore.sh \
  --stage 7 --N 1000 --cmd "$decode_cmd" --inv-acwt 17 \
  0.75 $lang_test $rnndir $testdir \
  $decode_srcdir $dir

## Different in no. of neurons

#dir=exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm300_0.75
#rm -rf $dir
#cp -r exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm300_0.5 $dir
#steps/rnnlmrescore.sh \
#  --stage 7 --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
#  0.75 data/lang_test_bd_fg data/local/rnnlm.h300.voc40k data/test_eval92 \
#  exp/tri3b/decode_bd_tgpr_eval92_fg $dir

#dir=exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm300_0.25
#rm -rf $dir
#cp -r exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm300_0.5 $dir
#steps/rnnlmrescore.sh \
#  --stage 7 --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
#  0.25 data/lang_test_bd_fg data/local/rnnlm.h300.voc40k data/test_eval92 \
#  exp/tri3b/decode_bd_tgpr_eval92_fg $dir

steps/rnnlmrescore.sh \
  --N 10 --cmd "$decode_cmd" --inv-acwt 17 \
  0.5 $lang_test $rnndir $testdir \
  $decode_srcdir ${decode_srcdir}_${suffix}_0.5_N10 \
  || exit 1;

