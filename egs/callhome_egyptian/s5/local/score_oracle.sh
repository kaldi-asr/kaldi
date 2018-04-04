#!/usr/bin/env bash

oracle_dir=exp/sgmm2x_6a_mmi_b0.2/decode_dev_it1/oracle
split=dev
data_dir=data/dev
lang_dir=data/lang

for i in {5..20}; do
    mkdir -p $oracle_dir/score_$i
    cp $oracle_dir/$split.ctm $oracle_dir/score_$i/
done

. /export/babel/data/software/env.sh

# Start scoring
/export/a11/guoguo/babel/103-bengali-limitedLP.official/local/score_stm.sh $data_dir $lang_dir \
    $oracle_dir

# Print a summary of the result
grep "Percent Total Error" $oracle_dir/score_*/$split.ctm.dtl
