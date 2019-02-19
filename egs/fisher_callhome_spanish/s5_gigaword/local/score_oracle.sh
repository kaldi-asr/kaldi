#!/usr/bin/env bash
# Copyright 2014  Gaurav Kumar.   Apache 2.0

oracle_dir=exp/tri5a/decode_callhome_test/oracle
split=callhome_test
data_dir=data/callhome_test
lang_dir=data/lang

# Make sure that your STM and CTM files are in UTF-8 encoding
# Any other encoding will cause this script to fail/misbehave

if [ ! -e $oracle_dir -o ! -e $data_dir -o ! -e $lang_dir ]; then
  echo "Missing pre-requisites"
  exit 1
fi

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
