#!/bin/bash

# Copyright 2015 Hossein Hadian

log_file=
cmd=queue.pl
sclite=false

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [[ $# != 4 ]]; then
  echo "Usage: $0 [--log-file <log-file>] <nnet-dur-model> <test-dir> <graph|lang> <decode-dir> "
  exit 1;
fi

durmod=$1
testdir=$2
graphdir=$3
decodedir=$4

if [ -z $log_file ]; then
  log_file=$(basename $(dirname $durmod))"_"$(basename $durmod '.mdl')"_"$(basename $decodedir)".log" || exit 1;
  echo $log_file
fi

echo "Scoring $durmod" >$log_file
echo "" >>$log_file
nnet3-durmodel-info $durmod 2>/dev/null >>$log_file
echo "" >>$log_file

echo "original decode-dir best word error rate:" >>$log_file
grep WER $decodedir/wer* 2>/dev/null | utils/best_wer.sh | sed "s/WER/baseline/g" >>$log_file
grep Sum $decodedir/score*/*ys 2>/dev/null | utils/best_wer.sh | sed "s/WER/baseline/g" >>$log_file

echo "" >>$log_file

srcdir=$(dirname $decodedir)
durmod_dir_name=$(basename $(dirname $durmod))
rescored_decode_dir=$srcdir"/"$durmod_dir_name"_"$(basename $decodedir)"_dm_rescored"
echo "Rescore dir:" $rescored_decode_dir

scales="0.1 0.3 0.5 0.7 0.8 1.0 1.2"

for scale in $scales; do
  echo "scale: $scale" >>$log_file
  echo -n "(sc: $scale, WER: "
  steps/dur_model/rescore.sh --cmd $cmd --duration-model-scale $scale $durmod $decodedir $rescored_decode_dir || exit 1;
  local/score.sh --cmd $cmd $testdir $graphdir $rescored_decode_dir >/dev/null 2>>$log_file || exit 1;
  if $sclite; then
    wer_string=$(grep Sum $rescored_decode_dir/score*/*ys | utils/best_wer.sh)
  else
    wer_string=$(grep WER $rescored_decode_dir/wer* 2>/dev/null | utils/best_wer.sh)
  fi
  echo $wer_string >>$log_file
  echo -n $wer_string | awk '{ printf $2 }'
  echo -n ")"
  echo "" >>$log_file
done
echo ""

allWERs="WERs=[ "$(grep WER $log_file | awk '{printf "%s ", $2}')" ];"
echo $allWERs >>$log_file
best_wer_scale=$(grep WERs $log_file | awk -v scales="$scales" 'BEGIN {best_wer=200} { for(i=2;i<NF;i++) if($i<best_wer){best_wer=$i; best_i=i;}} END {split(scales, a, " "); print best_wer, a[best_i-1];}')
best_wer=$(echo $best_wer_scale | awk '{print $1}')
best_scale=$(echo $best_wer_scale | awk '{print $2}')

echo "Best duration-model WER for $durmod on $decodedir: $best_wer (with scale = $best_scale)" >>$log_file
echo "Best duration-model WER for $durmod on $decodedir: $best_wer (with scale = $best_scale)"

