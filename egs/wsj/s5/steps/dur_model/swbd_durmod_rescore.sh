#!/bin/bash

# Copyright 2015 Hossein Hadian
# Apache 2.0.

log_file=
cmd=queue.pl
scales="0.2 0.3 0.5"
kaldi_score=true
rescored_decode_dir=
wip=0.0
subtract_avg_logprobs=true

cmdline="$0 $@"

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
echo "Scoring $durmod...Log file: $log_file"


echo "Scoring $durmod" >$log_file

echo "" >>$log_file
echo "Options:" >>$log_file

echo "subtract_avg_logprobs: "$subtract_avg_logprobs >>$log_file
echo "wip: "$wip >>$log_file
echo "kaldi_score: "$kaldi_score >>$log_file

echo "" >>$log_file
echo "last train cmdline: " >>$log_file
cat $(dirname $durmod)/log/cmd.log >>$log_file

echo "" >>$log_file
echo "final train objective (pertaining to final_nnet_durmodel.mdl): " >>$log_file
cat $(dirname $durmod)/log/final_objectives.log >>$log_file


echo "" >>$log_file
echo "cmdline: " >>$log_file
echo $cmdline >>$log_file


echo "" >>$log_file
nnet3-durmodel-info $durmod 2>/dev/null >>$log_file
echo "" >>$log_file

echo "original decode-dir best word error rate:" >>$log_file
grep WER $decodedir/wer* 2>/dev/null | utils/best_wer.sh | sed "s/WER/baseline/g" >>$log_file
grep Sum $decodedir/*score*/*.ctm.filt.sys 2>/dev/null | utils/best_wer.sh | sed "s/WER/baseline/g" >>$log_file

echo "" >>$log_file

srcdir=$(dirname $decodedir)
durmod_dir_name=$(basename $(dirname $durmod))

[ -z $rescored_decode_dir ] && rescored_decode_dir=$srcdir"/"$durmod_dir_name"_"$(basename $decodedir)"_dm_rescored"
echo "Rescore dir:" $rescored_decode_dir
rm -rf $rescored_decode_dir
echo "Deleted the contents of rescored-dir."

if $kaldi_score; then
  score_cmd="steps/score_kaldi.sh --word-ins-penalty $wip"
else
  score_cmd="local/score.sh --word-ins-penalty $wip"
fi

echo "Score cmd: "$score_cmd
echo "Score cmd: "$score_cmd >>$log_file

avg_logprobs_opts=
if $subtract_avg_logprobs; then
  #tmp# avg_logprobs_opts="--avg-logprobs-file "$(dirname $durmod)"/avg_logprobs.data"
  avg_logprobs_opts="--avg-logprobs-file "$(dirname $durmod)"/priors.vec"
fi

echo "Scales: "$scales
for scale in $scales; do
  echo "scale: $scale" >>$log_file
  echo -n "(sc: $scale, WER:"
  steps/dur_model/rescore.sh --cmd $cmd $avg_logprobs_opts --duration-model-scale $scale $durmod $decodedir $rescored_decode_dir || exit 1;
  echo -n "+"
  cp $decodedir/num_jobs $rescored_decode_dir/num_jobs
  $score_cmd --cmd $cmd $testdir $graphdir $rescored_decode_dir >/dev/null 2>>$log_file || exit 1;
  echo -n "+"
  if ls $rescored_decode_dir/wer* 2>1 >/dev/null; then
    wer_string=$(grep WER $rescored_decode_dir/wer* 2>/dev/null | utils/best_wer.sh)
  else
    wer_string=$(grep Sum $rescored_decode_dir/*score*/*.ctm.filt.sys | utils/best_wer.sh)
  fi
  echo -n "+ "
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

