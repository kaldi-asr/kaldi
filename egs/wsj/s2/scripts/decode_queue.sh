#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# Decode the testing data, using the "qsub" command.

. path.sh || exit 1;

acwt=0.0625
beam=13.0
max_active=7000
cmn=false
qsub_opts="-l ram_free=1200M,mem_free=1200M"

for n in 1 2 3 4; do
   if [ "$1" == "--acwt" ]; then
      shift;
      acwt=$1;
      shift;
   fi
   if [ "$1" == "--beam" ]; then
      shift;
      beam=$1;
      shift;
   fi
   if [ "$1" == "--max-active" ]; then
      shift;
      max_active=$1;
      shift;
   fi
   if [ $1 == "--cmn" ]; then
      cmn=true;
      shift;
   fi
   if [ $1 == "--qsub-opts" ]; then
      shift; 
      qsub_opts=$1
      shift;
   fi
done

if [ $# != 3 ]; then
   echo "Usage: scripts/decode.sh <graphdir> <model> <decode-dir>"
   exit 1;
fi


graphdir=$1
model=$2
dir=$3
# Make "dir" an absolute pathname.
dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

mkdir -p $dir
mkdir -p $dir/qlog

queue=all.q@@blade

# Use 10 cores to decode

scripts/split_scp.pl data/test.scp $dir/test{0,1,2,3,4,5,6,7,8,9}.scp

for x in 0 1 2 3 4 5 6 7 8 9; do
  if [ $cmn == "true" ]; then
    feats="ark:add-deltas --print-args=false scp:$dir/test${x}.scp ark:- | remove-mean ark:- ark:- |"
  else
    feats="ark:add-deltas --print-args=false scp:$dir/test${x}.scp ark:- |"
  fi
  scriptfile=$dir/decode${x}.sh
 ( echo '#!/bin/bash' 
   echo "echo running on \`hostname\` > $dir/decode${x}.log" 
   echo "cd ${PWD}"
   cat ./path.sh
   echo gmm-decode-faster --beam=$beam --max-active=$max_active --acoustic-scale=$acwt --word-symbol-table=data/words.txt $model $graphdir/HCLG.fst \"$feats\" ark,t:$dir/test${x}.tra ark,t:$dir/test${x}.ali  2\>\> $dir/decode${x}.log ) > $scriptfile
  # -sync y causes the qsub to wait till the job is done. 
  # Then the wait statement below waits for all the jobs to finish.
  cmd=" qsub $qsub_opts -sync y  -q ${queue} -o $dir/qlog/log.$x -e $dir/qlog/err.$x  $scriptfile "
  # this script retries once in case of failure.
  sleep 1; # Wait a bit for the file system to sync up...
  ( $cmd || ( cp $dir/decode${x}.log{,.first_try}; echo "Retrying command for part $x"; $cmd ) ) &
done

wait;

cat data/test_trans.txt | sed 's:<NOISE>::g' |  sed 's:<SPOKEN_NOISE>::g' > $dir/test_trans.filt

cat $dir/test{0,1,2,3,4,5,6,7,8,9}.tra  | \
  scripts/int2sym.pl --ignore-first-field data/words.txt | \
  sed 's:<s>::' | sed 's:</s>::' | sed 's:<UNK>::g' | \
  compute-wer --text --mode=present ark:$dir/test_trans.filt  ark,p:-   >& $dir/wer

