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


# Decode the testing data, using the "qsub" command,
# and fMLLR

. path.sh || exit 1;

acwt=0.0625
prebeam=11.0
beam=13.0
max_active=7000
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
   if [ "$1" == "--pre-beam" ]; then
      shift;
      prebeam=$1;
      shift;
   fi
   if [ "$1" == "--max-active" ]; then
      shift;
      max_active=$1;
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
silphones=`cat data/silphones.csl`

mkdir -p $dir
mkdir -p $dir/qlog

queue=all.q@@blade

# Use 1 core per test speaker.

nspk=`cat data/test.spk2utt | wc -l`
n=1
while [ $n -le $nspk ]; do # Split up the testing data per speaker.
  head -$n data/test.spk2utt | tail -n 1 > $dir/$n.spk2utt
  cat $dir/$n.spk2utt | awk '{for(n=2;n<=NF;n++) { print $n; }}' > $dir/$n.uttlist
  scripts/filter_scp.pl $dir/$n.uttlist data/test.scp  > $dir/$n.scp
  n=$[$n+1]
done


n=1
while [ $n -le $nspk ]; do 
  sifeats="ark:add-deltas --print-args=false scp:$dir/$n.scp ark:- |"
  feats="ark:add-deltas --print-args=false scp:$dir/$n.scp ark:- | transform-feats --utt2spk=ark:data/test.utt2spk ark:$dir/$n.fmllr ark:- ark:- |"


  scriptfile=$dir/decode$n.sh
 ( echo '#!/bin/bash' 
   echo "echo running on \`hostname\` > $dir/predecode$n.log" 
   echo "cd ${PWD}"
   cat ./path.sh
   echo gmm-decode-faster --beam=$prebeam --max-active=$max_active --acoustic-scale=$acwt --word-symbol-table=data/words.txt $model $graphdir/HCLG.fst \"$sifeats\" ark,t:$dir/$n.pre_tra ark,t:$dir/pretest_$n.ali  2\>\> $dir/predecode$n.log  "||" exit 1
   echo "(" ali-to-post ark:$dir/pretest_$n.ali ark:- "|" \
        weight-silence-post 0.0 $silphones $model ark:- ark:- "|" \
        gmm-est-fmllr --spk2utt=ark:$dir/$n.spk2utt $model \"$sifeats\" ark:- ark:$dir/$n.fmllr ")" 2\>$dir/fmllr$n.log  "||" exit 1
   echo gmm-decode-faster --beam=$beam --max-active=$max_active --acoustic-scale=$acwt --word-symbol-table=data/words.txt $model $graphdir/HCLG.fst \"$feats\" ark,t:$dir/$n.tra ark,t:$dir/$n.ali  2\>\> $dir/decode$n.log 
 ) > $scriptfile
  # -sync y causes the qsub to wait till the job is done. 
  # Then the wait statement below waits for all the jobs to finish.
  cmd=" qsub $qsub_opts -sync y  -q ${queue} -o $dir/qlog/log.$x -e $dir/qlog/err.$x  $scriptfile "
  # this script retries once in case of failure.
  sleep 1; # Wait a bit for the file system to sync up...
  ( $cmd || ( cp $dir/decode$n.log{,.first_try}; echo "Retrying command for part $x"; $cmd ) ) &
  n=$[$n+1]; 
done

wait;

cat data/test_trans.txt | sed 's:<NOISE>::g' |  sed 's:<SPOKEN_NOISE>::g' > $dir/test_trans.filt

cat $dir/*.tra  | \
  scripts/int2sym.pl --ignore-first-field data/words.txt | \
  sed 's:<s>::' | sed 's:</s>::' | sed 's:<UNK>::g' | \
  compute-wer --text --mode=present ark:$dir/test_trans.filt  ark,p:-   >& $dir/wer

