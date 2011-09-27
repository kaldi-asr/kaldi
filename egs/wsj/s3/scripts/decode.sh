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


qsub_opts="-l ram_free=1200M,mem_free=1200M"
queue=all.q@@blade # Change this in the script if you need something different.
use_queue=true
num_jobs=  # If not set, will set it equal to the number of
           # speakers in the test set.


for n in 1 2 3 4; do
   if [ "$1" == "--qsub-opts" ]; then
      shift; 
      qsub_opts=$1
      shift;
   fi
   if [ "$1" == "--num-jobs" ]; then
      shift; 
      num_jobs=$1
      shift;
   fi
   if [ "$1" == "--no-queue" ]; then
      shift; 
      use_queue=false
   fi
done

if [ $# -lt 4 ]; then
   echo "Usage: scripts/decode.sh [--qsub-opts opts] [--no-queue] [--num-jobs n]  <decode_script> <graph-dir> <data-dir> <decode-dir> [extra-args...]"
   exit 1;
fi

script=$1
graphdir=$2
graph=$graphdir/HCLG.fst
data=$3
dir=$4
# Make "dir" an absolute pathname.
dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`
mkdir -p $dir || exit 1
shift;shift;shift;shift;
# Remaining args will be supplied to decoding script.
extra_args=$* 

for file in $graph $script $scp $data/utt2spk; do
  if [ ! -f $file ]; then
     echo "decode.sh: no such file $file"
     exit 1
  fi 
done

if [ "$num_jobs" == "" ]; then # Figure out num-jobs.
  num_jobs=`scripts/utt2spk_to_spk2utt.pl $data/utt2spk | wc -l`
fi

echo "Decoding with num-jobs = $num_jobs"
if [[ $num_jobs -gt 1 || ! -d $data/split$num_jobs || $data/split$num_jobs -ot $data/feats.scp ]]; then
  scripts/split_data.sh $data $num_jobs
fi

n=0
while [ $n -lt $num_jobs ]; do
  if [ $use_queue == "true" ]; then
     mkdir -p $dir/qlog
     scriptfile=$dir/decode$n.sh
    ( echo '#!/bin/bash' 
      echo "cd ${PWD}"
      echo "$script -j $num_jobs $n $graphdir $data $dir $extra_args" ) > $scriptfile
     # -sync y causes the qsub to wait till the job is done. 
     # Then the wait statement below waits for all the jobs to finish.
     cmd=" qsub $qsub_opts -sync y  -q ${queue} -o $dir/qlog/log.$n -e $dir/qlog/err.$n  $scriptfile "
     # this script retries once in case of failure.
     chmod +x $scriptfile
     sleep 1; # Wait a bit for the file system to sync up...
     ( $cmd || ( cp $dir/decode${n}.log{,.first_try}; echo "Retrying command for part $n"; $cmd ) ) &
  else
    $script $graphdir $dir $n $extra_args &
  fi
  n=$[$n+1]
done

wait


scripts/score_lats.sh $dir $graphdir/words.txt $data

