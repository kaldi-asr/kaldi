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

# usage: scripts/decode.sh [--per-spk] [--qsub-opts opts] [--no-queue] [--num-jobs n] <decode_dir> <graph> <decode_script> <test-scp>  [extra args to script]

# Decode the testing data.   By default, uses "qsub"; if you don't have this on
# your system, you may want to use options "--no-queue --num-jobs 4" e.g. if you have a 4 core
# machine.
# Note: this script calls another script whose name you must specify on the command line,
# which does the actual decoding.  This script is mainly responsible for parallelization
# and job control.
#
# "decode_dir" is a directory to decode in.
# "graph" is somedir/HCLG.fst, a graph to decode with.
# "decode_script" is a script such as steps/decode_tri1.sh that can
#  decode an individual batch of data.
# "test_scp" is e.g. data/test_nov93.scp.
# "test_spk2utt" is e.g. data/test_nov93.spk2utt

# If you include the --per-spk option it will assume
# you are doing some speaker adaptation and require the data to be broken up
# by speaker; it will put the split-up spk2utt and utt2spk files in the
# decoding directory (e.g. test1.spk2utt and test1.utt2spk).
# option will ensure that the splitting-up is done on a per-speaker basis and
# will put files like test1.spk2utt and test1.utt2spk in the decode_dir directory.
# The decoding scripts will detect the presence of these files and do their
# adaptation on a per-speaker basis.


perspk=false
qsub_opts="-l ram_free=1200M,mem_free=1200M"
queue=all.q@@blade # Change this in the script if you need something different.
use_queue=true
num_jobs=  # If not set, will set it equal to either 10, or if the
   # test-spk2utt option is given, the number of speakers in the
   # test set (8 or 10).
include_wav=false # Needed for some VTLN scripts.

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
   if [ "$1" == "--per-spk" ]; then
      shift; 
      perspk=true
   fi
   if [ "$1" == "--wav" ]; then
      shift; 
      include_wav=true
   fi
done

if [ $# -lt 4 ]; then
   echo "Usage: scripts/decode.sh [--per-spk] [--qsub-opts opts] [--no-queue] [--num-jobs n] <decode_dir> <graph> <decode_script> <test-scp> [extra args to script]"
   exit 1;
fi

dir=$1
# Make "dir" an absolute pathname.
dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`
mkdir -p $dir || exit 1
graph=$2
script=$3 # decoding script
scp=$4 # .scp file with e.g. mfccs
shift;shift;shift;shift;
# Remaining args will be supplied to decoding script.
extra_args=$* 


if [ "$perspk" == "true" ]; then
  utt2spk=`echo $scp | sed 's:\.scp:.utt2spk:'`
else
  utt2spk=""
fi

for file in $graph $script $scp $utt2spk; do
  if [ ! -f $file ]; then
     echo "decode.sh: no such file $file"
     exit 1
  fi 
done

if [ "$num_jobs" == "" ]; then # Figure out num-jobs.
  if [ "$utt2spk" != "" ]; then # utt2spk specified..
     num_jobs=`scripts/utt2spk_to_spk2utt.pl $utt2spk | wc -l`
  else
     num_jobs=10 # A default value
  fi
fi

echo "Decoding with num-jobs = $num_jobs"

split_scp=""
n=1
while [ $n -le $num_jobs ]; do
  split_scp="$split_scp $dir/$n.scp"  
  n=$[$n+1]
done

if [ "$utt2spk" != "" ]; then 
  # We have to split up the data respecting speaker boundaries..
  # the script split_scp.pl does this when given the --utt2spk option.
  scripts/split_scp.pl --utt2spk=$utt2spk $scp $split_scp || exit 1;
  # Now create the corresponding utt2spk and spk2utt files..
  n=1
  while [ $n -le $num_jobs ]; do
    scripts/filter_scp.pl $dir/$n.scp $utt2spk > $dir/$n.utt2spk
    scripts/utt2spk_to_spk2utt.pl $dir/$n.utt2spk > $dir/$n.spk2utt 
    n=$[$n+1]
  done
else
  # splitting doesn't have to respect speaker boundaries.
  scripts/split_scp.pl $scp $split_scp || exit 1;  
  rm $dir/*.{utt2spk,spk2utt} 2>/dev/null # In case we ran previously in same dir
     # with --per-spk... this would confuse the lower-level decoding script.
fi

if [ "$include_wav" == "true" ]; then
  wav_scp=`echo $scp | sed 's:\.scp:_wav.scp:'`
  if [ ! -f $wav_scp ]; then 
     echo No such file $wav_scp 
     exit 1
  fi  
  for file in $split_scp; do
    this_wav_scp=`echo $file | sed 's:\.scp:_wav.scp:'`
    scripts/filter_scp.pl $file $wav_scp > $this_wav_scp
  done
fi

n=1
while [ $n -le $num_jobs ]; do
  if [ $use_queue == "true" ]; then
     mkdir -p $dir/qlog
     scriptfile=$dir/decode$n.sh
    ( echo '#!/bin/bash' 
      echo "cd ${PWD}"
      echo "$script $graph $dir $n $extra_args" ) > $scriptfile
     # -sync y causes the qsub to wait till the job is done. 
     # Then the wait statement below waits for all the jobs to finish.
     cmd=" qsub $qsub_opts -sync y  -q ${queue} -o $dir/qlog/log.$n -e $dir/qlog/err.$n  $scriptfile "
     # this script retries once in case of failure.
     chmod +x $scriptfile
     sleep 1; # Wait a bit for the file system to sync up...
     ( $cmd || ( cp $dir/decode${n}.log{,.first_try}; echo "Retrying command for part $n"; $cmd ) ) &
  else
    $script $graph $dir $n $extra_args &
  fi
  n=$[$n+1]
done

wait

n=1
tra=""
while [ $n -le $num_jobs ]; do
  if [ ! -f $dir/$n.tra ]; then
     echo "Decoding failed for job $n: no such file $dir/$n.tra";
     exit 1
  fi 
  tra="$tra $dir/$n.tra" 
  n=$[$n+1]
done


# text-format transcript has same name as .scp file but with the .txt extension. 
trans=`echo $scp | sed s/\.scp/\.txt/`

cat $trans | sed 's:<NOISE>::g' |  sed 's:<SPOKEN_NOISE>::g' > $dir/test_trans.filt

. path.sh || exit 1; # for compute-wer

cat $tra | \
  scripts/int2sym.pl --ignore-first-field data/words.txt | \
  sed 's:<s>::' | sed 's:</s>::' | sed 's:<UNK>::g' | \
  compute-wer --text --mode=present ark:$dir/test_trans.filt  ark,p:-   >& $dir/wer

