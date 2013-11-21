#!/bin/bash

# Copyright Johns Hopkins University (Author: Daniel Povey)  2013
# Apache 2.0.

# Begin configuration section.
wildcard_symbols=
cmd=run.pl
acwt=0.08333
beam=
stage=0
cleanup=true
# End configuration section.

. utils/parse_options.sh

echo "$0 $@"  # Print the command line for logging

if [ $# != 3 ]; then
   echo "Compute lattice oracle WER and depth, optionally pruning and minimizing the lattice"
   echo "beforehand.  To produce oracle WER, requires there to be a file 'text' in data dir"
   echo "(not usable if only stm is present)"
   echo ""
   echo "Usage: $0 [options] <data-dir> <lang-dir> <decode-dir>"
   echo "e.g.: $0 --wildcard-symbols=1:3:4 data/test data/lang exp/tri5/test_tg"
   echo "Options:"
   echo "  --wildcard-symbols <colon-separated-integer-list>  # Allows you to specify words"
   echo "                                                     # to be removed from both reference"
   echo "                                                     # and hypothesis before computing oracle."
   echo "  --cmd <cmd>                                        # How to run the jobs (default: run.pl)"
   echo "  --acwt <acwt>                                      # Acoustic scale, default $acwt: only"
   echo "                                                     # has an effect if --prune option used."
   echo "  --beam <prune-beam, e.g. 6.0>                      # Lattice pruning beam (optional; can"
   echo "                                                     # be used to compute oracle and depth at"
   echo "                                                     # various beams."
   echo "  --stage <stage>                                    # Used to control partial re-runs"
   echo "  --cleanup <true|false>                             # If true, remove pruned lattices."
   exit 1;
fi

. ./path.sh || exit 1;

data=$1
lang=$2
dir=$3


for f in $data/text $lang/words.txt $dir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done

nj=`cat $dir/num_jobs` || exit 1;
oov_sym=`cat $lang/oov.int`
sdata=$data/split$nj;
split_data.sh $data $nj || exit 1;

nl=$(grep -v IGNORE_TIME_SEGMENT_IN_SCORING $data/text | wc -l)
if [ $nl -eq 0 ]; then
  echo "$0: error: $data/text only contains IGNORE_TIME_SEGMENT_IN_SCORING, or is empty."
  exit 1;
fi

if [ ! -z "$beam" ]; then
  prunedir=${dir}/lats_beam${beam}
  mkdir -p $prunedir/log
  
  if [ $stage -le 0 ]; then
    echo "$0: creating pruned lattices"
    $cmd JOB=1:$nj $prunedir/log/prune.JOB.log \
      lattice-prune --acoustic-scale=$acwt --beam=$beam  \
        "ark:gunzip -c $dir/lat.JOB.gz|" "ark:|gzip -c >$prunedir/lat.JOB.gz" || exit 1;
  fi
else
  prunedir=$dir
fi

mkdir -p $prunedir/log


if [ $stage -le 1 ]; then
  echo "$0: measuring lattice depth"
  $cmd JOB=1:$nj $prunedir/log/lattice_depth.JOB.log \
    lattice-depth "ark:gunzip -c $prunedir/lat.JOB.gz|" ark:/dev/null || exit 1;

  # look for lines like: LOG (blah:blah.cc:95) Overall density is 153.3 over 164361 frames
  grep -w Overall $prunedir/log/lattice_depth.*.log | \
    awk -v nj=$nj '{num+=$6*$8; den+=$8; nl++} END{ 
      if (nl != nj) { print "Error: expected " nj " lines, got " nl >"/dev/stderr"; }
      printf("%.2f ( %d / %d )\n", num/den, num, den); }' > $prunedir/depth || exit 1;
  echo -n "Depth is: "
  cat $prunedir/depth
fi


if [ $stage -le 2 ]; then
  echo "$0: measuring lattice oracle WER"
  $cmd JOB=1:$nj $prunedir/log/lattice_oracle.JOB.log \
    lattice-oracle --wildcard-symbols=$wildcard_symbols  \
    "ark:gunzip -c $prunedir/lat.JOB.gz|" \
   "ark:sym2int.pl --map-oov $oov_sym -f 2- $lang/words.txt $sdata/JOB/text | grep -v IGNORE_TIME_SEGMENT_IN_SCORING |"  \
   ark:/dev/null || exit 1;

  # look for lines like: LOG (blah:blah.cc:95) Overall %WER 25.6 [ 1243 / 6331, ... ]  
  grep -w Overall $prunedir/log/lattice_oracle.*.log | \
    awk -v nj=$nj '{num+=$7; den+=$9; ins+=$10; del+=$12; sb+=$14; nl++} END{ 
      if (nl != nj) { print "Error: expected " nj " lines, got " nl >"/dev/stderr"; }
      printf("%.2f%% [ %d / %d, %d insertions, %d deletions, %d substitutions ]\n", (100.0 * num/den), num, den, ins, del, sb); }' > \
      $prunedir/oracle_wer || exit 1;
  echo -n "Oracle WER is: "
  cat $prunedir/oracle_wer
fi

if $cleanup && [ ! -z $beam ]; then
  echo "$0: removing pruned lattices in $prunedir"
  rm $prunedir/lat.*.gz
fi

exit 0;
