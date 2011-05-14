#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation  Arnab Ghoshal

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


if [ -f path.sh ]; then . path.sh; fi

dir=exp/mono
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
tmparchive=/tmp/tmp.wsj.mono.$USER  # don't make it dependent on the process-id, just
# to make it easier to copy-paste commands and have them still work.
feats="ark:add-deltas --print-args=false ark:$tmparchive ark:- |"

numiters=40
maxiterinc=30
numgauss=300 # Initial total #Gaussians.  Must be >= #states.
totgauss=1000 # Target #Gaussians.  
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss
realign_iters="1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 23 26 29 32 35 38";


mkdir -p $dir

# Here we are selecting an evenly distributed subset of 2000 utterances from
# the SI-84 data (which is the first 7138 lines of the scp in data/train.scp).
# We get this from across all speakers.
head -7138 data/train.scp | scripts/subset_scp.pl 2000 - > $dir/train.scp
head -7138 data/train.tra | scripts/subset_scp.pl 2000 - > $dir/train.tra


trap "rm -f $tmparchive" 0 2 3 15 # if this process is interrupted, remove the temp file.
# It's nicer on the disk to compact the features to one file before processing it, as 
# the features we use are not contiguous in the archive on disk. (also, this one is local).
copy-feats scp:$dir/train.scp ark:$tmparchive >$dir/copy_feats.log



silphones=`cat data/silphones.csl | sed 's/:/ /g'`
nonsilphones=`cat data/nonsilphones.csl | sed 's/:/ /g'`
cat conf/topo.proto | sed "s:NONSILENCEPHONES:$nonsilphones:" | sed "s:SILENCEPHONES:$silphones:" > $dir/topo

# The next script is specialized for WSJ.
scripts/make_shared_phones.sh > $dir/shared_phones0.txt
scripts/sym2int.pl data/phones.txt $dir/shared_phones0.txt > $dir/shared_phones.txt

gmm-init-mono --shared-phones=$dir/shared_phones.txt '--train-feats=ark:head -10 data/train.scp | add-deltas scp:- ark:- |' $dir/topo 39  $dir/0.mdl $dir/tree 2> $dir/init.out || exit 1;

echo Compiling training graphs

compile-train-graphs $dir/tree $dir/0.mdl  data/L.fst "ark:cat $dir/train.tra|" \
   "ark:|gzip -c >$dir/graphs.fsts.gz"  2>$dir/compile_graphs.log || exit 1 

echo Pass 0

align-equal-compiled "ark:gunzip -c $dir/graphs.fsts.gz|" "$feats" \
   ark,t,f:-  2>$dir/align.0.log | \
 gmm-acc-stats-ali --binary=true $dir/0.mdl "$feats" ark:- \
     $dir/0.acc 2> $dir/acc.0.log  || exit 1;

# In the following steps, the --min-gaussian-occupancy=3 option is 
# important, otherwise
# we fail to est "rare" phones and later on, they never align properly.

gmm-est --mix-up=$numgauss --min-gaussian-occupancy=3 $dir/0.mdl \
   $dir/0.acc $dir/1.mdl 2> $dir/update.0.log || exit 1;

rm $dir/0.mdl
rm $dir/0.acc

( 
 # Just put a few graphs in human-readable form
 # for easier debugging.
  mkdir -p $dir/graph_egs
  n=5
  head -$n $dir/train.tra | awk '{printf("%s '$dir'/graph_egs/%s.fst\n", $1, $1); }' > $dir/some_graphs.scp
  compile-train-graphs $dir/tree $dir/1.mdl  data/L.fst "ark:head -$n $dir/train.tra|" \
    "scp:$dir/some_graphs.scp"  2>$dir/compile_some_graphs.log || exit 1 
  for filename in `cat $dir/some_graphs.scp | awk '{print $2;}'`; do
     fstprint --osymbols=data/words.txt $filename > $filename.txt
  done
)

x=1
beam=6 # Use smaller beam for 1st pass as all models are very similar. 
while [ $x -lt $numiters ]; do
  echo "Pass $x"
  if echo $realign_iters | grep -w $x >/dev/null; then
    echo "Aligning data"
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$[$beam*2] $dir/$x.mdl "ark:gunzip -c $dir/graphs.fsts.gz|" "$feats" t,ark:$dir/cur.ali 2> $dir/align.$x.log || exit 1;
  fi
  gmm-acc-stats-ali --binary=false $dir/$x.mdl "$feats" ark:$dir/cur.ali $dir/$x.acc 2> $dir/acc.$x.log  || exit 1;
  gmm-est --mix-up=$numgauss $dir/$x.mdl $dir/$x.acc $dir/$[$x+1].mdl 2> $dir/update.$x.log || exit 1;
  rm $dir/$x.acc $dir/$x.mdl
  if [ $x -le $maxiterinc ]; then
    numgauss=$[$numgauss+$incgauss];
  fi
  beam=10
  x=$[$x+1];
done


rm -f $tmparchive
rm $dir/final.mdl 2>/dev/null
ln -s $x.mdl $dir/final.mdl

# example of showing the alignments:
# show-alignments data/phones.txt $dir/30.mdl ark:$dir/cur.ali | head -4

