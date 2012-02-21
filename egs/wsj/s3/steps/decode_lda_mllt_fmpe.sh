#!/bin/bash

# Decoding script that works with a GMM model and the baseline
# [e.g. MFCC] features plus cepstral mean subtraction plus
# LDA+MLLT or similar transform, plus fMPE/FMMI.
# This script just generates lattices for a single broken-up
# piece of the data.

if [ -f ./path.sh ]; then . ./path.sh; fi

numjobs=1
jobid=0
rescore=false
if [ "$1" == "-j" ]; then
  shift;
  numjobs=$1;
  jobid=$2;
  shift; shift;
fi

if [ $# != 3 ]; then
   echo "Usage: steps/decode_lda_mllt_fmpe.sh [-j num-jobs job-number] <graph-dir> <data-dir> <decode-dir>"
   echo " e.g.: steps/decode_lda_mllt_fmpe.sh -j 8 0 exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi/decode_dev93_tgpr"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

mkdir -p $dir

if [ $numjobs -gt 1 ]; then
  mydata=$data/split$numjobs/$jobid
else
  mydata=$data
fi

requirements="$mydata/feats.scp $srcdir/final.mdl $srcdir/final.fmpe $srcdir/final.mat $graphdir/HCLG.fst"
for f in $requirements; do
  if [ ! -f $f ]; then
     echo "decode_lda_mllt_fmpe.sh: no such file $f";
     exit 1;
  fi
done


basefeats="ark:compute-cmvn-stats --spk2utt=ark:$mydata/spk2utt scp:$mydata/feats.scp ark:- | apply-cmvn --norm-vars=false --utt2spk=ark:$mydata/utt2spk ark:- scp:$mydata/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

# Get the Gaussian-selection info for the fMPE.  
ngselect=2; # Just the 2 top Gaussians. 
gmm-gselect --n=$ngselect $srcdir/final.fmpe "$basefeats" \
  "ark:|gzip -c >$dir/gselect.$jobid.gz" 2>$dir/gselect.$jobid.log

# Now set up the fMPE features.
feats="$basefeats fmpe-apply-transform $srcdir/final.fmpe ark:- 'ark,s,cs:gunzip -c $dir/gselect.$jobid.gz|' ark:- |"

gmm-latgen-faster --max-active=7000 --beam=13.0 --lattice-beam=6.0 --acoustic-scale=0.083333 \
  --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $srcdir/final.mdl $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.$jobid.gz" \
     2> $dir/decode.$jobid.log || exit 1;

