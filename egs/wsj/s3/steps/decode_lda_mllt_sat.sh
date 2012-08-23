#!/bin/bash

# Copyright 2010-2012 Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)

# Decoding script that works with a GMM model and the baseline
# [e.g. MFCC] features plus cepstral mean subtraction plus
# LDA + MLLT + fMLLR features.  
# This version of the script first does a tight-beam decoding
# with the "alignment model" to get a first estimate of the SAT
# transform, then generates a state-level lattice with the SAT
# model and re-estimates the SAT transform, before doing the
# final decoding.

if [ -f ./path.sh ]; then . ./path.sh; fi


beam1=10.0
beam2=13.0
numjobs=1
jobid=0
fmllr_update_type=full

for x in `seq 5`; do
  if [ "$1" == "-j" ]; then
    shift;
    numjobs=$1;
    jobid=$2;
    shift 2;
    ! scripts/get_splits.pl $numjobs | grep -w $jobid >/dev/null && \
      echo Invalid job-number $jobid "(num-jobs = $numjobs)" && exit 1;
  fi
  if [ "$1" == "--fmllr-update-type" ]; then
    fmllr_update_type=$2  # full|diag|offset|none
    shift 2;
  fi
  if [ "$1" == "--beam1" ]; then
    beam1=$2
    shift 2;
  fi
  if [ "$1" == "--beam2" ]; then
    beam2=$2
    shift 2;
  fi
done

if [ $# != 3 ]; then
   echo "Usage: steps/decode_lda_mllt_sat.sh [-j num-jobs job-number] <graph-dir> <data-dir> <decode-dir>"
   echo " e.g.: steps/decode_lda_mllt_sat.sh -j 10 0 exp/tri2c/graph_tgpr data/test_dev93 exp/tri2c/decode_dev93_tgpr"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
acwt=0.08333 # Just used for adaptation and beam-pruning..
#acwt=0.0625 # Just used for adaptation and beam-pruning..
silphonelist=`cat $graphdir/silphones.csl`

srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

mkdir -p $dir

if [ $numjobs -gt 1 ]; then
  mydata=$data/split$numjobs/$jobid
else
  mydata=$data
fi

requirements="$mydata/feats.scp $srcdir/final.mdl $srcdir/final.mat $srcdir/final.alimdl $graphdir/HCLG.fst"
for f in $requirements; do
  if [ ! -f $f ]; then
     echo "decode_lda_mllt.sh: no such file $f";
     exit 1;
  fi
done


# basefeats is the speaker independent features.
basefeats="ark:compute-cmvn-stats --spk2utt=ark:$mydata/spk2utt scp:$mydata/feats.scp ark:- | apply-cmvn --norm-vars=false --utt2spk=ark:$mydata/utt2spk ark:- scp:$mydata/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

# The first-pass decoding uses the speaker-independent features and the
# alignment model, and a very tight beam.  We generate a small lattice as
# we won't be rescoring it, we'll just get the posteriors from it to get
# the first estimate of the transform.

gmm-latgen-faster --max-active=7000 --beam=$beam1 --lattice-beam=3.0 --acoustic-scale=$acwt  \
  --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $srcdir/final.alimdl $graphdir/HCLG.fst "$basefeats" "ark:|gzip -c > $dir/pre_lat1.$jobid.gz" \
   2> $dir/decode_pass1.$jobid.log || exit 1;

(  gunzip -c $dir/pre_lat1.$jobid.gz | \
   lattice-to-post --acoustic-scale=$acwt ark:- ark:- | \
   weight-silence-post 0.0 $silphonelist $srcdir/final.alimdl ark:- ark:- | \
   gmm-post-to-gpost $srcdir/final.alimdl "$basefeats" ark:- ark:- | \
   gmm-est-fmllr-gpost --fmllr-update-type=$fmllr_update_type \
       --spk2utt=ark:$mydata/spk2utt $srcdir/final.mdl "$basefeats" \
       ark,s,cs:- ark:$dir/$jobid.pre_trans ) \
    2> $dir/fmllr1.$jobid.log || exit 1;


feats="$basefeats transform-feats --utt2spk=ark:$mydata/utt2spk ark:$dir/$jobid.pre_trans ark:- ark:- |"

# Generate a state-level lattice for rescoring, using the 1st-pass estimated SAT
# transform.

gmm-latgen-faster --max-active=7000 --beam=$beam2 --lattice-beam=6.0 --acoustic-scale=$acwt  \
  --determinize-lattice=false --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $srcdir/final.mdl $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/pre_lat2.$jobid.gz" \
   2> $dir/decode_pass2.$jobid.log || exit 1;

# Do another pass of estimating the transform.

(  lattice-determinize --acoustic-scale=$acwt --prune=true --beam=4.0 \
     "ark:gunzip -c $dir/pre_lat2.$jobid.gz|" ark:- | \
   lattice-to-post --acoustic-scale=$acwt ark:- ark:- | \
   weight-silence-post 0.0 $silphonelist $srcdir/final.mdl ark:- ark:- | \
   gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
      --spk2utt=ark:$mydata/spk2utt $srcdir/final.mdl "$feats" \
      ark,s,cs:- ark:$dir/$jobid.trans.tmp ) \
    2> $dir/fmllr2.$jobid.log || exit 1;

compose-transforms --b-is-affine=true ark:$dir/$jobid.trans.tmp ark:$dir/$jobid.pre_trans \
    ark:$dir/$jobid.trans 2>$dir/compose_transforms.$jobid.log || exit 1;
#rm $dir/$jobid.pre_trans $dir/$jobid.trans.tmp || exit 1;

feats="$basefeats transform-feats --utt2spk=ark:$mydata/utt2spk ark:$dir/$jobid.trans ark:- ark:- |"

# Now rescore the state-level lattices with the adapted features and the
# corresponding model.  Prune and determinize the lattices to limit
# their size.

gmm-rescore-lattice $srcdir/final.mdl "ark:gunzip -c $dir/pre_lat2.$jobid.gz|" "$feats" \
 "ark:|lattice-determinize --acoustic-scale=$acwt --prune=true --beam=8.0 ark:- ark:- | gzip -c > $dir/lat.$jobid.gz" \
  2>$dir/rescore.$jobid.log || exit 1;

rm $dir/pre_lat1.$jobid.gz
rm $dir/pre_lat2.$jobid.gz

# The top-level decoding script will rescore "lat.$jobid.gz" to get the final output.
