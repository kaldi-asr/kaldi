#!/bin/bash

# Decoding script that works with an SGMM model; uses the word information in
# the lattices in the base decoding directory, which must be provided
# [note: if this directory contains transforms, we'll use this: thus, for
# system where the SGMM is built on top of speaker adapted features,
# the command-line syntax would be the same as decode_sgmm_lda_etc.sh, which
# looks for the transforms in that directory, if you provide it.
# This script works on top of LDA + [something] features.
# Note: the num-jobs in the base-directory must be the same as in this one.

if [ -f ./path.sh ]; then . ./path.sh; fi

nj=1
jobid=0
scale_opts="--transition-scale=1.0 --self-loop-scale=0.1"

for n in 1 2; do
  if [ "$1" == "-j" ]; then
    shift;
    nj=$1; 
    jobid=$2;
    shift; shift;
  fi
  if [ "$1" == "--scale-opts" ]; then
     scale_opts="$2";
     shift; shift;
  fi
done

if [ $# -ne 4 ]; then
   echo "Usage: steps/decode_sgmm_lda_etc_fromlats.sh [-j num-jobs job-number] <lang-dir> <data-dir> <decode-dir> <old-decode-dir>"
   echo " e.g.: steps/decode_sgmm_lda_etc_fromlats.sh -j 10 0 data/lang_test_tgpr data/test_dev93 exp/sgmm3c/decode_dev93_tgpr_fromlats exp/tri2b/decode_dev93_tgpr"
   exit 1;
fi


lang=$1
data=$2
dir=$3
olddir=$4
acwt=0.1  # Just a default value, used for adaptation and beam-pruning..
silphonelist=`cat $lang/silphones.csl` || exit 1

srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

mkdir -p $dir

if [ $nj -gt 1 ]; then
  mydata=$data/split$nj/$jobid
else
  mydata=$data
fi

requirements="$mydata/feats.scp $srcdir/final.mdl $srcdir/final.mat $srcdir/final.alimdl $lang/G.fst $lang/L_disambig.fst $lang/phones_disambig.txt"
for f in $requirements; do
  if [ ! -f $f ]; then
     echo "decode_sgmm_lda_etc.sh: no such file $f";
     exit 1;
  fi
done
[ ! -d "$olddir" ] && echo "Expected $olddir to be a directory" && exit 1;
[ ! -f $olddir/lat.$jobid.gz ] && echo No such file $olddir/lat.$jobid.gz && exit 1;


feats="ark:compute-cmvn-stats --spk2utt=ark:$mydata/spk2utt scp:$mydata/feats.scp ark:- | apply-cmvn --norm-vars=false --utt2spk=ark:$mydata/utt2spk ark:- scp:$mydata/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
[ -f $olddir/$jobid.trans ] && feats="$feats transform-feats --utt2spk=ark:$mydata/utt2spk ark:$olddir/$jobid.trans ark:- ark:- |"

# Do Gaussian selection, since we'll have two decoding passes and don't want to redo this.
# Note: it doesn't make a difference if we use final.mdl or final.alimdl, they have the
# same UBM.
sgmm-gselect $srcdir/final.mdl "$feats" "ark:|gzip -c > $dir/$jobid.gselect.gz" \
    2>$dir/gselect$jobid.log || exit 1;
gselect_opt="--gselect=ark:gunzip -c $dir/$jobid.gselect.gz|"

# Generate FSTs to search, based on the word sequences in the lattices in the base
# directory, and then decode these to make state-level lattices.


#grep \# $lang/phones_disambig.txt | awk '{print $2}' > $dir/disambig.list

# Generate a state-level lattice for rescoring, with the alignment model and no speaker
# vectors.  Instead of using the decoding graph, we create lattices and rescore them.
( lattice-to-fst "ark:gunzip -c $olddir/lat.$jobid.gz|" ark:- | \
  fsttablecompose "fstproject --project_output=true $lang/G.fst | fstarcsort |" ark:- ark:- | \
  fstdeterminizestar ark:- ark:- | \
  compile-train-graphs-fsts --read-disambig-syms="grep \# $lang/phones_disambig.txt | awk '{print \$2}'|" \
    $scale_opts $srcdir/tree $srcdir/final.mdl $lang/L_disambig.fst ark:- ark:- |  \
  sgmm-latgen-faster --max-active=7000 --beam=25.0 --lattice-beam=7.0 --acoustic-scale=$acwt  \
  --determinize-lattice=false --allow-partial=true --word-symbol-table=$lang/words.txt \
  "$gselect_opt" $srcdir/final.alimdl ark:- "$feats" "ark:|gzip -c > $dir/pre_lat.$jobid.gz" ) \
   2> $dir/decode_pass1.$jobid.log || exit 1;

( lattice-determinize --acoustic-scale=$acwt --prune=true --beam=4.0 \
     "ark:gunzip -c $dir/pre_lat.$jobid.gz|" ark:- | \
   lattice-to-post --acoustic-scale=$acwt ark:- ark:- | \
   weight-silence-post 0.0 $silphonelist $srcdir/final.alimdl ark:- ark:- | \
   sgmm-post-to-gpost "$gselect_opt" $srcdir/final.alimdl "$feats" ark:- ark:- | \
   sgmm-est-spkvecs-gpost --spk2utt=ark:$mydata/spk2utt \
    $srcdir/final.mdl "$feats" ark:- "ark:$dir/$jobid.vecs" ) \
      2> $dir/vecs.$jobid.log || exit 1;

# Now rescore the state-level lattices with the adapted features and the
# corresponding model.  Prune and determinize the lattices to limit
# their size.

sgmm-rescore-lattice "$gselect_opt" --utt2spk=ark:$mydata/utt2spk --spk-vecs=ark:$dir/$jobid.vecs \
  $srcdir/final.mdl "ark:gunzip -c $dir/pre_lat.$jobid.gz|" "$feats" \
 "ark:|lattice-determinize --acoustic-scale=$acwt --prune=true --beam=6.0 ark:- ark:- | gzip -c > $dir/lat.$jobid.gz" \
  2>$dir/rescore.$jobid.log || exit 1;

rm $dir/pre_lat.$jobid.gz

# The top-level decoding script will rescore "lat.$jobid.gz" to get the final output.
