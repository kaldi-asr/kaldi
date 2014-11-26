#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2014  Vimal Manohar

# Decoding script for LVTLN models.  Will estimate VTLN warping factors
# as a by product, which can be used to extract VTLN-warped features.

# Begin configuration section
stage=0
acwt=0.083333 
max_active=3000 # Have a smaller than normal max-active, to limit decoding time.
beam=13.0
lattice_beam=6.0
nj=4
silence_weight=0.0
logdet_scale=0.0
cmd=run.pl
skip_scoring=false
num_threads=1 # if >1, will use gmm-latgen-faster-parallel
parallel_opts=  # If you supply num-threads, you should supply this too.
scoring_opts=
cleanup=true
# End configuration section
echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 3)"
   echo "Usage: steps/decode_lvtln.sh [options] <graph-dir> <data-dir> <decode-dir>"
   echo " e.g.: steps/decode_lvtln.sh exp/tri2d/graph_tgpr data/test_dev93 exp/tri2d/decode_dev93_tgpr"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                   # config containing options"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   echo "  --acwt <acoustic-weight>                 # default 0.08333 ... used to get posteriors"
   echo "  --scoring-opts <opts>                    # options to local/score.sh"
   exit 1;
fi


graphdir=$1
data=$2
dir=`echo $3 | sed 's:/$::g'` # remove any trailing slash.

srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.
sdata=$data/split$nj;

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

if [ -f $data/spk2warp ]; then
  echo "$0: file $data/spk2warp exists.  This script expects non-VTLN features"
  exit 1;
fi


mkdir -p $dir/log
split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`

silphonelist=`cat $graphdir/phones/silence.csl` || exit 1;

# Some checks.  Note: we don't need $srcdir/tree but we expect
# it should exist, given the current structure of the scripts.
for f in $graphdir/HCLG.fst $data/feats.scp $srcdir/tree $srcdir/final.mdl \
  $srcdir/final.alimdl $srcdir/final.lvtln; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

## Set up the unadapted features "$sifeats"
if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type";
case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |";;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac


## Generate lattices.
if [ $stage -le 0 ]; then
  echo "$0: doing main lattice generation phase"
  if [ -f "$graphdir/num_pdfs" ]; then
    [ "`cat $graphdir/num_pdfs`" -eq `am-info --print-args=false $srcdir/final.alimdl | grep pdfs | awk '{print $NF}'` ] || \
      { echo "Mismatch in number of pdfs with $srcdir/final.alimdl"; exit 1; }
  fi
  $cmd $parallel_opts JOB=1:$nj $dir/log/decode.JOB.log \
    gmm-latgen-faster$thread_string --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
     --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    $srcdir/final.alimdl $graphdir/HCLG.fst "$sifeats" "ark:|gzip -c > $dir/lat_pass1.JOB.gz" \
    || exit 1;
fi


## Get the first-pass LVTLN transforms
if [ $stage -le 1 ]; then
  echo "$0: getting first-pass LVTLN transforms."
  if [ -f "$graphdir/num_pdfs" ]; then
    [ "`cat $graphdir/num_pdfs`" -eq `am-info --print-args=false $srcdir/final.mdl | grep pdfs | awk '{print $NF}'` ] || \
      { echo "Mismatch in number of pdfs with $srcdir/final.mdl"; exit 1; }
  fi
  $cmd JOB=1:$nj $dir/log/lvtln_pass1.JOB.log \
    gunzip -c $dir/lat_pass1.JOB.gz \| \
    lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
    weight-silence-post $silence_weight $silphonelist $srcdir/final.alimdl ark:- ark:- \| \
    gmm-post-to-gpost $srcdir/final.alimdl "$sifeats" ark:- ark:- \| \
    gmm-est-lvtln-trans --logdet-scale=$logdet_scale --verbose=1 --spk2utt=ark:$sdata/JOB/spk2utt \
       $srcdir/final.mdl $srcdir/final.lvtln "$sifeats" ark,s,cs:- ark:$dir/trans_pass1.JOB \
       ark,t:$dir/warp_pass1.JOB || exit 1;
fi
##

feats1="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$dir/trans_pass1.JOB ark:- ark:- |"

## Do a second pass of estimating the LVTLN transform.

if [ $stage -le 3 ]; then
  echo "$0: rescoring the lattices with first-pass LVTLN transforms"
  $cmd $parallel_opts JOB=1:$nj $dir/log/rescore.JOB.log \
    gmm-rescore-lattice $srcdir/final.mdl "ark:gunzip -c $dir/lat_pass1.JOB.gz|" "$feats1" \
     "ark:|gzip -c > $dir/lat_pass2.JOB.gz" || exit 1;
fi

if [ $stage -le 4 ]; then
  echo "$0: re-estimating LVTLN transforms"
  $cmd JOB=1:$nj $dir/log/lvtln_pass2.JOB.log \
    gunzip -c $dir/lat_pass2.JOB.gz \| \
    lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
    weight-silence-post $silence_weight $silphonelist $srcdir/final.mdl ark:- ark:- \| \
    gmm-post-to-gpost $srcdir/final.mdl "$feats1" ark:- ark:- \| \
    gmm-est-lvtln-trans --logdet-scale=$logdet_scale --verbose=1 --spk2utt=ark:$sdata/JOB/spk2utt \
      $srcdir/final.mdl $srcdir/final.lvtln "$sifeats" ark,s,cs:- ark:$dir/trans.JOB \
      ark,t:$dir/warp.JOB || exit 1;
fi

feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$dir/trans.JOB ark:- ark:- |"

if [ $stage -le 5 ]; then
  # This second rescoring is only really necessary for scoring purposes,
  # it does not affect the transforms.
  echo "$0: rescoring the lattices with second-pass LVTLN transforms"
  $cmd $parallel_opts JOB=1:$nj $dir/log/rescore.JOB.log \
    gmm-rescore-lattice $srcdir/final.mdl "ark:gunzip -c $dir/lat_pass2.JOB.gz|" "$feats" \
     "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
fi

if [ -f $dir/warp.1 ]; then
  for j in $(seq $nj); do cat $dir/warp_pass1.$j; done > $dir/0.warp || exit 1;
  for j in $(seq $nj); do cat $dir/warp.$j; done > $dir/final.warp || exit 1;
  ns1=$(cat $dir/0.warp | wc -l)
  ns2=$(cat $dir/final.warp | wc -l)
  ! [ "$ns1" == "$ns2" ] && echo "$0: Number of speakers differ pass1 vs pass2, $ns1 != $ns2" && exit 1;

  paste $dir/0.warp $dir/final.warp | awk '{x=$2 - $4; if ((x>0?x:-x) > 0.010001) { print $1, $2, $4; }}' > $dir/warp_changed
  nc=$(cat $dir/warp_changed | wc -l)
  echo "$0: For $nc speakers out of $ns1, warp changed pass1 vs pass2 by >0.01, see $dir/warp_changed for details"
fi

if true; then # Diagnostics
  if [ -f $data/spk2gender ]; then 
    # To make it easier to eyeball the male and female speakers' warps
    # separately, separate them out.
    for g in m f; do # means: for gender in male female
      cat $dir/final.warp | \
        utils/filter_scp.pl <(grep -w $g $data/spk2gender | awk '{print $1}') > $dir/final.warp.$g
      echo -n "The last few warp factors for gender $g are: "
      tail -n 10 $dir/final.warp.$g | awk '{printf("%s ", $2);}'; 
      echo
    done
  fi
fi

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "$0: not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
fi

if $cleanup; then
  rm $dir/lat_pass?.*.gz $dir/trans_pass1.* $dir/warp_pass1.* $dir/warp.*
fi


exit 0;
