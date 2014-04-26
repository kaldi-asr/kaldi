#!/bin/bash

# Copyright 2014  Vimal Manohar

# Computes training alignments; assumes features are (LDA+MLLT or delta+delta-delta)
# Will ignore fMLLR.
# Will estimate VTLN warping factors
# as a by product, which can be used to extract VTLN-warped features.

# Begin configuration section
stage=0
nj=4
cmd=run.pl
use_graphs=false
# Begin configuration.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10.0
retry_beam=40
boost_silence=1.0 # factor by which to boost silence during alignment.
logdet_scale=1.0
cleanup=false

# End configuration section
echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Wrong #arguments ($#, expected 4)"
   echo "Usage: steps/align_lvtln.sh [options] <data-dir> <lang-dir> <src-dir>  <align-dir>"
   echo " e.g.: steps/align_lvtln.sh data/train data/lang exp/tri2c exp/tri2c_ali"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                   # config containing options"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --use-graphs true                                # use graphs in src-dir"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4

oov=`cat $lang/oov.int` || exit 1;
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
sdata=$data/split$nj

if [ -f $data/spk2warp ]; then
  echo "$0: file $data/spk2warp exists.  This script expects non-VTLN features"
  exit 1;
fi

mkdir -p $dir/log
echo $nj > $dir/num_jobs
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

cp $srcdir/{tree,final.mdl,final.lvtln} $dir || exit 1;
cp $srcdir/final.occs $dir;

splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
cp $srcdir/splice_opts $dir 2>/dev/null # frame-splicing options.
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
cp $srcdir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.

## Set up the unadapted features "$sifeats"
if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type";
case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    cp $srcdir/final.mat $srcdir/full.mat $dir    
    ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

## Set up model and alignment model.
mdl=$srcdir/final.mdl
if [ -f $srcdir/final.alimdl ]; then
  alimdl=$srcdir/final.alimdl
else
  alimdl=$srcdir/final.mdl
fi
[ ! -f $mdl ] && echo "$0: no such model $mdl" && exit 1;
alimdl_cmd="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $alimdl - |"
mdl_cmd="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $mdl - |"

## Work out where we're getting the graphs from.
if $use_graphs; then
  [ "$nj" != "`cat $srcdir/num_jobs`" ] && \
    echo "$0: you specified --use-graphs true, but #jobs mismatch." && exit 1;
  [ ! -f $srcdir/fsts.1.gz ] && echo "No graphs in $srcdir" && exit 1;
  graphdir=$srcdir
else
  graphdir=$dir
  if [ $stage -le 0 ]; then
    echo "$0: compiling training graphs"
    tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";   
    $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log  \
      compile-train-graphs $dir/tree $dir/final.mdl  $lang/L.fst "$tra" \
        "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
  fi
fi

if [ $stage -le 1 ]; then
  echo "$0: aligning data in $data using $alimdl and speaker-independent features."
  $cmd JOB=1:$nj $dir/log/align_pass1.JOB.log \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$alimdl_cmd" \
    "ark:gunzip -c $graphdir/fsts.JOB.gz|" "$sifeats" "ark:|gzip -c >$dir/pre_ali.JOB.gz" || exit 1;
fi

if [ -f $data/segments ]; then
  subset_utts="ark:extract-segments scp:$sdata/JOB/wav.scp $sdata/JOB/segments ark:- |"
else
  echo "$0 [info]: no segments file exists: using wav.scp directly."
  subset_utts="ark:wav-copy scp:$sdata/JOB/wav.scp ark:- |"
fi

## Get the first-pass LVTLN transforms
if [ $stage -le 2 ]; then
  echo "$0: computing first-pass LVTLN transforms."
  $cmd JOB=1:$nj $dir/log/lvtln_pass1.JOB.log \
    ali-to-post "ark:gunzip -c $dir/pre_ali.JOB.gz|" ark:- \| \
    weight-silence-post 0.0 $silphonelist $alimdl ark:- ark:- \| \
    gmm-post-to-gpost $alimdl "$sifeats" ark:- ark:- \| \
    gmm-est-lvtln-trans --verbose=1 --spk2utt=ark:$sdata/JOB/spk2utt --logdet-scale=$logdet_scale \
    $mdl $dir/final.lvtln "$sifeats" ark,s,cs:- ark:$dir/trans_pass1.JOB \
    ark,t:$dir/warp_pass1.JOB || exit 1;
fi
##

feats1="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$dir/trans_pass1.JOB ark:- ark:- |"

## Do a second pass of estimating the LVTLN transform.

if [ $stage -le 3 ]; then
  echo "$0: realigning with transformed features"
  $cmd JOB=1:$nj $dir/log/align_pass2.JOB.log \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$mdl_cmd" \
    "ark:gunzip -c $graphdir/fsts.JOB.gz|" "$feats1" "ark:|gzip -c >$dir/ali_pass2.JOB.gz" || exit 1;
fi

if [ $stage -le 4 ]; then
  echo "$0: re-estimating LVTLN transforms"
  $cmd JOB=1:$nj $dir/log/lvtln_pass1.JOB.log \
    ali-to-post "ark:gunzip -c $dir/ali_pass2.JOB.gz|" ark:- \| \
    weight-silence-post 0.0 $silphonelist $alimdl ark:- ark:- \| \
    gmm-post-to-gpost $alimdl "$feats1" ark:- ark:- \| \
    gmm-est-lvtln-trans --verbose=1 --spk2utt=ark:$sdata/JOB/spk2utt --logdet-scale=$logdet_scale \
    $mdl $dir/final.lvtln "$sifeats" ark,s,cs:- ark:$dir/trans.JOB \
    ark,t:$dir/warp.JOB || exit 1;
fi

feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$dir/trans.JOB ark:- ark:- |"

if [ $stage -le 5 ]; then
  # This second alignment does not affect the transforms.
  echo "$0: realigning with the second-pass LVTLN transforms"
  $cmd JOB=1:$nj $dir/log/align.JOB.log \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$mdl_cmd" \
    "ark:gunzip -c $graphdir/fsts.JOB.gz|" "$feats" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
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

if $cleanup; then
  rm $dir/pre_ali.*.gz $dir/ali_pass?.*.gz $dir/trans_pass1.* $dir/warp_pass1.* $dir/warp.*
fi

exit 0;

