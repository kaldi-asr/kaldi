#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#                 Korbinian Riedhammer

# Create denominator lattices for MMI/MPE training, with SGMM models.  If the
# features have fMLLR transforms you have to supply the --transform-dir option.
# It gets any speaker vectors from the "alignment dir" ($srcdir).  Note: this is
# possibly a slight mismatch because the speaker vectors come from supervised
# adaptation.

# Begin configuration section.
nj=4
cmd=run.pl
sub_split=1
beam=13.0
lattice_beam=7.0
acwt=0.1
max_active=5000
transform_dir=
max_mem=20000000 # This will stop the processes getting too large.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Usage: steps/tandem/make_denlats_sgmm2.sh [options] <data1-dir> <data2-dir> <lang-dir> <src-dir|srcdir> <exp-dir>"
   echo "  e.g.: steps/tandem/make_denlats_sgmm2.sh {mfcc,bottleneck}/data1/train data1/lang exp/sgmm4a_ali exp/sgmm4a_denlats"
   echo "Works for (delta|lda) features, and (with --transform-dir option) such features"
   echo " plus transforms."
   echo ""
   echo "Main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --sub-split <n-split>                            # e.g. 40; use this for "
   echo "                           # large databases so your jobs will be smaller and"
   echo "                           # will (individually) finish reasonably soon."
   echo "  --transform-dir <transform-dir>   # directory to find fMLLR transforms."
   exit 1;
fi

data1=$1
data2=$2
lang=$3
srcdir=$4 # could also be $srcdir, but only if no vectors supplied.
dir=$5

mkdir -p $dir/log
echo $nj > $dir/num_jobs

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt || exit 1;

sdata1=$data1/split$nj
sdata2=$data2/split$nj
[[ -d $sdata1 && $data1/feats.scp -ot $sdata1 ]] || split_data.sh $data1 $nj || exit 1;
[[ -d $sdata2 && $data2/feats.scp -ot $sdata2 ]] || split_data.sh $data2 $nj || exit 1;

oov=`cat $lang/oov.int` || exit 1;

mkdir -p $dir

cp -r $lang $dir/

# Compute grammar FST which corresponds to unigram decoding graph.

cat $data1/text | utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt | \
  awk '{for(n=2;n<=NF;n++){ printf("%s ", $n); } printf("\n"); }' | \
  utils/make_unigram_grammar.pl | fstcompile > $dir/lang/G.fst \
   || exit 1;

# mkgraph.sh expects a whole directory "lang", so put everything in one directory...
# it gets L_disambig.fst and G.fst (among other things) from $dir/lang, and
# final.mdl from $srcdir; the output HCLG.fst goes in $dir/graph.

if [ -s $dir/dengraph/HCLG.fst ]; then
   echo "Graph $dir/dengraph/HCLG.fst already exists: skipping graph creation."
else
  utils/mkgraph.sh $dir/lang $srcdir $dir/dengraph || exit 1;
fi


## Set up features.
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
normft2=`cat $srcdir/normft2 2>/dev/null`

if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi

case $feat_type in
  delta)
    echo "$0: feature type is $feat_type"
    ;;
  lda)
    echo "$0: feature type is $feat_type"
    cp $srcdir/{lda,final}.mat $dir/ || exit 1;
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

# set up feature stream 1;  this are usually spectral features, so we will add
# deltas or splice them
feats1="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata1/JOB/utt2spk scp:$sdata1/JOB/cmvn.scp scp:$sdata1/JOB/feats.scp ark:- |"

if [ "$feat_type" == "delta" ]; then
  feats1="$feats1 add-deltas ark:- ark:- |"
elif [ "$feat_type" == "lda" ]; then
  feats1="$feats1 splice-feats $splice_opts ark:- ark:- | transform-feats $dir/lda.mat ark:- ark:- |"
fi

# set up feature stream 2;  this are usually bottleneck or posterior features,
# which may be normalized if desired
feats2="scp:$sdata2/JOB/feats.scp"

if [ "$normft2" == "true" ]; then
  feats2="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata2/JOB/utt2spk scp:$sdata2/JOB/cmvn.scp $feats2 ark:- |"
fi

# assemble tandem features
feats="ark,s,cs:paste-feats '$feats1' '$feats2' ark:- |"

# add transformation, if applicable
if [ "$feat_type" == "lda" ]; then
  feats="$feats transform-feats $dir/final.mat ark:- ark:- |"
fi

# splicing/normalization options
cp $srcdir/{splice_opts,normft2,tandem} $dir 2>/dev/null

if [ ! -z "$transform_dir" ]; then # add transforms to features...
  echo "$0: using fMLLR transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "Expected $transform_dir/trans.1 to exist."
  [ "`cat $transform_dir/num_jobs`" -ne "$nj" ] \
    && echo "$0: mismatch in number of jobs with $transform_dir" && exit 1;
  [ -f $srcdir/final.mat ] && ! cmp $transform_dir/final.mat $srcdir/final.mat && \
     echo "$0: LDA transforms differ between $srcdir and $transform_dir"
  feats="$feats transform-feats --utt2spk=ark:$sdata1/JOB/utt2spk ark:$transform_dir/trans.JOB ark:- ark:- |"
else
  echo "Assuming you don't have a SAT system, since no --transform-dir option supplied "
fi

if [ -f $srcdir/gselect.1.gz ]; then
  gselect_opt="--gselect=ark:gunzip -c $srcdir/gselect.JOB.gz|"
else
  echo "$0: no such file $srcdir/gselect.1.gz" && exit 1;
fi

if [ -f $srcdir/vecs.1 ]; then
  spkvecs_opt="--spk-vecs=ark:$srcdir/vecs.JOB --utt2spk=ark:$sdata1/JOB/utt2spk"
else
  if [ -f $srcdir/final.alimdl ]; then
    echo "$0: You seem to have an SGMM system with speaker vectors,"
    echo "yet we can't find speaker vectors.  Perhaps you supplied"
    echo "the model director instead of the alignment directory?"
    exit 1;
  fi
fi

if [ $sub_split -eq 1 ]; then
  $cmd JOB=1:$nj $dir/log/decode_den.JOB.log \
   sgmm2-latgen-faster $spkvecs_opt "$gselect_opt" --beam=$beam \
     --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
     --max-mem=$max_mem --max-active=$max_active --word-symbol-table=$lang/words.txt $srcdir/final.mdl  \
     $dir/dengraph/HCLG.fst "$feats" "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
else
  for n in `seq $nj`; do
    if [ -f $dir/.done.$n ] && [ $dir/.done.$n -nt $srcdir/final.mdl ]; then
      echo "Not processing subset $n as already done (delete $dir/.done.$n if not)";
    else
      ssdata1=$data1/split$nj/$n/split${sub_split}utt;
      split_data.sh --per-utt $sdata1/$n $sub_split || exit 1;
      ssdata2=$data2/split$nj/$n/split${sub_split}utt;
      split_data.sh --per-utt $sdata2/$n $sub_split || exit 1;
      mkdir -p $dir/log/$n
      mkdir -p $dir/part
      feats_subset=`echo $feats | sed "s/trans.JOB/trans.$n/g" | sed s:JOB/:$n/split${sub_split}utt/JOB/:g`
      spkvecs_opt_subset=`echo $spkvecs_opt | sed "s/JOB/$n/g"`
      gselect_opt_subset=`echo $gselect_opt | sed "s/JOB/$n/g"`
      $cmd JOB=1:$sub_split $dir/log/$n/decode_den.JOB.log \
        sgmm2-latgen-faster $spkvecs_opt_subset "$gselect_opt_subset" \
          --beam=$beam --lattice-beam=$lattice_beam \
          --acoustic-scale=$acwt --max-mem=$max_mem --max-active=$max_active \
          --word-symbol-table=$lang/words.txt $srcdir/final.mdl  \
          $dir/dengraph/HCLG.fst "$feats_subset" "ark:|gzip -c >$dir/lat.$n.JOB.gz" || exit 1;
      echo Merging archives for data subset $n
      rm $dir/.error 2>/dev/null;
      for k in `seq $sub_split`; do
        gunzip -c $dir/lat.$n.$k.gz || touch $dir/.error;
      done | gzip -c > $dir/lat.$n.gz || touch $dir/.error;
      [ -f $dir/.error ] && echo Merging lattices for subset $n failed && exit 1;
      rm $dir/lat.$n.*.gz
      touch $dir/.done.$n
    fi
  done
fi


echo "$0: done generating denominator lattices with SGMMs."
