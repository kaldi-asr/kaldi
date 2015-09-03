#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#                 Carnegie Mellon University (Author: Yajie Miao)
# Apache 2.0.
# This is a modified version by Yajie for BNF front-end.

# Create denominator lattices for MMI/MPE training, with SGMM models.  If the
# features have fMLLR transforms you have to supply the --transform-dir option.
# It gets any speaker vectors from the "alignment dir" ($alidir).  Note: this is
# possibly a slight mismatch because the speaker vectors come from supervised
# adaptation.

# Begin configuration section.
nj=4
cmd=run.pl
sub_split=1
beam=13.0
lattice_beam=7.0
acwt=0.0667  # LM weight = 15 for BNF
max_active=5000
transform_dir=
max_mem=20000000 # This will stop the processes getting too large.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: steps/make_denlats_sgmm2.sh [options] <data-dir> <lang-dir> <src-dir|alidir> <exp-dir>"
   echo "  e.g.: steps/make_denlats_sgmm2.sh data/train data/lang exp/sgmm4a_ali exp/sgmm4a_denlats"
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

data=$1
lang=$2
alidir=$3 # could also be $srcdir, but only if no vectors supplied.
dir=$4

sdata=$data/split$nj
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

oov=`cat $lang/oov.int` || exit 1;

mkdir -p $dir

cp -r $lang $dir/

# Compute grammar FST which corresponds to unigram decoding graph.

cat $data/text | utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt | \
  awk '{for(n=2;n<=NF;n++){ printf("%s ", $n); } printf("\n"); }' | \
  utils/make_unigram_grammar.pl | fstcompile > $dir/lang/G.fst \
   || exit 1;

# mkgraph.sh expects a whole directory "lang", so put everything in one directory...
# it gets L_disambig.fst and G.fst (among other things) from $dir/lang, and
# final.mdl from $alidir; the output HCLG.fst goes in $dir/graph.

if [ -s $dir/dengraph/HCLG.fst ]; then
   echo "Graph $dir/dengraph/HCLG.fst already exists: skipping graph creation."
else
  utils/mkgraph.sh $dir/lang $alidir $dir/dengraph || exit 1;
fi

## Set up features.
if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=bnf; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  bnf) feats="ark,s,cs:splice-feats --left-context=0 --right-context=0 scp:$sdata/JOB/feats.scp ark:- |";;
  lda) feats="ark,s,cs:splice-feats $splice_opts scp:$sdata/JOB/feats.scp ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
##

if [ -f $alidir/gselect.1.gz ]; then
  gselect_opt="--gselect=ark:gunzip -c $alidir/gselect.JOB.gz|"
else
  echo "$0: no such file $alidir/gselect.1.gz" && exit 1;
fi

if [ -f $alidir/vecs.1 ]; then
  spkvecs_opt="--spk-vecs=ark:$alidir/vecs.JOB --utt2spk=ark:$sdata/JOB/utt2spk"
else
  if [ -f $alidir/final.alimdl ]; then
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
     --max-mem=$max_mem --max-active=$max_active --word-symbol-table=$lang/words.txt $alidir/final.mdl  \
     $dir/dengraph/HCLG.fst "$feats" "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
else
  for n in `seq $nj`; do
    if [ -f $dir/.done.$n ] && [ $dir/.done.$n -nt $alidir/final.mdl ]; then
      echo "Not processing subset $n as already done (delete $dir/.done.$n if not)";
    else 
      sdata2=$data/split$nj/$n/split$sub_split;
      if [ ! -d $sdata2 ] || [ $sdata2 -ot $sdata/$n/feats.scp ]; then
        split_data.sh --per-utt $sdata/$n $sub_split || exit 1;
      fi
      mkdir -p $dir/log/$n
      mkdir -p $dir/part
      feats_subset=`echo $feats | sed "s/trans.JOB/trans.$n/g" | sed s:JOB/:$n/split$sub_split/JOB/:g`
      spkvecs_opt_subset=`echo $spkvecs_opt | sed "s/JOB/$n/g"`
      gselect_opt_subset=`echo $gselect_opt | sed "s/JOB/$n/g"`
      $cmd JOB=1:$sub_split $dir/log/$n/decode_den.JOB.log \
        sgmm2-latgen-faster $spkvecs_opt_subset "$gselect_opt_subset" \
          --beam=$beam --lattice-beam=$lattice_beam \
          --acoustic-scale=$acwt --max-mem=$max_mem --max-active=$max_active \
          --word-symbol-table=$lang/words.txt $alidir/final.mdl  \
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
