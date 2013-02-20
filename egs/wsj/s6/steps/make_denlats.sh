#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# Create denominator lattices for MMI/MPE training.
# Creates its output in $dir/lat.*.gz

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
# This is in bytes, but not "real" bytes-- you have to multiply
# by something like 5 or 10 to get real bytes (not sure why so large)
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: steps/make_denlats.sh [options] <data-dir> <lang-dir> <src-dir> <exp-dir>"
   echo "  e.g.: steps/make_denlats.sh data/train data/lang exp/tri1 exp/tri1_denlats"
   echo "Works for CMN+LDA+MLLT features, plus (with --transform-dir option) fMLLR"
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
srcdir=$3
dir=$4

for f in $data/feats.scp $data/cmvn.scp $lang/phones.txt $srcdir/final.mdl $srcdir/final.mat $srcdir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

sdata=$data/split$nj
splice_opts=`cat $srcdir/splice_opts || exit 1`
cmvn_opts=`cat $srcdir/cmvn_opts || exit 1`
mkdir -p $dir/log
split_data.sh $data $nj || exit 1;
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
# final.mdl from $srcdir; the output HCLG.fst goes in $dir/graph.

if [ -s $dir/dengraph/HCLG.fst ]; then
   echo "Graph $dir/dengraph/HCLG.fst already exists: skipping graph creation."
else
  utils/mkgraph.sh $dir/lang $srcdir $dir/dengraph || exit 1;
fi

feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

if [ ! -z "$transform_dir" ]; then # add transforms to features...
  echo "$0: using fMLLR transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "Expected $transform_dir/trans.1 to exist."
  [ "`cat $transform_dir/num_jobs`" -ne "$nj" ] \
    && echo "$0: mismatch in number of jobs with $transform_dir" && exit 1;
  [ -f $srcdir/final.mat ] && ! cmp $transform_dir/final.mat $srcdir/final.mat && \
     echo "$0: LDA transforms differ between $srcdir and $transform_dir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/trans.JOB ark:- ark:- |"
else
  if [ -f $srcdir/final.alimdl ]; then
    echo "$0: you seem to have a SAT system but you did not supply the --transform-dir option.";
    exit 1;
  fi
fi

if [ $sub_split -eq 1 ]; then 
  $cmd JOB=1:$nj $dir/log/decode_den.JOB.log \
   gmm-latgen-faster --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
    --max-mem=$max_mem --max-active=$max_active --word-symbol-table=$lang/words.txt $srcdir/final.mdl  \
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
      $cmd JOB=1:$sub_split $dir/log/$n/decode_den.JOB.log \
        gmm-latgen-faster --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
        --max-mem=$max_mem --max-active=$max_active --word-symbol-table=$lang/words.txt $srcdir/final.mdl  \
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


echo "$0: done generating denominator lattices."
