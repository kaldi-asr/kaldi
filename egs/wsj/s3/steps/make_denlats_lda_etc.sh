#!/bin/bash

# To be run from ..

# This script generates denominator lattices for MMI training.
# As a language model, it uses a unigram language model with an LM 
# that it estimates from the training transcripts. 

# The features must be raw features + CMN + LDA + [something],
# where [something] could be e.g. MLLT, or speaker-specific
# transforms such as fMLLR or ET (in this case this script
# expects transforms in $alidir/*.trans)

# The output of this script is *.lats.gz

nj=4
cmd=scripts/run.pl
beam=13.0
latticebeam=7.0
acwt=0.1
maxactive=5000
maxmem=20000000 # This will stop the processes getting too large 
# (default is 50M, but this can result in the process getting up to 2G
#  ... the units are not quite "real" units due to inaccuracies in the
# way that program measures how much memory it is using).
subsplit=1 # If this option is given, it will go sequentially over each
    # part of the data, and decode it in parallel with this many jobs.

for x in 1 2 3; do
  if [ "$1" == "--num-jobs" ]; then
     shift
     nj=$1
     shift
  fi
  if [ "$1" == "--sub-split" ]; then
     shift
     subsplit=$1
     shift
  fi
  if [ "$1" == "--cmd" ]; then
     shift
     cmd=$1
     [ "$cmd" == "" ] && echo "Empty string given to --cmd option" && exit 1;
     shift
  fi  
  if [ "$1" == "--beam" ]; then
     shift; beam=$1; shift
  fi  
  if [ "$1" == "--acwt" ]; then
     shift; acwt=$1; shift
  fi  
  if [ "$1" == "--lattice-beam" ]; then
     shift; latticebeam=$1; shift
  fi  
  if [ "$1" == "--max-active" ]; then
     shift; maxactive=$1; shift
  fi  
done

if [ $# != 4 ]; then
   echo "Usage: steps/make_denlats_lda_etc.sh <data-dir> <lang-dir> <ali-dir> <exp-dir>"
   echo "Note: alignment directory only used for CMS and (if needed) transforms"
   echo " e.g.: steps/make_denlats_lda_etc.sh data/train data/lang exp/tri1_ali exp/tri1_denlats"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
lang=$2
alidir=$3
dir=$4

oov_sym=`cat $lang/oov.txt`
silphonelist=`cat $lang/silphones.csl`

mkdir -p $dir

cp -r $lang $dir/

# Compute grammar FST which corresponds to unigram decoding graph.

cat $data/text | \
  scripts/sym2int.pl --map-oov "$oov_sym" --ignore-first-field $lang/words.txt | \
  awk '{for(n=2;n<=NF;n++){ printf("%s ", $n); } printf("\n"); }' | \
  scripts/make_unigram_grammar.pl | fstcompile > $dir/lang/G.fst \
   || exit 1;

# mkgraph.sh expects a whole directory "lang", so put everything in one directory...
# it gets L_disambig.fst and G.fst (among other things) from $dir/lang, and
# final.mdl from $alidir; the output HCLG.fst goes in $dir/graph.


if [ -s $dir/dengraph/HCLG.fst ]; then
   echo Not creating denominator graph $dir/dengraph/HCLG.fst since it already exists.
else
 scripts/mkgraph.sh $dir/lang $alidir $dir/dengraph || exit 1;
fi

if [ ! -d $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  scripts/split_data.sh $data $nj || exit 1;
fi

n=`get_splits.pl $nj | awk '{print $1}'`
if [ -f $alidir/$n.trans ]; then
  use_trans=true
  echo Using transforms from directory $alidir
else
  echo No transforms present in alignment directory: assuming speaker independent.
  use_trans=false
fi


rm $dir/.error 2>/dev/null


if [ $subsplit -eq 1 ]; then 
  for n in `get_splits.pl $nj`; do
    feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    $use_trans && feats="$feats transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.trans ark:- ark:- |"
    $cmd $dir/decode_den.$n.log \
      gmm-latgen-faster --beam=$beam --lattice-beam=$latticebeam --acoustic-scale=$acwt \
      --max-mem=$maxmem --max-active=$maxactive --word-symbol-table=$lang/words.txt $alidir/final.mdl  \
      $dir/dengraph/HCLG.fst "$feats" "ark:|gzip -c >$dir/lat.$n.gz" \
        || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo Error generating denominator lattices && exit 1;
else # Decode each subset of the data with multiple jobs.
  for n in `get_splits.pl $nj`; do
    if [ -f $dir/.done.$n ]; then
      echo Not processing subset $n because file $dir/.done.$n exists # This is so we
    else # can rerun this script without redoing everything, if we succeeded with some parts.
      nk=$subsplit
      if [ ! -d $data/split$nj/$n/split$nk -o $data/split$nj/$n/split$nk -ot $data/split$nj/feats.scp ]; then      
        scripts/split_data.sh $data/split$nj/$n $nk || exit 1;
      fi
    fi
    mkdir -p $dir/log$n
    for o in `get_splits.pl $nk`; do
      if [ ! -s $data/split$nj/$n/split$nk/$o/feats.scp ]; then
        echo "Empty subset; no lines in $data/split$nj/$n/split$nk/$o/feats.scp"
      else 
        feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/split$nk/$o/utt2spk ark,s,cs:$alidir/$n.cmvn scp:$data/split$nj/$n/split$nk/$o/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
        $use_trans && feats="$feats transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark,s,cs:$alidir/$n.trans ark:- ark:- |"
        $cmd $dir/log$n/decode_den.$o.log \
          gmm-latgen-faster --beam=$beam --lattice-beam=$latticebeam --acoustic-scale=$acwt \
          --max-mem=$maxmem --max-active=$maxactive --word-symbol-table=$lang/words.txt $alidir/final.mdl  \
          $dir/dengraph/HCLG.fst "$feats" "ark:|gzip -c >$dir/lat.$n.$o.gz" \
            || touch $dir/.error &
      fi
    done
    wait
    [ -f $dir/.error ] && echo Error generating denominator lattices for subset $n && exit 1;
    echo Merging archives for data subset $n
    for o in `get_splits.pl $nk`; do
      gunzip -c $dir/lat.$n.$o.gz || touch $dir/.error;
    done | gzip -c > $dir/lat.$n.gz || touch $dir/.error;
    [ -f $dir/.error ] && echo Error merging denominator lattices for subset $n && exit 1;
    rm $dir/lat.$n.*.gz
    touch $dir/.done.$n # so we don't re-do it if we run this script again.
  done
fi


echo "Done generating denominator lattices."

