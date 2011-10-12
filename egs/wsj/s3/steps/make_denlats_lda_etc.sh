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
beam=16.0 # was 13
latticebeam=8.0 # was 7
acwt=0.1
maxactive=5000

for x in 1 2 3; do
  if [ "$1" == "--num-jobs" ]; then
     shift
     nj=$1
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
  
scripts/mkgraph.sh $dir/lang $alidir $dir/dengraph || exit 1;

if [ ! -d $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  scripts/split_data.sh $data $nj
fi

n=`get_splits.pl $nj | awk '{print $1}'`
if [ -f $alidir/$n.trans ]; then
  use_trans=true
  echo Using transforms from directory $dir
else
  echo No transforms present in alignment directory: assuming speaker independent.
  use_trans=false
fi


rm $dir/.error 2>/dev/null
for n in `get_splits.pl $nj`; do
  featspart[$n]="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
  $use_trans && featspart[$n]="${featspart[$n]} transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.trans ark:- ark:- |"

  # Puttings slightly more conservative max-arcs and max-loop, to keep
  # determinization from using too much memory.
  $cmd $dir/decode_den.$n.log \
    gmm-latgen-faster --max-arcs=25000 --max-loop=100000 \
     --beam=$beam --lattice-beam=$latticebeam --acoustic-scale=$acwt \
    --max-active=$maxactive --word-symbol-table=$lang/words.txt $alidir/final.mdl  \
    $dir/dengraph/HCLG.fst "${featspart[$n]}" "ark:|gzip -c >$dir/lat.$n.gz" \
      || touch $dir/.error &
done
wait
[ -f $dir/.error ] && echo Error generating denominator lattices && exit 1;

echo "Done generating denominator lattices."

