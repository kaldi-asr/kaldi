#!/bin/bash

# steps/make_phone_graph.sh data/train_100k_nodup/ data/lang exp/tri2_ali_100k_nodup/ exp/tri2

# Copyright 2013  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This script makes a phone-based LM, without smoothing to unigram, that 
# is to be used for segmentation, and uses that together with a model to
# make a decoding graph.
# Uses SRILM.

# Begin configuration section.
stage=0
cmd=run.pl
N=3  # change N and P for non-trigram systems.
P=1
tscale=1.0 # transition scale.
loopscale=0.1 # scale for self-loops.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0  [options] <lang-dir> <alignment-dir> <model-dir>"
  echo " e.g.: $0 data/lang exp/tri3b_ali exp/tri4b_seg"
  echo "Makes the graph in $dir/phone_graph, corresponding to the model in $dir"
  echo "The alignments from $ali_dir are used to train the phone LM."
  exit 1;
fi

lang=$1
alidir=$2
dir=$3


for f in $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl $dir/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected $f to exist"
    exit 1;
  fi
done

loc=`which ngram-count`;
if [ -z $loc ]; then
  if uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
    sdir=`pwd`/../../../tools/srilm/bin/i686-m64 
  else
    sdir=`pwd`/../../../tools/srilm/bin/i686
  fi
  if [ -f $sdir/ngram-count ]; then
    echo Using SRILM tools from $sdir
    export PATH=$PATH:$sdir
  else
    echo You appear to not have SRILM tools installed, either on your path,
    echo or installed in $sdir.  See tools/install_srilm.sh for installation
    echo instructions.
    exit 1
  fi
fi

set -e # exit on error status

mkdir -p $dir/phone_graph

if [ $stage -le 0 ]; then
  echo "$0: creating phone LM-training data"
  gunzip -c $alidir/ali.*gz | ali-to-phones $alidir/final.mdl ark:- ark,t:- | \
    awk '{for (x=2; x <= NF; x++) printf("%s ", $x); printf("\n"); }' | \
    utils/int2sym.pl $lang/phones.txt > $dir/phone_graph/train_phones.txt
fi

if [ $stage -le 1 ]; then
  echo "$0: building ARPA LM"
  ngram-count -text $dir/phone_graph/train_phones.txt -order 3  \
    -addsmooth1 1 -kndiscount2 -kndiscount3 -interpolate -lm $dir/phone_graph/arpa.gz
fi

# Set the unigram and unigram-backoff log-probs to -99.  we'll later remove the
# arcs from the FST.  This is to avoid CLG blowup, and to increase speed.

if [ $stage -le 2 ]; then
  echo "$0: removing unigrams from ARPA LM"

  gunzip -c $dir/phone_graph/arpa.gz | \
    awk '/\\1-grams/{state=1;} /\\2-grams:/{ state=2; }
       {if(state == 1 && NF == 3) { printf("-99\t%s\t-99\n", $2); } else {print;}}' | \
         gzip -c >$dir/phone_graph/arpa_noug.gz
fi

if [ $stage -le 3 ]; then
  echo "$0: creating G_phones.fst from ARPA"
  gunzip -c $dir/phone_graph/arpa_noug.gz | arpa2fst - - | fstprint | \
    utils/eps2disambig.pl | utils/s2eps.pl | \
    awk '{if (NF < 5 || $5 < 100.0) { print; }}' | \
    fstcompile --isymbols=$lang/phones.txt --osymbols=$lang/phones.txt \
       --keep_isymbols=false --keep_osymbols=false | \
    fstconnect | \
    fstrmepsilon > $dir/phone_graph/G_phones.fst
   fstisstochastic $dir/phone_graph/G_phones.fst  || echo "[info]: G_phones not stochastic."
fi

  
if [ $stage -le 4 ]; then
  echo "$0: creating CLG."

  fstcomposecontext --context-size=$N --central-position=$P \
   --read-disambig-syms=$lang/phones/disambig.int \
   --write-disambig-syms=$dir/phone_graph/disambig_ilabels_${N}_${P}.int \
    $dir/phone_graph/ilabels_${N}_${P} < $dir/phone_graph/G_phones.fst | \
      fstdeterminize >$dir/phone_graph/CLG.fst
  fstisstochastic $dir/phone_graph/CLG.fst  || echo "[info]: CLG not stochastic."
fi

if [ $stage -le 5 ]; then
  echo "$0: creating Ha.fst"
  make-h-transducer --disambig-syms-out=$dir/phone_graph/disambig_tid.int \
    --transition-scale=$tscale $dir/phone_graph/ilabels_${N}_${P} $dir/tree $dir/final.mdl \
       > $dir/phone_graph/Ha.fst 
fi

if [ $stage -le 6 ]; then
  echo "$0: creating HCLGa.fst"
  fsttablecompose $dir/phone_graph/Ha.fst $dir/phone_graph/CLG.fst | \
      fstdeterminizestar --use-log=true | \
      fstrmsymbols $dir/phone_graph/disambig_tid.int | fstrmepslocal | \
      fstminimizeencoded > $dir/phone_graph/HCLGa.fst || exit 1;
  fstisstochastic $dir/phone_graph/HCLGa.fst || echo "HCLGa is not stochastic"
fi

if [ $stage -le 7 ]; then
  add-self-loops --self-loop-scale=$loopscale --reorder=true \
    $dir/final.mdl < $dir/phone_graph/HCLGa.fst > $dir/phone_graph/HCLG.fst || exit 1;

  if [ $tscale == 1.0 -a $loopscale == 1.0 ]; then
    # No point doing this test if transition-scale not 1, as it is bound to fail. 
    fstisstochastic $dir/phone_graph/HCLG.fst || echo "[info]: final HCLG is not stochastic."
  fi

  # $lang/phones.txt is the symbol table that corresponds to the output
  # symbols on the graph; decoding scripts expect it as words.txt.
  cp $lang/phones.txt $dir/phone_graph/words.txt
  cp -r $lang/phones $dir/phone_graph/
fi
