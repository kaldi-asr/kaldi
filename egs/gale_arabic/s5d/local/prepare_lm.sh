#!/usr/bin/env bash

# Copyright 2012  Vassil Panayotov
#           2017  Ewald Enzinger
# Apache 2.0

. ./path.sh || exit 1

echo "=== Building a language model ..."

dir=data/local/lm/
text=data/train/text
lexicon=data/local/dict/lexicon.txt
arabic_giga_dir=Arabic_giga
# Language model order
order=4

. utils/parse_options.sh

# Prepare a LM training corpus from the transcripts
mkdir -p $dir

for f in "$text" "$lexicon"; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

loc=`which ngram-count`;
if [ -z $loc ]; then
  if uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
    sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64 
  else
    sdir=$KALDI_ROOT/tools/srilm/bin/i686
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


#cat Arabic_giga/text.1000000 > $dir/text.txt
[ -f $dir/text.txt ] && rm $dir/text.txt && echo "deleted"
cat data/train/text | cut -d " " -f2- > $dir/text.txt
cat Arabic_giga/text.all >> $dir/text.txt
echo "text.txt contains `wc -l $dir/text.txt` lines"
cut -d' ' -f1 $lexicon > $dir/wordlist

ngram-count -text $dir/text.txt -order $order -limit-vocab -vocab $dir/wordlist \
  -unk -map-unk "<UNK>" -kndiscount -interpolate -lm $dir/lm.$order.all.gz

ngram -lm $dir/lm.$order.all.gz -ppl $dir/dev.txt
echo "*** Finished building the LM model!"
