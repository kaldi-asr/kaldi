#!/bin/bash

order=8

. ./path.sh
. ./utils/parse_options.sh

if [ $# -lt 1 ]; then
  echo "Usage: $0 <lm-dir>";
  exit 1;
fi

dir=$1

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


set -o errexit
mkdir -p $dir
export LC_ALL=C

src=data/local/local_lm/data/text
wordlist=data/local/local_lm/data/wordlist

cat $src/{iam,lob,brown}.txt | gzip -c > $dir/train.gz
cp $src/dev.txt $dir/heldout

ngram-count -text $dir/train.gz -order $order \
            -kndiscount -interpolate -lm $dir/sw1.o${order}g.kn.gz

echo "PPL for ${order}-gram LM:"
ngram -order $order -lm $dir/sw1.o${order}g.kn.gz -ppl $dir/heldout
#ngram -lm $dir/sw1.o${order}g.kn.gz -ppl $dir/heldout -debug 2 >& $dir/3gram.ppl2
