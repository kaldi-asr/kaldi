#!/bin/bash
# Copyright 2014  Johns Hopkins University (Author: Yenda Trmal)
# Apache 2.0

# Begin configuration section.  
iters=5
stage=0
encoding='utf-8'
remove_tags=true
only_words=true
icu_transform="Any-Lower"
cmd=run.pl
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -u
set -e

if [ $# != 2 ]; then
   echo "Usage: $0 [options] <lexicon-path> <work-dir>"
   echo "... where <lexicon-dir> is where you have the lexicon in the usuall "
   echo "    format (one pronunciation per word per line) and <work-dir> is  "
   echo "    directory where the models will be stored.                      "
   echo "e.g.: train_g2p.sh data/local exp/g2p/"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --iters <int>                                    # How many iterations. Relates to N-ngram order"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
wdir=$2


mkdir -p $wdir/log

lexicon=$data/lexicon.txt

[ ! -f $lexicon ] && echo "File lexicon.txt not found in the data directory ($data)." && exit 1

if $only_words ; then
  cat $lexicon | sed 's/^<.*>.*$//g' | sed 's/^#.*//g' > $wdir/lexicon_onlywords.txt
  lexicon=$wdir/lexicon_onlywords.txt
fi

if $remove_tags ; then
  cat $lexicon | sed "s/_[\"1-9]//g" > $wdir/lexicon_notags.txt
  lexicon=$wdir/lexicon_notags.txt
fi

if [ ! -z $icu_transform ] ; then
  paste \
    <(cat $lexicon | awk '{print $1}' | uconv -f $encoding -t $encoding -x "$icu_transform") \
    <(cat $lexicon | sed 's/^[^ \t][^ \t]*[ \t]//g') \
  > $wdir/lexicon_transformed.txt
  lexicon=$wdir/lexicon_transformed.txt
fi

if ! g2p=`which g2p.py` ; then
  echo "Sequitur was not found !"
  echo "Go to $KALDI_ROOT/tools and execute extras/install_sequitur.sh"
  exit 1
fi

echo "Training the G2P model (iter 0)"

if [ $stage -le 0 ]; then
  $cmd $wdir/log/g2p.0.log \
    g2p.py -S --encoding $encoding --train $lexicon --devel 5% --write-model $wdir/g2p.model.0
fi

for i in `seq 0 $(($iters-2))`; do
  
  echo "Training the G2P model (iter $[$i + 1] )"

  if [ $stage -le $i ]; then
    $cmd $wdir/log/g2p.$(($i + 1)).log \
      g2p.py -S --encoding $encoding --model $wdir/g2p.model.$i --ramp-up --train $lexicon --devel 5% --write-model $wdir/g2p.model.$(($i+1))
  fi

done

! (set -e; cd $wdir; ln -sf g2p.model.$[$iters-1] g2p.model.final ) && echo "Problem finalizing training... " && exit 1

if [ $stage -le $(($i + 2)) ]; then
  echo "Running test..."
  $cmd $wdir/log/test.log \
    g2p.py --encoding $encoding --model $wdir/g2p.model.final --test $lexicon
fi

