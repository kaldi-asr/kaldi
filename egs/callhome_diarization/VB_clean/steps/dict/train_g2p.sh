#!/bin/bash
# Copyright 2014  Johns Hopkins University (Author: Yenda Trmal)
# Copyright 2016  Xiaohui Zhang
# Apache 2.0

# Begin configuration section.
iters=5
stage=0
encoding='utf-8'
only_words=true
cmd=run.pl
# a list of silence phones, like data/local/dict/silence_phones.txt
silence_phones=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. utils/parse_options.sh || exit 1;

set -u
set -e

if [ $# != 2 ]; then
   echo "Usage: $0 [options] <lexicon-in> <work-dir>"
   echo "    where <lexicon-in> is the training lexicon (one pronunciation per "
   echo "    word per line) and <word-dir> is directory where the models will "
   echo "    be stored"
   echo "e.g.: train_g2p.sh data/local/lexicon.txt exp/g2p/"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --iters <int>                                    # How many iterations. Relates to N-ngram order"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

lexicon=$1
wdir=$2


mkdir -p $wdir/log

[ ! -f $lexicon ] && echo "$0: Training lexicon does not exist." && exit 1

# Optionally remove words that are mapped to a single silence phone from the lexicon.
if $only_words && [ ! -z "$silence_phones" ]; then
  awk -v s=$silence_phones \
    'BEGIN{while((getline<s)>0) {for(i=1;i<=NF;i++) sil[$i]=1;}}
    {if (!(NF == 2 && $2 in sil)) print;}' $lexicon > $wdir/lexicon_onlywords.txt
  lexicon=$wdir/lexicon_onlywords.txt
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
