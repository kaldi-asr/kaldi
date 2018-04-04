#!/bin/bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.

# Modified by Takafumi Moriya for Japanese speech recognition using CSJ.
# This script is for scoring with morpheme.

# begin configuration section.
cmd=run.pl
min_lmwt=5
max_lmwt=17
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score_csj.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1;
fi

data=$1
lang=$2
dir=$3

model=$dir/../final.mdl # assume model one level up from decoding dir.

hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl
#hubscr=$KALDI_ROOT/tools/sctk-2.4.0/bin/hubscr.pl
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

for f in $data/text $lang/words.txt $dir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

name=`basename $data`; # e.g. eval1

mkdir -p $dir/scoring/log
mkdir -p $dir/label
mkdir -p $dir/label/log
mkdir -p $dir/label/wer

function filter_text_mor {
  perl -e 'foreach $w (@ARGV) { $bad{$w} = 1; }
   while(<STDIN>) { @A  = split(" ", $_); $id = shift @A; print "$id ";
     foreach $a (@A) { if (!defined $bad{$a}) { print "$a "; }} print "\n"; }' \
   '<UNK>'
}

function filter_text {
  perl -e 'foreach $w (@ARGV) { $bad{$w} = 1; }
   while(<STDIN>) { @A  = split(" ", $_); $id = shift @A; print "$id ";
     foreach $a (@A) { if (!defined $bad{$a}){ @W=split(/\+/,$a); $word=$W[0]; { print "$word "; }}} print "\n"; }' \
   '<UNK>'
}


$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/best_path.LMWT.log \
  lattice-best-path --lm-scale=LMWT --word-symbol-table=$lang/words.txt \
    "ark:gunzip -c $dir/lat.*.gz|" ark,t:$dir/scoring/LMWT.tra || exit 1;

for lmwt in `seq $min_lmwt $max_lmwt`; do
  utils/int2sym.pl -f 2- $lang/words.txt <$dir/scoring/$lmwt.tra | \
   filter_text > $dir/scoring/$lmwt.txt || exit 1;

  utils/int2sym.pl -f 2- $lang/words.txt <$dir/scoring/$lmwt.tra | \
   filter_text_mor > $dir/label/${lmwt}-trans.text || exit 1;
done

filter_text <$data/text >$dir/scoring/text.filt
filter_text_mor <$data/text >$dir/label/text.filt

$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
  compute-wer --text --mode=present \
   ark:$dir/scoring/text.filt ark:$dir/scoring/LMWT.txt ">&" $dir/wer_LMWT || exit 1;

$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
  compute-wer --text --mode=present \
   ark:$dir/label/text.filt ark:$dir/label/LMWT-trans.text ">&" $dir/label/wer/wer_LMWT || exit 1;

exit 0
