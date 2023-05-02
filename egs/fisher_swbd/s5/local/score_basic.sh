#!/usr/bin/env bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.

# begin configuration section.
cmd=run.pl
min_lmwt=5
max_lmwt=17
word_ins_penalty=0.0,0.5,1.0
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score_basic.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1;
fi

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
dir=$3

model=$dir/../final.mdl # assume model one level up from decoding dir.

hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

for f in $data/text $lang/words.txt $dir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

name=`basename $data`; # e.g. eval2000

mkdir -p $dir/scoring/log


function filter_text {
  perl -e 'foreach $w (@ARGV) { $bad{$w} = 1; }
   while(<STDIN>) { @A  = split(" ", $_); $id = shift @A; print "$id ";
     foreach $a (@A) { if (!defined $bad{$a}) { print "$a "; }} print "\n"; }' \
   '[NOISE]' '[LAUGHTER]' '[VOCALIZED-NOISE]' '<UNK>' '%HESITATION'
}

for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/best_path.LMWT.${wip}.log \
    lattice-best-path --lm-scale=LMWT --word-symbol-table=$lang/words.txt \
    "ark:gunzip -c $dir/lat.*.gz|" ark,t:$dir/scoring/LMWT.${wip}.tra || exit 1;
done

for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
  for lmwt in `seq $min_lmwt $max_lmwt`; do
    utils/int2sym.pl -f 2- $lang/words.txt <$dir/scoring/$lmwt.${wip}.tra | \
     filter_text > $dir/scoring/$lmwt.${wip}.txt || exit 1;
  done
done

filter_text <$data/text >$dir/scoring/text.filt

for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.${wip}.log \
    compute-wer --text --mode=present \
     ark:$dir/scoring/text.filt ark:$dir/scoring/LMWT.${wip}.txt ">&" $dir/wer_LMWT_${wip} || exit 1;
done
exit 0
