#!/bin/bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012-2013. 
#           Hong Kong University of Science and Technology (Ricky Chan). Apache 2.0.
#
# An alternative chinese character scoring script; If the wordlist (words in the lexicon) used by the decoder is a subset of the words in 195k_chinese_word2char_map (1st column of 195k_chinese_word2char_map contains chinese words, 2nd to end column fields are corresponding seperated chinese characters), scoring with score_basic_ext.sh will give pure chinese character error rate

# begin configuration section.
cmd=run.pl
min_lmwt=9
max_lmwt=20
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

hubscr=$KALDI_ROOT/tools/sctk-2.4.0/bin/hubscr.pl 
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
   '[NOISE]' '[LAUGHTER]' '[VOCALIZED-NOISE]' '<UNK>' '%HESITATION' 'FOREIGNGMM' 'NOISEGMM' 'UNKNOWNGMM' '<s>' '</s>'
}

$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/best_path.LMWT.log \
  lattice-best-path --lm-scale=LMWT --word-symbol-table=$lang/words.txt \
    "ark:gunzip -c $dir/lat.*.gz|" ark,t:$dir/scoring/LMWT.tra || exit 1;

for lmwt in `seq $min_lmwt $max_lmwt`; do
  utils/int2sym.pl -f 2- $lang/words.txt <$dir/scoring/$lmwt.tra | \
   filter_text > $dir/scoring/$lmwt.txt || exit 1;
done

filter_text <$data/text >$dir/scoring/text.filt

unset LC_ALL
#for Chinese character error rate
perl local/ext/hkust_word2ch_tran.pl local/ext/195k_chinese_word2char_map $dir/scoring/text.filt > $dir/scoring/cchar.filt

for lmwt in `seq $min_lmwt $max_lmwt`; do
perl local/ext/hkust_word2ch_tran.pl local/ext/195k_chinese_word2char_map $dir/scoring/$lmwt.txt > $dir/scoring/${lmwt}.cchar
done

export LC_ALL=C

$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
  compute-wer --text --mode=present \
   ark:$dir/scoring/text.filt ark:$dir/scoring/LMWT.txt ">&" $dir/wer_LMWT || exit 1;

$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.cer.log \
  compute-wer --text --mode=present \
   ark:$dir/scoring/cchar.filt ark:$dir/scoring/LMWT.cchar ">&" $dir/cer_LMWT || exit 1;

exit 0
