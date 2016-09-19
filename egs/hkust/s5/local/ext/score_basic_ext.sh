#!/bin/bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012-2013. 
#           Hong Kong University of Science and Technology (Author: Ricky Chan Ho Yin). Apache 2.0.
#
# A Chinese character error rate (CER) scoring script for Kaldi; 
#
# By default the script assumes the input Chinese words/characters are 3 bytes UTF8 code \
# and continuous printable English/ASCII characters without space are treated as single token.
#
# When --useword2charmap option is applied, an input Chinese words to Chinese characters mapping table \
# (e.g. a word2char_map likes "195k_chinese_word2char_map" - 1st column of 195k_chinese_word2char_map contains chinese words, \
# 2nd to end column fields are corresponding seperated chinese characters) \
# is used for converting the corresponding Chinese words to seperate Chinese characters for scoring. \
# Please make sure the Chinese words in the mapping table is a superset of the Chinese words for decoding if you apply this option.

# begin configuration section.
cmd=run.pl
min_lmwt=7
max_lmwt=17
useword2charmap=
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score_basic_ext.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  echo "    --useword2charmap word2charmap  # use a chinese word to chinese characters mapping "
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
if [ -z $useword2charmap  ]; then
  perl local/ext/hkust_word2ch_tran.pl $dir/scoring/text.filt > $dir/scoring/cchar.filt
  # perl local/ext/hkust_word2ch_tran.pl --encodeoutput $dir/scoring/text.filt > $dir/scoring/cchar.filt
  for lmwt in `seq $min_lmwt $max_lmwt`; do
    perl local/ext/hkust_word2ch_tran.pl $dir/scoring/$lmwt.txt > $dir/scoring/${lmwt}.cchar
    # perl local/ext/hkust_word2ch_tran.pl --encodeoutput $dir/scoring/$lmwt.txt > $dir/scoring/${lmwt}.cchar
  done
else 
  perl local/ext/hkust_word2ch_tran.pl --useword2charmap $useword2charmap $dir/scoring/text.filt > $dir/scoring/cchar.filt
  # perl local/ext/hkust_word2ch_tran.pl --useword2charmap local/ext/195k_chinese_word2char_map $dir/scoring/text.filt > $dir/scoring/cchar.filt
  for lmwt in `seq $min_lmwt $max_lmwt`; do
    perl local/ext/hkust_word2ch_tran.pl --useword2charmap $useword2charmap $dir/scoring/$lmwt.txt > $dir/scoring/${lmwt}.cchar
    # perl local/ext/hkust_word2ch_tran.pl --useword2charmap local/ext/195k_chinese_word2char_map $dir/scoring/$lmwt.txt > $dir/scoring/${lmwt}.cchar
  done
fi

export LC_ALL=C

$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
  compute-wer --text --mode=present \
   ark:$dir/scoring/text.filt ark:$dir/scoring/LMWT.txt ">&" $dir/wer_LMWT || exit 1;

$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.cer.log \
  compute-wer --text --mode=present \
   ark:$dir/scoring/cchar.filt ark:$dir/scoring/LMWT.cchar ">&" $dir/cer_LMWT || exit 1;

for lmwt in `seq $min_lmwt $max_lmwt`; do
  sed 's/%WER /%CER /g' $dir/cer_${lmwt} > $dir/tmpf; mv $dir/tmpf $dir/cer_${lmwt};
done

exit 0
