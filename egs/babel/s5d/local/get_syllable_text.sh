#!/bin/bash

# Copyright Johns Hopkins University 2013 (author: Daniel Povey)
# Apache 2.0.

if [ $# -ne 7 ]; then
  echo "Usage: get_syllable_text.sh <data> <lang> <syllable-lang-nopos> <word2syllable-fst> <ali-dir> <tempdir> <tgt-data>"
  echo "e.g.: get_syllable_text.sh data/train data/lang ../s5-vietnamese-limited-syllables/data/lang_nopos \\"
  echo "      ../s5-vietnamese-limited-syllables/data/local/syllables/word2syllable_lexicon_unweighted.fst"
  echo "       exp/tri5h_ali exp/tri5_align_syllables ../s5-vietnamese-limited-syllables/data/train"
  echo "This script copies the data-directory <data> to <tgt-data> but converts the text into syllable-level text."
  echo "The inputs are as follows (those that are not self-explanatory):"
  echo "  <syllable-lang-nopos> is the syllable-level lang/ directory that has been built without"
  echo "   word-position dependency (we'll strip the suffixes from phones and expect them to be compatible with this)"
  echo "  <word2syllable-fst> is a kind of lexicon FST that describes words as syllable sequences."
  echo "  <ali-dir> contains a word-level alignment of the data in <data>"
  echo "  <tempdir> will be used to put temporary files and logs (make it somewhere in exp/)"
  echo "  <tgt-data> is a data directory to put the syllable-level data; transcripts go to <tgt-data>/text"
  exit 1;
fi

[ -f path.sh ] && . ./path.sh

data=$1
lang=$2
lang_nopos=$3
word2syllable_fst=$4
alidir=$5
dir=$6
tgtdata=$7

for f in $data/text $lang/L.fst $lang_nopos/L.fst $word2syllable_fst $alidir/ali.1.gz \
  $alidir/final.mdl $alidir/num_jobs; do
  if [ ! -f $f ]; then
    echo "Expected file $f to exist" 
    exit 1;
  fi
done

mkdir -p $dir/log
nj=`cat $alidir/num_jobs` || exit 1;
sil=`cat data/lang/phones/optional_silence.txt` || exit 1

! ( ( for n in `seq $nj`; do gunzip -c $alidir/ali.$n.gz; done ) | \
  ali-to-phones $alidir/final.mdl ark:- ark,t:- | \
  utils/int2sym.pl -f 2- $lang/phones.txt - | \
  sed -E 's/_I( |$)/ /g' |  sed -E 's/_E( |$)/ /g' | sed -E 's/_B( |$)/ /g' | sed -E 's/_S( |$)/ /g' | \
  utils/sym2int.pl -f 2- $lang_nopos/phones.txt | \
  gzip -c > $dir/phones.ark.gz ) 2>&1 | tee $dir/log/align.log \
  && echo "Error getting phone-level (non-word-position-dependent) alignments" && exit 1;

# Get an archive of syllable-level acceptors corresponding to the training data.
# transcripts.  We don't have an fstproject program for archives so we use a line of awk.

! ( cat $data/text | utils/sym2int.pl --map-oov `cat $lang/oov.int` -f 2- $lang/words.txt | \
  transcripts-to-fsts ark:- ark:- | \
  fsttablecompose $word2syllable_fst ark:- ark,t:- | \
  awk '{if (NF < 4) { print; } else { print $1, $2, $3, $3, $5; }}' | \
  gzip -c > $dir/syllables.ark.gz ) 2>&1 | tee $dir/log/get_syllable_fsts.log && \
 echo "Error getting syllable FSTs" && exit 1;

cp -rT $data $tgtdata || exit 1;
rm -rf $tgtdata/split*

# From the phone-level transcripts and the syllable-level acceptors, work out
# the syllable sequence for each .  Remove consecutive silences.
! ( fsttablecompose $lang_nopos/L.fst "ark:gunzip -c $dir/syllables.ark.gz|" ark:- | \
  fsttablecompose "ark:gunzip -c $dir/phones.ark.gz | transcripts-to-fsts ark:- ark:- |" \
  ark,s,cs:- ark,t:- | fsts-to-transcripts ark:- ark,t:- | int2sym.pl -f 2- $lang_nopos/words.txt | \
  sed "s/$sil $sil/$sil/g" > $tgtdata/text ) && echo "Error getting text data" && exit 1;

! utils/fix_data_dir.sh $tgtdata/ && echo "Error fixing data dir" && exit 1;

exit 0;



