#!/usr/bin/env bash
# Copyright (c) 2018, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
. ./utils/parse_options.sh

. ./path.sh

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error


# The parts of the output of this that will be needed are
# [in data/local/dict/ ]
# lexicon.txt
# extra_questions.txt
# nonsilence_phones.txt
# optional_silence.txt
# silence_phones.txt


# check existing directories
[ $# != 1 ] && echo "Usage: $0" && exit 1;

dir=$1

mkdir -p $dir
echo "$0: Getting CMU dictionary"
if [ ! -f $dir/cmudict.done ]; then
  [ -d $dir/cmudict ] && rm -rf $dir/cmudict
  svn co https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict $dir/cmudict
  touch $dir/cmudict.done
fi

# silence phones, one per line.
for w in sil spn inaudible laughs noise; do
  echo $w;
done > $dir/silence_phones.txt
echo sil > $dir/optional_silence.txt

# For this setup we're discarding stress.
cat $dir/cmudict/cmudict-0.7b.symbols | \
  perl -ne 's:[0-9]::g; s:\r::; print lc($_)' | \
  sort -u > $dir/nonsilence_phones.txt

# An extra question will be added by including the silence phones in one class.
paste -d ' ' -s $dir/silence_phones.txt > $dir/extra_questions.txt

grep -v ';;;' $dir/cmudict/cmudict-0.7b |\
  uconv -f latin1 -t utf-8 -x Any-Lower |\
  perl -ne 's:(\S+)\(\d+\) :$1 :; s:  : :; print;' |\
  perl -ne '@F = split " ",$_,2; $F[1] =~ s/[0-9]//g; print "$F[0] $F[1]";' \
  > $dir/lexicon1_raw_nosil.txt || exit 1;

# Add prons for laughter, noise, oov
for w in `grep -v sil $dir/silence_phones.txt`; do
  echo "[$w] $w"
done | cat - $dir/lexicon1_raw_nosil.txt > $dir/lexicon2_raw.txt || exit 1;

# we keep all words from the cmudict in the lexicon
# might reduce OOV rate on dev and eval
cat $dir/lexicon2_raw.txt  \
   <( echo "mm m"
      echo "<unk> spn"
      echo "cuz k aa z"
      echo "cuz k ah z"
      echo "cuz k ao z"
      echo "mmm m"; \
      echo "hmm hh m"; \
    ) | sort -u | sed 's/[\t ]/\t/' > $dir/iv_lexicon.txt


cat data/train*/text  | \
  awk '{for (n=2;n<=NF;n++){ count[$n]++; } } END { for(n in count) { print count[n], n; }}' | \
  sort -nr > $dir/word_counts

cat $dir/word_counts | awk '{print $2}' > $dir/word_list

awk '{print $1}' $dir/iv_lexicon.txt | \
  perl -e '($word_counts)=@ARGV;
   open(W, "<$word_counts")||die "opening word-counts $word_counts";
   while(<STDIN>) { chop; $seen{$_}=1; }
   while(<W>) {
     ($c,$w) = split;
     if (!defined $seen{$w}) { print; }
   } ' $dir/word_counts > $dir/oov_counts.txt

echo "*Highest-count OOVs (including fragments) are:"
head -n 10 $dir/oov_counts.txt
echo "*Highest-count OOVs (excluding fragments) are:"
grep -v -E '^-|-$' $dir/oov_counts.txt | head -n 10 || true

echo "*Training a G2P and generating missing pronunciations"
mkdir -p $dir/g2p/
phonetisaurus-align --input=$dir/iv_lexicon.txt --ofile=$dir/g2p/aligned_lexicon.corpus
ngram-count -order 4 -kn-modify-counts-at-end -ukndiscount\
  -gt1min 0 -gt2min 0 -gt3min 0 -gt4min 0 \
  -text $dir/g2p/aligned_lexicon.corpus -lm $dir/g2p/aligned_lexicon.arpa
phonetisaurus-arpa2wfst --lm=$dir/g2p/aligned_lexicon.arpa --ofile=$dir/g2p/g2p.fst
awk '{print $2}' $dir/oov_counts.txt > $dir/oov_words.txt
phonetisaurus-apply --nbest 2 --model $dir/g2p/g2p.fst --thresh 5 --accumulate \
  --word_list $dir/oov_words.txt > $dir/oov_lexicon.txt

## The next section is again just for debug purposes
## to show words for which the G2P failed
cat $dir/oov_lexicon.txt $dir/iv_lexicon.txt | sort -u > $dir/lexicon.txt
rm -f $dir/lexiconp.txt 2>/dev/null; # can confuse later script if this exists.
awk '{print $1}' $dir/lexicon.txt | \
  perl -e '($word_counts)=@ARGV;
   open(W, "<$word_counts")||die "opening word-counts $word_counts";
   while(<STDIN>) { chop; $seen{$_}=1; }
   while(<W>) {
     ($c,$w) = split;
     if (!defined $seen{$w}) { print; }
   } ' $dir/word_counts > $dir/oov_counts.g2p.txt

echo "*Highest-count OOVs (including fragments) after G2P are:"
head -n 10 $dir/oov_counts.g2p.txt

utils/validate_dict_dir.pl $dir
exit 0;

