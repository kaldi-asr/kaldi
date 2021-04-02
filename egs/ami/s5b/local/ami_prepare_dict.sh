#!/usr/bin/env bash

#adapted from fisher dict preparation script, Author: Pawel Swietojanski

dir=data/local/dict
mkdir -p $dir
echo "Getting CMU dictionary"
svn co  https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict  $dir/cmudict

# silence phones, one per line. 
for w in sil laughter noise oov; do echo $w; done > $dir/silence_phones.txt
echo sil > $dir/optional_silence.txt

# For this setup we're discarding stress.
cat $dir/cmudict/cmudict.0.7a.symbols | sed s/[0-9]//g | \
  perl -ane 's:\r::; print;' | sort | uniq > $dir/nonsilence_phones.txt

# An extra question will be added by including the silence phones in one class.
cat $dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;

grep -v ';;;' $dir/cmudict/cmudict.0.7a | \
  perl -ane 'if(!m:^;;;:){ s:(\S+)\(\d+\) :$1 :; s:  : :; print; }' | \
  perl -ane '@A = split(" ", $_); for ($n = 1; $n<@A;$n++) { $A[$n] =~ s/[0-9]//g; } print join(" ", @A) . "\n";' | \
  sort | uniq > $dir/lexicon1_raw_nosil.txt || exit 1;

#cat eddie_data/rt09.ami.ihmtrain09.v3.dct | sort > $dir/lexicon1_raw_nosil.txt

# limit the vocabulary to the predefined 50k words
wget -nv -O $dir/wordlist.50k.gz http://www.openslr.org/resources/9/wordlist.50k.gz
gunzip -c $dir/wordlist.50k.gz > $dir/wordlist.50k
join $dir/lexicon1_raw_nosil.txt $dir/wordlist.50k > $dir/lexicon1_raw_nosil_50k.txt

# Add prons for laughter, noise, oov
for w in `grep -v sil $dir/silence_phones.txt`; do
  echo "[$w] $w"
done | cat - $dir/lexicon1_raw_nosil_50k.txt > $dir/lexicon2_raw_50k.txt || exit 1;

# add some specific words, those are only with 100 missing occurences or more
( echo "MM M"; \
  echo "HMM HH M"; \
  echo "MM-HMM M HH M"; \
  echo "COLOUR  K AH L ER"; \
  echo "COLOURS  K AH L ER Z"; \
  echo "REMOTES  R IH M OW T Z"; \
  echo "FAVOURITE F EY V ER IH T"; \
  echo "<unk> oov" ) | cat - $dir/lexicon2_raw_50k.txt \
     | sort -u > $dir/lexicon3_extra_50k.txt

cp $dir/lexicon3_extra_50k.txt $dir/lexicon.txt
rm $dir/lexiconp.txt 2>/dev/null; # can confuse later script if this exists.

[ ! -f $dir/lexicon.txt ] && exit 1;

# This is just for diagnostics:
cat data/ihm/train/text  | \
  awk '{for (n=2;n<=NF;n++){ count[$n]++; } } END { for(n in count) { print count[n], n; }}' | \
  sort -nr > $dir/word_counts

awk '{print $1}' $dir/lexicon.txt | \
  perl -e '($word_counts)=@ARGV;
   open(W, "<$word_counts")||die "opening word-counts $word_counts";
   while(<STDIN>) { chop; $seen{$_}=1; }
   while(<W>) {
     ($c,$w) = split;
     if (!defined $seen{$w}) { print; }
   } ' $dir/word_counts > $dir/oov_counts.txt

echo "*Highest-count OOVs are:"
head -n 20 $dir/oov_counts.txt

utils/validate_dict_dir.pl $dir
