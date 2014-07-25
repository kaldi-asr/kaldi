#!/usr/bin/env bash

. path.sh

#First get the list of unique words from our text file
if [ $# -lt 1 ]; then
  echo 'Usage callhome_prepare_dict.sh lexicon'
  exit 1;
fi

stage=0

dir=`pwd`/data/local/dict
datadir=`pwd`/data/local/data/train_all
mkdir -p $dir
local=`pwd`/local
utils=`pwd`/utils
tmpdir=`pwd`/data/local/tmp
lexicon=$1

cat local/ldc_arabic_phones.txt | grep -e "^$" -v | awk '{print $1}' > $tmpdir/phones

if [ ! -d "$lexicon/callhome_arabic_lexicon_991012" ]; then
    echo "Could not find folder callhome_arabic_lexicon_991012 in the lexicon folder"
    exit 1;
fi

# Get the two columns of the lexicon we care about
cat $lexicon/callhome_arabic_lexicon_991012/ar_lex.v07 | awk 'BEGIN {OFS="\t"} {print $1,$3};' \
    > $tmpdir/lexicon.1

python local/split_alt_punc.py

cat $tmpdir/lexicon.2 | awk '{print $1}' > $tmpdir/uniquewords
cat $tmpdir/lexicon.2 | awk '{print $2}' > $tmpdir/lexicon_raw

# Break down the phones based on what kaldi needs
perl $local/isolate_phones.pl $tmpdir

cat $tmpdir/phones_extended | sort | awk '{if ($1 != "") {print;}}' > $tmpdir/phones_extended.1
mv $tmpdir/phones $tmpdir/phones.small
mv $tmpdir/phones_extended.1 $tmpdir/phones
sort $tmpdir/phones -o $tmpdir/phones
paste -d ' ' $tmpdir/uniquewords $tmpdir/lexicon_one_column > $tmpdir/lexicon.3 

cp $tmpdir/phones $dir/nonsilence_phones.txt

# silence phones, one per line. 
for w in sil laughter noise oov hes; do echo $w; done > $dir/silence_phones.txt
echo sil > $dir/optional_silence.txt

# An extra question will be added by including the silence phones in one class.
cat $dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > \
$dir/extra_questions.txt || exit 1;

# Add prons for laughter, noise, oov
for w in `grep -v sil $dir/silence_phones.txt`; do
sed -i "/\[$w\]/d" $tmpdir/lexicon.3
done

for w in `grep -v sil $dir/silence_phones.txt`; do
echo "[$w] $w"
done | cat - $tmpdir/lexicon.3  > $tmpdir/lexicon.4 || exit 1;

cat $tmpdir/lexicon.4  \
<( echo "mm m"
  echo "<unk> oov") > $tmpdir/lexicon.5

cp "$tmpdir/lexicon.5" $dir/lexicon.txt

cat $datadir/text  | \
awk '{for (n=2;n<=NF;n++){ count[$n]++; } } END { for(n in count) { print count[n], n; }}' | \
sort -nr > $tmpdir/word_counts

awk '{print $1}' $dir/lexicon.txt | \
perl -e '($word_counts)=@ARGV;
open(W, "<$word_counts")||die "opening word-counts $word_counts";             
while(<STDIN>) { chop; $seen{$_}=1; }
while(<W>) {
 ($c,$w) = split;
 if (!defined $seen{$w}) { print; }                                          
} ' $tmpdir/word_counts > $tmpdir/oov_counts.txt                                    
                                                                             
echo "*Highest-count OOVs are:"                                                  
head -n 20 $tmpdir/oov_counts.txt 

$utils/validate_dict_dir.pl $dir                                                  
exit 0;

