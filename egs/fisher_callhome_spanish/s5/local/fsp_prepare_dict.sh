#!/usr/bin/env bash
# Copyright 2014  Gaurav Kumar.   Apache 2.0

. ./path.sh

#First get the list of unique words from our text file
if [ $# -lt 1 ]; then
  echo 'Usage fsp_prepare_dict.sh lexicon'
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

#Get all unique words, remove punctuation.
if [ $stage -le 0 ]; then
  cat $datadir/text | sed 's:[0-9][0-9]\S*::g' | sed 's:[\.,\?]::g' | tr " " "\n" | sort | uniq | awk '{if (NF > 0){ print; }}' > $tmpdir/uniquewords
  if [ ! -f "${tmpdir}/es_wordlist.json" ]; then
    echo "Could not find the large collection of Spanish words es_wordlist.json"
    echo "Trying to download it via wget"

    if ! which wget >&/dev/null; then
      echo "This script requires you to first install wget"
      exit 1;
    fi

    cwd=`pwd`
    cd $tmpdir
    wget -T 10 -t 3 -c http://www.openslr.org/resources/21/es_wordlist.json.tgz

    if [ ! -e ${tmpdir}/es_wordlist.json.tgz ]; then
      echo "Download of the large Spanish word list failed"
      exit 1;
    fi

    tar -xovzf es_wordlist.json.tgz || exit 1;
    cd $cwd
  fi

  # Merge with gigaword corpus
  $local/merge_lexicons.py ${tmpdir} ${lexicon}
  mv $tmpdir/uniquewords $tmpdir/uniquewords.small
  mv $tmpdir/uniquewords64k $tmpdir/uniquewords
fi

#Then get the list of phones form basic_rules in the lexicon folder
if [ $stage -le 1 ]; then
  if [ ! -d "$lexicon/callhome_spanish_lexicon_970908" ]; then
    echo "Could not find folder callhome_spanish_lexicon_970908 in the lexicon folder"
    exit 1;
  fi

  # This is a preliminary attempt to get the unique phones from the LDC lexicon
  # This will be extended based on our lexicon later
  perl $local/find_unique_phones.pl $lexicon/callhome_spanish_lexicon_970908 $tmpdir

fi

#Get pronunciation for each word using the spron.pl file in the lexicon folder
if [ $stage -le 2 ]; then
  #cd $lexicon/callhome_spanish_lexicon_970908
  # Replace all words for which no pronunciation was generated with an orthographic
  # representation
  cat $tmpdir/uniquewords | $local/spron.pl $lexicon/callhome_spanish_lexicon_970908/preferences $lexicon/callhome_spanish_lexicon_970908/basic_rules \
    | cut -f1 | sed -r 's:#\S+\s\S+\s\S+\s\S+\s(\S+):\1:g' \
    | awk -F '[/][/]' '{print $1}' \
    > $tmpdir/lexicon_raw
fi

#Break the pronunciation down according to the format required by Kaldi
if [ $stage -le 3 ]; then
  # Creates a KALDI compatible lexicon, and extends the phone list
  perl $local/isolate_phones.pl $tmpdir
  cat $tmpdir/phones_extended | sort | awk '{if ($1 != "") {print;}}' > $tmpdir/phones_extended.1
  mv $tmpdir/phones $tmpdir/phones.small
  mv $tmpdir/phones_extended.1 $tmpdir/phones
  sort $tmpdir/phones -o $tmpdir/phones
  paste -d ' ' $tmpdir/uniquewords $tmpdir/lexicon_one_column | sed -r 's:(\S+)\s#.*:\1 oov:g' > $tmpdir/lexicon.1
  #paste -d ' ' $tmpdir/uniquewords $tmpdir/lexicon_one_column | grep -v '#' > $tmpdir/lexicon.1
fi

if [ $stage -le 4 ]; then
  # silence phones, one per line.
  for w in sil laughter noise oov; do echo $w; done > $dir/silence_phones.txt
  echo sil > $dir/optional_silence.txt

  # An extra question will be added by including the silence phones in one class.
  cat $dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > \
  $dir/extra_questions.txt || exit 1;

  # Remove [] chars from phones
  cat $tmpdir/phones | awk '{if ($1 != "_" && $1 != "[" && $1 != "]") {print;}}' > $tmpdir/phones.1
  rm $tmpdir/phones
  mv $tmpdir/phones.1 $tmpdir/phones
  cp $tmpdir/phones $dir/nonsilence_phones.txt

  if [ -f $tmpdir/lexicon.2 ]; then rm $tmpdir/lexicon.2; fi
  cp "$tmpdir/lexicon.1" "$tmpdir/lexicon.2"

  # Add prons for laughter, noise, oov
  w=$(grep -v sil $dir/silence_phones.txt | tr '\n' '|')
  perl -i -ne "print unless /\[(${w%?})\]/"  $tmpdir/lexicon.2

  for w in `grep -v sil $dir/silence_phones.txt`; do
    echo "[$w] $w"
  done | cat - $tmpdir/lexicon.2  > $tmpdir/lexicon.3 || exit 1;

  cat $tmpdir/lexicon.3  \
   <( echo "mm m"
      echo "<unk> oov" ) > $tmpdir/lexicon.4

  # From the lexicon remove _ from the phonetic representation
  cat $tmpdir/lexicon.4 | sed 's:\s_::g' > $tmpdir/lexicon.5

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
fi

$utils/validate_dict_dir.pl $dir
exit 0;
