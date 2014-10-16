#!/bin/bash
export LC_ALL=C

words_file=
train_text=
dev_text=

. ./utils/parse_options.sh

echo "-------------------------------------"
echo "Building an SRILM language model     "
echo "-------------------------------------"

datadir=$1
tgtdir=$2
outlm=lm.gz

##End of configuration
loc=`which ngram-count`;
if [ -z $loc ]; then
  if uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
    sdir=`pwd`/../../../tools/srilm/bin/i686-m64 
  else
    sdir=`pwd`/../../../tools/srilm/bin/i686
  fi
  if [ -f $sdir/ngram-count ]; then
    echo Using SRILM tools from $sdir
    export PATH=$PATH:$sdir
  else
    echo You appear to not have SRILM tools installed, either on your path,
    echo or installed in $sdir.  See tools/install_srilm.sh for installation
    echo instructions.
    exit 1
  fi
fi

[ -z $words_file ] && words_file=$datadir/lang/words.txt 
[ -z $train_text ] && train_text=$datadir/train/text
[ -z $dev_text ] && dev_text=$datadir/dev2h/text

echo "Using words file: $words_file"
echo "Using train text: $train_text"
echo "Using dev text  : $dev_text"

for f in $words_file $train_text $dev_text; do
  [ ! -s $f ] && echo "No such file $f" && exit 1;
done

# Prepare the destination directory
mkdir -p $tgtdir

# Extract the word list from the training dictionary; exclude special symbols
sort $words_file | awk '{print $1}' | grep -v '\#0' | grep -v '<eps>' > $tgtdir/vocab
if (($?)); then
  echo "Failed to create vocab from $words_file"
  exit 1
else
  # wc vocab # doesn't work due to some encoding issues
  echo vocab contains `cat $tgtdir/vocab | perl -ne 'BEGIN{$l=$w=0;}{split; $w+=$#_; $w++; $l++;}END{print "$l lines, $w words\n";}'`
fi

# Kaldi transcript files contain Utterance_ID as the first word; remove it
cat $train_text | cut -f2- -d' ' > $tgtdir/train.txt
if (($?)); then
    echo "Failed to create $tgtdir/train.txt from $train_text"
    exit 1
else
    echo "Removed first word (uid) from every line of $train_text"
    # wc text.train train.txt # doesn't work due to some encoding issues
    echo $train_text contains `cat $train_text | perl -ne 'BEGIN{$w=$s=0;}{split; $w+=$#_; $w++; $s++;}END{print "$w words, $s sentences\n";}'`
    echo train.txt contains `cat $tgtdir/train.txt | perl -ne 'BEGIN{$w=$s=0;}{split; $w+=$#_; $w++; $s++;}END{print "$w words, $s sentences\n";}'`
fi

# Kaldi transcript files contain Utterance_ID as the first word; remove it
cat $dev_text | cut -f2- -d' ' > $tgtdir/dev.txt
if (($?)); then
    echo "Failed to create $tgtdir/dev.txt from $dev_text"
    exit 1
else
    echo "Removed first word (uid) from every line of $dev_text"
    # wc text.train train.txt # doesn't work due to some encoding issues
    echo $train_text contains `cat $dev_text | perl -ne 'BEGIN{$w=$s=0;}{split; $w+=$#_; $w++; $s++;}END{print "$w words, $s sentences\n";}'`
    echo $tgtdir/dev.txt contains `cat $tgtdir/dev.txt | perl -ne 'BEGIN{$w=$s=0;}{split; $w+=$#_; $w++; $s++;}END{print "$w words, $s sentences\n";}'`
fi

echo "-------------------"
echo "Good-Turing 3grams"
echo "-------------------"
ngram-count -lm $tgtdir/3gram.gt011.gz -gt1min 0 -gt2min 1 -gt3min 1 -order 3 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/3gram.gt012.gz -gt1min 0 -gt2min 1 -gt3min 2 -order 3 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/3gram.gt022.gz -gt1min 0 -gt2min 2 -gt3min 2 -order 3 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/3gram.gt023.gz -gt1min 0 -gt2min 2 -gt3min 3 -order 3 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort

echo "-------------------"
echo "Kneser-Ney 3grams"
echo "-------------------"
ngram-count -lm $tgtdir/3gram.kn011.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 1 -order 3 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/3gram.kn012.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 2 -order 3 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/3gram.kn022.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 2 -kndiscount3 -gt3min 2 -order 3 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/3gram.kn023.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 2 -kndiscount3 -gt3min 3 -order 3 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort


echo "-------------------"
echo "Good-Turing 4grams"
echo "-------------------"
ngram-count -lm $tgtdir/4gram.gt0111.gz -gt1min 0 -gt2min 1 -gt3min 1 -gt4min 1 -order 4 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/4gram.gt0112.gz -gt1min 0 -gt2min 1 -gt3min 1 -gt4min 2 -order 4 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/4gram.gt0122.gz -gt1min 0 -gt2min 1 -gt3min 2 -gt4min 2 -order 4 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/4gram.gt0123.gz -gt1min 0 -gt2min 1 -gt3min 2 -gt4min 3 -order 4 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/4gram.gt0113.gz -gt1min 0 -gt2min 1 -gt3min 1 -gt4min 3 -order 4 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/4gram.gt0222.gz -gt1min 0 -gt2min 2 -gt3min 2 -gt4min 2 -order 4 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/4gram.gt0223.gz -gt1min 0 -gt2min 2 -gt3min 2 -gt4min 3 -order 4 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort

echo "-------------------"
echo "Kneser-Ney 4grams"
echo "-------------------"
ngram-count -lm $tgtdir/4gram.kn0111.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 1 -kndiscount4 -gt4min 1 -order 4 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/4gram.kn0112.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 1 -kndiscount4 -gt4min 2 -order 4 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/4gram.kn0113.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 1 -kndiscount4 -gt4min 3 -order 4 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/4gram.kn0122.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 2 -kndiscount4 -gt4min 2 -order 4 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/4gram.kn0123.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 2 -kndiscount4 -gt4min 3 -order 4 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/4gram.kn0222.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 2 -kndiscount3 -gt3min 2 -kndiscount4 -gt4min 2 -order 4 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort
ngram-count -lm $tgtdir/4gram.kn0223.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 2 -kndiscount3 -gt3min 2 -kndiscount4 -gt4min 3 -order 4 -text $tgtdir/train.txt -vocab $tgtdir/vocab -unk -sort

echo "--------------------"
echo "Computing perplexity"
echo "--------------------"
(
  for f in $tgtdir/3gram* ; do ( echo $f; ngram -order 3 -lm $f -unk -ppl $tgtdir/dev.txt ) | paste -s -d ' ' ; done 
  for f in $tgtdir/4gram* ; do ( echo $f; ngram -order 4 -lm $f -unk -ppl $tgtdir/dev.txt ) | paste -s -d ' ' ; done
)  | sort  -r -n -k 13 | column -t | tee $tgtdir/perplexities.txt

echo "The perlexity scores report is stored in $tgtdir/perplexities.txt "

#This will link the lowest perplexity LM as the output LM.
#ln -sf $tgtdir/`head -n 1 $tgtdir/perplexities.txt | cut -f 1 -d ' '` $outlm

#A slight modification of the previous approach:
#We look at the two lowest perplexity LMs and use a 3gram LM if one of the two, even if the 4gram is of lower ppl
nof_trigram_lm=`head -n 2 $tgtdir/perplexities.txt | grep 3gram | wc -l`
if [[ $nof_trigram_lm -eq 0 ]] ; then
  lmfilename=`head -n 1 $tgtdir/perplexities.txt | cut -f 1 -d ' '`
elif [[ $nof_trigram_lm -eq 2 ]] ; then
  lmfilename=`head -n 1 $tgtdir/perplexities.txt | cut -f 1 -d ' '` 
else  #exactly one 3gram LM
  lmfilename=`head -n 2 $tgtdir/perplexities.txt | grep 3gram | cut -f 1 -d ' '` 
fi
(cd $tgtdir; ln -sf `basename $lmfilename` $outlm )

