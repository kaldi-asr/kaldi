#!/usr/bin/env bash


# To be run from one directory above this script.


lexicon=data/local/dict/lexicon.txt
[ ! -f $lexicon ] && echo "$0: No such file $lexicon" && exit 1;

# check if sri is installed or no
sri_installed=false
which ngram-count  &>/dev/null
if [[ $? == 0 ]]; then
sri_installed=true
fi

# This script takes no arguments.  It assumes you have already run
# previus steps successfully
# It takes as input the files
#data/local/train.*/text
#data/local/dict/lexicon.txt


export LC_ALL=C # You'll get errors about things being not sorted, if you
# have a different locale.
export PATH=$PATH:$KALDI_ROOT/tools/kaldi_lm
( # First make sure the kaldi_lm toolkit is installed.
 cd $KALDI_ROOT/tools || exit 1;
 if [ -d kaldi_lm ]; then
   echo Not installing the kaldi_lm toolkit since it is already there.
 else
   echo Downloading and installing the kaldi_lm tools
   if [ ! -f kaldi_lm.tar.gz ]; then
     wget http://www.danielpovey.com/files/kaldi/kaldi_lm.tar.gz || exit 1;
   fi
   tar -xvzf kaldi_lm.tar.gz || exit 1;
   cd kaldi_lm
   make || exit 1;
   echo Done making the kaldi_lm tools
 fi
) || exit 1;


dir=data/local/lm
 mkdir -p $dir
 text=data/local/train/text
 [ ! -f $text ] && echo "$0: No such file $text" && exit 1;

 cleantext=$dir/text.no_oov

 cat $text | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } }
   {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ",$n);} } printf("\n");}' \
   > $cleantext || exit 1;


 cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | sort | uniq -c | \
    sort -nr > $dir/word.counts || exit 1;


# Get counts from acoustic training transcripts, and add  one-count
# for each word in the lexicon (but not silence, we don't want it
# in the LM-- we'll add it optionally later).
 cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | \
   cat - <(grep -w -v '!SIL' $lexicon | awk '{print $1}') | \
    sort | uniq -c | sort -nr > $dir/unigram.counts || exit 1;

# note: we probably won't really make use of <UNK> as there aren't any OOVs
 cat $dir/unigram.counts  | awk '{print $2}' | get_word_map.pl "<s>" "</s>" "<UNK>" > $dir/word_map \
    || exit 1;

# note: ignore 1st field of train.txt, it's the utterance-id.
 cat $cleantext | awk -v wmap=$dir/word_map 'BEGIN{while((getline<wmap)>0)map[$1]=$2;}
   { for(n=2;n<=NF;n++) { printf map[$n]; if(n<NF){ printf " "; } else { print ""; }}}' | gzip -c >$dir/train.gz \
    || exit 1;

 train_lm.sh --arpa --lmtype 3gram-mincount $dir || exit 1;

# LM is small enough that we don't need to prune it (only about 0.7M N-grams).
# Perplexity over 128254.000000 words is 90.446690

# note: output is
# data/local/lm/3gram-mincount/lm_unpruned.gz


# From here is some commands to do a baseline with SRILM (assuming
# you have it installed).

if $sri_installed; then

 heldout_sent=10000 # Don't change this if you want result to be comparable with
    # kaldi_lm results
 sdir=$dir/srilm # in case we want to use SRILM to double-check perplexities.
 mkdir -p $sdir
 cat $cleantext | awk '{for(n=2;n<=NF;n++){ printf $n; if(n<NF) printf " "; else print ""; }}' | \
   head -$heldout_sent > $sdir/heldout
 cat $cleantext | awk '{for(n=2;n<=NF;n++){ printf $n; if(n<NF) printf " "; else print ""; }}' | \
   tail -n +$heldout_sent > $sdir/train

 cat $dir/word_map | awk '{print $1}' | cat - <(echo "<s>"; echo "</s>" ) > $sdir/wordlist


 ngram-count -text $sdir/train -order 3 -limit-vocab -vocab $sdir/wordlist -unk \
   -map-unk "<UNK>" -kndiscount -interpolate -lm $sdir/srilm.o3g.kn.gz
 ngram -lm $sdir/srilm.o3g.kn.gz -ppl $sdir/heldout
# 0 zeroprobs, logprob= -250954 ppl= 90.5091 ppl1= 132.482

# Note: perplexity SRILM gives to Kaldi-LM model is same as kaldi-lm reports above.
# Difference in WSJ must have been due to different treatment of <UNK>.
 ngram -lm $dir/3gram-mincount/lm_unpruned.gz  -ppl $sdir/heldout
# 0 zeroprobs, logprob= -250913 ppl= 90.4439 ppl1= 132.379
fi


echo train lm succeeded
