#!/bin/bash
#

# To be run from one directory above this script.

# This script takes no arguments.  It assumes you have already run
# swbd_p1_data_prep.sh.  
# It takes as input the files
#data/local/train.txt
#data/local/lexicon.txt
dir=data/local/lm
mkdir -p $dir
export LC_ALL=C # You'll get errors about things being not sorted, if you
# have a different locale.
export PATH=$PATH:`pwd`/../../../tools/kaldi_lm
( # First make sure the kaldi_lm toolkit is installed.
 cd ../../../tools || exit 1;
 if [ -d kaldi_lm ]; then
   echo Not installing the kaldi_lm toolkit since it is already there.
 else
   echo Downloading and installing the kaldi_lm tools
   if [ ! -f kaldi_lm.tar.gz ]; then
     wget http://merlin.fit.vutbr.cz/kaldi/kaldi_lm.tar.gz || exit 1;
   fi
   tar -xvzf kaldi_lm.tar.gz || exit 1;
   cd kaldi_lm
   make || exit 1;
   echo Done making the kaldi_lm tools
 fi
) || exit 1;

mkdir -p $dir

cat data/local/train.txt | awk '{for(n=2;n<=NF;n++) print $n; }' | sort | uniq -c | \
   sort -nr > $dir/word.counts

# We don't have OOVs in the transcripts.  This next command just verifies this
# (otherwise we'd have to decide what to do about them).
! awk -v lex=data/local/lexicon.txt 'BEGIN{while(getline<lex){ seen[$1]=1; } } {if(!seen[$2])print;}' \
   <$dir/word.counts | cmp - /dev/null  && echo Error: OOVs present in transcripts && exit 1


# Get counts from acoustic training transcripts, and add  one-count
# for each word in the lexicon (but not silence, we don't want it
# in the LM-- we'll add it optionally later).
cat data/local/train.txt | awk '{for(n=2;n<=NF;n++) print $n; }' | \
  cat - <(grep -w -v '!SIL' data/local/lexicon.txt | awk '{print $1}') | \
   sort | uniq -c | sort -nr > $dir/unigram.counts

# note: we probably won't really make use of <UNK> as there aren't any OOVs
cat $dir/unigram.counts  | awk '{print $2}' | get_word_map.pl "<s>" "</s>" "<UNK>" > $dir/word_map

# note: ignore 1st field of train.txt, it's the utterance-id.
cat data/local/train.txt | awk -v wmap=$dir/word_map 'BEGIN{while((getline<wmap)>0)map[$1]=$2;}
  { for(n=2;n<=NF;n++) { printf map[$n]; if(n<NF){ printf " "; } else { print ""; }}}' | gzip -c >$dir/train.gz

train_lm.sh --arpa --lmtype 3gram-mincount $dir

# LM is small enough that we don't need to prune it (only about 0.7M N-grams).
# Perplexity over 128254.000000 words is 90.446690

# note: output is
# data/local/lm/3gram-mincount/lm_unpruned.gz 

exit 0


# From here is some commands to do a baseline with SRILM (assuming
# you have it installed).
heldout_sent=10000 # Don't change this if you want result to be comparable with
    # kaldi_lm results
sdir=$dir/srilm # in case we want to use SRILM to double-check perplexities.
mkdir -p $sdir
cat data/local/train.txt | awk '{for(n=2;n<=NF;n++){ printf $n; if(n<NF) printf " "; else print ""; }}' | \
  head -$heldout_sent > $sdir/heldout
cat data/local/train.txt | awk '{for(n=2;n<=NF;n++){ printf $n; if(n<NF) printf " "; else print ""; }}' | \
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
