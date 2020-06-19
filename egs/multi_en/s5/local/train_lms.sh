#!/usr/bin/env bash

###########################################################################################
# This script was copied from egs/fisher_swbd/s5/local/fisher_train_lms.sh
# The source commit was e69198c3dc5633f98eb88e1cdf20b2521a598f21
# Changes made:
#  - Specified path to path.sh
#  - Modified paths to match multi_en naming conventions
###########################################################################################

# To be run from one directory above this script.

. ./path.sh

text=data/local/lm/text
lexicon=data/local/dict_nosp/lexicon.txt 

for f in "$text" "$lexicon"; do
  [ ! -f $x ] && echo "$0: No such file $f" && exit 1;
done

# This script takes no arguments.  It assumes you have already run
# fisher_data_prep.sh and fisher_prepare_dict.sh
# It takes as input the files
#data/train_all/text
#data/local/dict/lexicon.txt

dir=data/local/lm
mkdir -p $dir

kaldi_lm=`which train_lm.sh`
if [ ! -x $kaldi_lm ]; then
  echo "train_lm.sh is not found. Look at tools/extra/install_kaldi_lm.sh"
  exit 1
fi

cleantext=$dir/text.no_oov

cat $text | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<unk> ");} } printf("\n");}' \
  > $cleantext || exit 1;


cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | sort | uniq -c | \
   sort -nr > $dir/word.counts || exit 1;


# Get counts from acoustic training transcripts, and add  one-count
# for each word in the lexicon (but not silence, we don't want it
# in the LM-- we'll add it optionally later).
cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | \
  cat - <(grep -w -v '!SIL' $lexicon | awk '{print $1}') | \
   sort | uniq -c | sort -nr > $dir/unigram.counts || exit 1;

# note: we probably won't really make use of <unk> as there aren't any OOVs
cat $dir/unigram.counts  | awk '{print $2}' | get_word_map.pl "<s>" "</s>" "<unk>" > $dir/word_map \
   || exit 1;

# note: ignore 1st field of train.txt, it's the utterance-id.
cat $cleantext | awk -v wmap=$dir/word_map 'BEGIN{while((getline<wmap)>0)map[$1]=$2;}
  { for(n=2;n<=NF;n++) { printf map[$n]; if(n<NF){ printf " "; } else { print ""; }}}' | gzip -c >$dir/train.gz \
   || exit 1;

train_lm.sh --arpa --lmtype 3gram-mincount $dir || exit 1;
# Perplexity over 88307.000000 words (excluding 691.000000 OOVs) is 71.241332
train_lm.sh --arpa --lmtype 4gram-mincount $dir || exit 1;

# note: output is
# data/local/lm/3gram-mincount/lm_unpruned.gz 


exit 0


# From here is some commands to do a baseline with SRILM (assuming
# you have it installed).
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
  -map-unk "<unk>" -kndiscount -interpolate -lm $sdir/srilm.o3g.kn.gz
ngram -lm $sdir/srilm.o3g.kn.gz -ppl $sdir/heldout 

# data/local/lm/srilm/srilm.o3g.kn.gz: line 71: warning: non-zero probability for <unk> in closed-vocabulary LM
# file data/local/lm/srilm/heldout: 10000 sentences, 78998 words, 0 OOVs
# 0 zeroprobs, logprob= -165170 ppl= 71.7609 ppl1= 123.258


# Note: perplexity SRILM gives to Kaldi-LM model is similar to what kaldi-lm reports above.
# Difference in WSJ must have been due to different treatment of <unk>.
ngram -lm $dir/3gram-mincount/lm_unpruned.gz  -ppl $sdir/heldout 

# data/local/lm/srilm/srilm.o3g.kn.gz: line 71: warning: non-zero probability for <unk> in closed-vocabulary LM
# file data/local/lm/srilm/heldout: 10000 sentences, 78998 words, 0 OOVs
# 0 zeroprobs, logprob= -164990 ppl= 71.4278 ppl1= 122.614
