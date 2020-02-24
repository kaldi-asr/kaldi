#!/usr/bin/env bash


# To be run from one directory above this script.
ngram_order=4
oov_sym="<UNK>"
prune_thres=1e-9
[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: [--ngram-order] [--prune-thres] <lm-src-dir> <dict-dir> <lm-dir> <heldout>"
  echo "E.g. $0 --ngram-order 4 --prune-thres 1e-9 data/local/train data/local/dict
  data/local/lm_no_extra datal/local/dev/text"
  exit 1
fi

text=$1/text
dict_dir=$2
dir=$3
dev_text=$4


[ ! -d $dir ] && mkdir -p $dir && exit 1;
[ ! -f $text ] && echo "$0: No such file $text" && exit 1;

lexicon=$dict_dir/lexicon.txt
[ ! -f $lexicon ] && echo "$0: No such file $lexicon" && exit 1;


cleantext=$dir/text.no_oov

cat $text | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } }
   {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ",$n);} } printf("\n");}' \
   > $cleantext || exit 1;


cat $cleantext | awk '{for(n=1;n<=NF;n++) print $n; }' | sort | uniq -c | \
    sort -nr > $dir/word.counts || exit 1;


# Get counts from acoustic training transcripts, and add  one-count
# for each word in the lexicon (but not silence, we don't want it
# in the LM-- we'll add it optionally later).
cat $cleantext | awk '{for(n=1;n<=NF;n++) print $n; }' | \
   cat - <(grep -w -v '!SIL' $lexicon | awk '{print $1}') | \
    sort | uniq -c | sort -nr > $dir/unigram.counts || exit 1;

cat $dir/unigram.counts  | awk '{print $2}' | get_word_map.pl "<s>" "</s>" "<UNK>" > $dir/word_map \
    || exit 1;

cat $dir/word_map | awk '{print $1}' | cat - <(echo "<s>"; echo "</s>" ) \
	> $dir/wordlist

ngram-count -text $dir/text.no_oov -order $ngram_order -limit-vocab -vocab $dir/wordlist -unk \
   -map-unk "<UNK>" -kndiscount -interpolate -prune $prune_thres -lm $dir/srilm.o${ngram_order}g.kn.gz

cut -d " " -f2- $dev_text > $dir/heldout
ngram -lm $dir/srilm.o${ngram_order}g.kn.gz -ppl $dir/heldout > $dir/ppl
# note: output is
# $dir/${ngram_order}gram-mincount/lm_unpruned.gz
echo train lm succeeded
