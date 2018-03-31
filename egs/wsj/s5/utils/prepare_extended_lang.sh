#!/bin/bash
# Copyright 2018  Xiaohui Zhang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This script adds word-position-dependent phones and constructs a host of other
# derived files, that go in data/lang/.

# Begin configuration section.
oov_unigram_prob=0.0001 # Unigram LM probability we'll assign to each word outside the original vocab.
srilm_opts="-subset -prune-lowprobs -unk"
prep_lang_opts=
lm=
to_carpa=false # convert the language model to const arpa format rather than FST 
stage=0
# end configuration sections

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh

if [ $# -ne 7 ]; then
  echo "usage: utils/prepare_extended_lang.sh <dict-src-dir> <oov-dict-entry> <extra-lexicon> <phone-symbol-table> <extended-dict-dir> <tmp-dir> <extended-lang-dir>"
  echo "e.g.: utils/prepare_extended_lang.sh data/local/dict <SPOKEN_NOISE> lexicon_extra.txt data/lang/phones.txt data/local/dict_ext data/local/lang_ext data/lang_ext"
  echo "The goal is to extend the lexicon from <dict-src-dir> with extra lexical entries from <extra-lexicon>, putting the extended lexicon into <extended-dict-dir>,"
  echo "and then build a valid lang dir <extended-lang-dir>."
  echo "<dict-src-dir> must be a valid dictionary dir and <oov-dict-entry> is the oov word (see utils/prepare_lang.sh for details)."
  echo "If a language model is provided, we'll also extend the language model and re-build G.fst (or G.carpa), by inserting extra words as extra unigrams,"
  echo "with probalities as the same as the unigram prob. (optionally boosted) of <oov-dict-entry> from the original LM."
  echo "A phone symbol table from a previsouly built lang dir is required, for validating provided lexical entries."
  echo "options: "
  echo "     --oov-unigram-prob (float)                      # a fixed float value we'll assign to words out of the old vocab, as unigram probs in the language model."
  echo "     --srilm-opts STRING                             # options to pass to SRILM tools (default: '$srilm_opts')"
  echo "     --prep-lang-opts STRING                         # options to pass to utils/prepare_lang.sh"
  echo "     --lm                                            # an ARPA-format language model which we want to insert extra words into, and compile as G.fst. "
  echo "     --to-carpa (true|false)                         # convert the ARPA language model to const arpa format rather than FST. (default: false)."
  exit 1;
fi

srcdict=$1
oov_word=$2
extra_lexicon=$3
phone_symbol_table=$4
extdict=$5 # extended dict dir
tmpdir=$6
extlang=$7 # extended lang dir

mkdir -p $extlang $tmpdir 

[ -f path.sh ] && . ./path.sh

loc=`which change-lm-vocab`
if [ -z $loc ] && [ ! -z $lm ] ; then
  echo You want to rebuild the language model but appear to not have SRILM tools installed.
  echo cd to $KALDI_ROOT/tools and run extras/install_srilm.sh.
  exit 1;
fi

! utils/validate_dict_dir.pl $srcdict && \
  echo "*Error validating directory $srcdict*" && exit 1;

if [[ ! -f $srcdict/lexicon.txt ]]; then
  echo "**Creating $dir/lexicon.txt from $dir/lexiconp.txt"
  perl -ape 's/(\S+\s+)\S+\s+(.+)/$1$2/;' < $srcdict/lexiconp.txt > $srcdict/lexicon.txt || exit 1;
fi

if [[ ! -f $srcdict/lexiconp.txt ]]; then
  echo "**Creating $srcdict/lexiconp.txt from $srcdict/lexicon.txt"
  perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < $srcdict/lexicon.txt > $srcdict/lexiconp.txt || exit 1;
fi

# Checks if the phone sets match.
echo "$(basename $0): Validating the source lexicon"
cat $srcdict/lexicon.txt | awk -v f=$phone_symbol_table '
BEGIN { while ((getline < f) > 0) { sub(/_[BEIS]$/, "", $1); phones[$1] = 1; }}
{ for (x = 2; x <= NF; ++x) { if (!($x in phones)) {
    print "The source lexicon contains a phone not in the phones.txt: "$x;
    print "You must provide a phones.txt from the lang built with the source lexicon.";  exit 1; }}}' || exit 1;

echo "$(basename $0): Validating the extra lexicon"
cat $extra_lexicon | awk -v f=$phone_symbol_table '
BEGIN { while ((getline < f) > 0) { sub(/_[BEIS]$/, "", $1); phones[$1] = 1; }}
{ for (x = 2; x <= NF; ++x) { if (!($x in phones)) {
    print "The extra lexicon contains a phone not in the phone symbol table: "$x; exit 1; }}}' || exit 1;

if [ $stage -le 0 ]; then
  # Genearte the extended dict dir
  echo "$(basename $0): Creating the extended lexicon $extdict/lexicon.txt"
  [ -d $extdict ] && rm -r $extdict 2>/dev/null
  cp -R $srcdict $extdict 2>/dev/null
  
  # Reformat the source lexicon
  perl -ape 's/(\S+\s+)\S+\s+(.+)/$1$2/;' <$srcdict/lexiconp.txt | awk '{ gsub(/\t/, " "); print }'  > $tmpdir/lexicon.txt || exit 1;
  
  # Filter lexical entries which are already in the source lexicon
  awk '{ gsub(/\t/, " "); print }' $extra_lexicon | sort -u | \
    awk 'NR==FNR{a[$0]=1;next} {if (!($0 in a)) print $0 }' $tmpdir/lexicon.txt - \
    > $extdict/lexicon_extra.txt || exit 1;
  
  echo "$(basename $0): Creating $extdict/lexiconp.txt from $srcdict/lexiconp.txt and $extdict/lexicon_extra.txt"
  perl -ape 's/(\S+\s+)(.+)/${1}1\t$2/;' < $extdict/lexicon_extra.txt | \
    cat $srcdict/lexiconp.txt - | sort -u > $extdict/lexiconp.txt || exit 1;
  
  perl -ape 's/(\S+\s+)\S+\s+(.+)/$1$2/;' <$extdict/lexiconp.txt  >$extdict/lexicon.txt || exit 1;
  
  # Create lexicon_silprobs.txt
  silprob=false
  [ -f $srcdict/lexiconp_silprob.txt ] && silprob=true
  if "$silprob"; then
    echo "$(basename $0): Creating $extdict/lexiconp_silprob.txt from $srcdict/lexicon_silprob.txt"
    # Here we assume no acoustic evidence for the extra word-pron pairs.
    # So we assign silprob1 = overall_silprob, silprob2 = silprob3 = 1.00
    overall_silprob=`awk '{if ($1=="overall") print $2}' $srcdict/silprob.txt`
    awk -v overall=$overall_silprob '{
      printf("%s %d %.1f %.2f %.2f",$1, 1, overall, 1.00, 1.00); 
      for(n=2;n<=NF;n++) printf " "$n; printf("\n");
      }' $extdict/lexicon_extra.txt | cat $srcdict/lexiconp_silprob.txt - | \
      sort -k1,1 -k2g,2 -k6 \
      > $extdict/lexiconp_silprob.txt || exit 1;
  fi
  
  if ! utils/validate_dict_dir.pl $extdict >&/dev/null; then
    utils/validate_dict_dir.pl $extdict  # show the output.
    echo "$(basename $0): Validation failed on the extended dict"
    exit 1;
  fi
fi

if [ $stage -le 1 ]; then
  echo "$(basename $0): Preparing the extended lang dir."
  [ -d $extlang ] && rm -r $extlang 2>/dev/null
  utils/prepare_lang.sh $prep_lang_opts $extdict \
    $oov_word $tmpdir $extlang || exit 1;
fi

if [ $stage -le 2 ]; then
  [ -z "$lm" ] && echo "$(basename $0): Not extending the language model since it's not provided." && exit 0;
  awk '{print $1}' $extlang/words.txt > $tmpdir/voc || exit 1;
  
  echo "$(basename $0): Inserting extra unigrams into '$lm'."
  gunzip -c $lm | \
    awk -v s=$extdict/lexicon_extra.txt -v oov=$oov_word -v boost=$oov_unigram_prob -v prob=$oov_prob \
    'BEGIN{ while((getline < s) > 0) { dict[$1] = 1; a += 1; } unigram = 1; } 
    { if ($1 == "ngram") { split($2, y, "="); if (y[1] == 1) $2 = y[1]"="y[2]+a; } print $0;
      if (unigram == 1 && $2 == oov) {
        prob = log(oov_unigram_prob); 
        for (x in dict) { $1 = prob; $2 = x; print $0; }
      } 
      if ($1 == "\\2-grams:") unigram = 0;
    }' | change-lm-vocab -vocab $tmpdir/voc -lm - -write-lm - $srilm_opts | gzip -c > $tmpdir/lm_tmp.gz

  if $to_carpa; then
    echo "$(basename $0): Convert the LM to FST."
    unk=`cat $extlang/oov.int`
    bos=`grep -w "<s>" $extlang/words.txt | awk '{print $2}'`
    eos=`grep "</s>" $extlang/words.txt | awk '{print $2}'`
    if [[ -z $bos || -z $eos ]]; then
      echo "$0: <s> and </s> symbols are not in $extlang/words.txt"
      exit 1
    fi
    arpa-to-const-arpa --bos-symbol=$bos \
      --eos-symbol=$eos --unk-symbol=$unk \
      "gunzip -c $tmpdir/lm_tmp.gz | utils/map_arpa_lm.pl $extlang/words.txt|"  $extlang/G.carpa  || exit 1;
   else
     echo "$(basename $0): Convert the LM to const arpa format."
     gunzip -c $tmpdir/lm_tmp.gz | arpa2fst --disambig-symbol=#0 \
       --read-symbol-table=$extlang/words.txt - $extlang/G.fst || exit 1
   fi
fi

exit 0;
