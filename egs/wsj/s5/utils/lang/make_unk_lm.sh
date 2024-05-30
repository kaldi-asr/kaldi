#!/usr/bin/env bash

# Copyright      2016 Johns Hopkins University (Author: Daniel Povey);

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


# Begin configuration section.
cmd=run.pl
ngram_order=4
num_extra_ngrams=10000
position_dependent_phones=true
use_pocolm=true
min_word_length=2
stage=0
phone_disambig_symbol="#1"

# end configuration sections

[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] <input-dict-dir> <work-dir>"
  echo "e.g.: $0 data/local/dict exp/make_unk"
  echo ""
  echo "This script creates, as an FST, a phone language model suitable for modeling"
  echo "the unknown word.  It first trains a language model on the phone sequences of the"
  echo "provided dictionary entries (which should be without any word-position-dependency"
  echo "tags); it then creates an FST from it, while, for compactness after context-dependency"
  echo "limiting the transitions to seen bigram pairs of phones.  Then, by composing with"
  echo "a separate FST it converts it into word-position-dependent phones if applicable,"
  echo "while imposing a minimum-number-of-phones constraint."
  echo ""
  echo "  <input-dict-dir>:  A dictionary directory (as validated by validate_dict_dir.pl);"
  echo "             the dictionary from this location (lexicon.txt, lexiconp.txt, or"
  echo "             lexiconp_silprob.txt) will be used to train the language model on"
  echo "             phones.  The files silence_phones.txt and nonsilence_phones.txt will"
  echo "             be used to construct a symbol table used internally, and to"
  echo "             exclude lexicon entries containing silences."
  echo " <work-dir>:    A place to put logs and the output of this script.  The output of"
  echo "                this script will be written to <work-dir>/unk_fst.txt (we write in"
  echo "                text form so that it's independent of the phones.txt)."
  echo "Options:"
  echo "    --ngram-order <n>                 # (default: 4)  N-gram order of the phone-level language"
  echo "                                      # model.  Must be in range [2, 7]"
  echo "    --num-extra-ngrams <n>            # (default: 10000).  The maximum the number of n-grams"
  echo "                                      # that may be present in the language model in addition"
  echo "                                      # to the unigrams.  The LM will be pruned to achieve this."
  echo "    --use-pocolm <true|false>         # (default: true).  If true, use pocolm to estimate the"
  echo "                                      # language model; you will be prompted to install it if"
  echo "                                      # needed.  (If false, we use the script make_phone_lm.py,"
  echo "                                      # which is simpler but the perplexity is not as good)."
  echo "    --position-dependent-phones <true|false>  # (default: true).  If true, assume position-dependent"
  echo "                                      # phones (although in any case the lexicon should use position-"
  echo "                                      # independent phones).  If position-dependent phones are used,"
  echo "                                      # after creating the LM we compose with an FST that converts"
  echo "                                      # into position-dependent phones while enforcing the natural"
  echo "                                      # constraints that they form a single word."
  echo "    --min-word-length <1|2>           # (default: 2).  May only be 1 or 2.  The minimum word length"
  echo "                                      # (in number of phones) that is allowed"
  echo "    --phone-disambig-symbol <symbol>  # default: '#1'.  This is the symbol that will be put on the"
  echo "                                      # input side of backoff arcs.  You won't normally have to change"
  echo "                                      # this because prepare_lang.sh expects '#1' there."
  exit 1;
fi


dict_dir=$1
dir=$2

set -e

mkdir -p $dir/log

if [ $stage -le 0 ]; then
  if ! utils/validate_dict_dir.pl $dict_dir >&$dir/log/validate_dict_dir.log; then
    cat $dir/log/validate_dict_dir.log
    echo "$0: failed to validate input dict-dir $dict_dir"
    exit 1
  fi
fi

if ! [ $ngram_order -ge 2 ] || ! [ $ngram_order -le 7 ]; then
  echo "$0: invalid --ngram-order $ngram_order (must be in [2,7])"
  exit 1
fi

if ! [ $min_word_length -ge 1 ] || ! [ $min_word_length -le 2 ]; then
  echo "$0: invalid --min-word-length $min_word_length (must be in [1,2])"
  exit 1
fi

# The next command creates a symbol table that will cover all the symbols we might
# possibly need in this script.  The word-position-dependent suffixes (_B and so on
# won't be needed if --position-dependent-phones is false, but it won't hurt.
cat $dict_dir/silence_phones.txt $dict_dir/nonsilence_phones.txt | \
  awk '{for(n=1;n<=NF;n++) print $n; }' | \
  awk '{print $1; print $1 "_B"; print $1 "_I"; print $1 "_S"; print $1 "_E";}' | \
      cat - <(echo "$phone_disambig_symbol") | \
  awk 'BEGIN{print "<eps> 0";} {print $1, NR;}' > $dir/phones.txt

phone_disambig_int=$(tail -n 1 <$dir/phones.txt | awk '{print $2}')
if ! [ $phone_disambig_int == $phone_disambig_int ]; then
  echo "$0: problem working out integer form of phone-disambig symbol."
  exit 1;
fi

if [ -e $dict_dir/lexicon.txt ]; then
  src_dict=$dict_dir/lexicon.txt
  first_phone_field=2
elif [ -e $dict_dir/lexiconp.txt ]; then
  src_dict=$dict_dir/lexiconp.txt
  first_phone_field=3
else
  [ ! -e $dict_dir/lexiconp_silprob.txt ] && \
    echo "$0: expected file $dict_dir/lexiconp_silprob.txt to exist" && exit 1
  src_dict=$dict_dir/lexiconp_silprob.tt
  first_phone_field=6
fi

cat $dict_dir/silence_phones.txt | awk '{for(n=1;n<=NF;n++) print $n; }' > $dir/silence_phones.txt

# prepare the cleaned up version of the dictionary (to train our phone LM), with
# the first field (the word) removed, with prons that have silence phones in
# them removed, and with empty prons (which should not be allowed anyway, but
# just in case..) removed.
awk -v dir=$dir -v ff=$first_phone_field \
   'BEGIN{ while ((getline <(dir"/silence_phones.txt")) > 0) sil[$1]=1;  }
         { ok=1; for (n=ff; n<=NF; n++) { if ($n in sil) ok=0; }
           if (ok && NF>=ff) { for (n=ff;n<=NF;n++) printf("%s ",$n); print ""; } else {
            print("make_unk_lm.sh: info: not including dict line: ", $0) >"/dev/stderr" }}' <$src_dict >$dir/training.txt
cat $dir/training.txt | awk '{for(n=1;n<=NF;n++) seen[$n]=1; } END{for (k in seen) print k;}' > $dir/all_nonsil_phones

num_dict_lines=$(wc -l <$src_dict)
num_train_lines=$(wc -l < $dir/training.txt)
if ! [ $num_train_lines -gt 0 ]; then
  echo "$0: something went wrong getting text to train phone-level LM."
  exit 1
fi
echo "$0: training on $num_train_lines words out of $num_dict_lines in the "
echo "     ... original dictionary (excluding words with silence phones)."


if [ $num_train_lines -lt 2000 ] && $use_pocolm; then
  echo "$0: the number of lines of training data is very small [$num_train_lines]."
  echo "    Setting --use-pocolm to false since it probably won't work well"
  echo "    on so little data (e.g. hard to estimate the discounting parameters)"
  echo "    Using make_phone_lm.py instead."
  use_pocolm=false
fi

if $use_pocolm; then
  if [ ! -e $KALDI_ROOT/tools/pocolm ]; then
    echo "$0: $KALDI_ROOT/tools/pocolm does not exist:"
    echo " ... please do:  cd $KALDI_ROOT/tools; extras/install_pocolm.sh"
    echo " ... and then rerun this script."
    exit 1
  fi

  PATH=$KALDI_ROOT/tools/pocolm/scripts:$PATH

  if [ $stage -le 1 ]; then
    echo "$0: training $ngram_order-gram LM with pocolm"

    mkdir -p $dir/pocolm/text
    heldout_ratio=5  # hold out one fifth of the data as validation to estimate
    # metaparameters; we'll fold it back in before estimating the
    # final LM.
    cat $dir/training.txt | awk -v h=$heldout_ratio '{if(NR%h == 0) print; }' > $dir/pocolm/text/dev.txt
    cat $dir/training.txt | awk -v h=$heldout_ratio '{if(NR%h != 0) print; }' > $dir/pocolm/text/train.txt


    # the following options are because we expect the amount of data to be small,
    # all the data subsampling isn't really needed and will increase the chance of
    # something going wrong.

    small_data_opts="--num-splits 4 --warm-start-ratio 1"
    $cmd $dir/log/train_lm.log \
         train_lm.py --wordlist $dir/all_nonsil_phones $small_data_opts \
         --fold-dev-into=train $dir/pocolm/text $ngram_order $dir/pocolm
  fi

  if [ $stage -le 2 ]; then
    echo "$0: pruning LM with pocolm"
    num_words=$(wc -l <$dir/all_nonsil_phones)
    num_ngrams=$[$num_extra_ngrams+$num_words]


    $cmd $dir/log/prune_lm_dir.log \
         prune_lm_dir.py --target-num-ngrams=$num_ngrams \
         $dir/pocolm/all_nonsil_phones_${ngram_order}.pocolm $dir/poclm/lm_pruned

    # format as arpa.
    format_arpa_lm.py $dir/poclm/lm_pruned > $dir/pocolm.arpa
  fi

  if [ $stage -le 3 ]; then
    echo "$0: applying bigram constraints and converting from ARPA to FST"
    # now get bigram constraints: we want to get an FST that only allows phone
    # bigrams that we've seen (this may enforce certain linguistic constraints,
    # and also stops the graph from blowing up too much once we introduce
    # phonetic context.
    # The NF > 0 is just a double-check that there are no empty prons, which
    # would be bad as it would allow an empty pronunciation of the unknown word.
    cat $dir/training.txt | awk '{ if (NF > 0) printf("<s> %s </s>\n", $0); }' | \
      awk '{for(n=1;n<NF;n++) { m=n+1; seen[ $n " " $m ] = 1; }} END{for(k in seen) print k;}' \
          > $dir/allowed_bigrams

    $cmd $dir/log/arpa2fst.log \
         utils/lang/internal/arpa2fst_constrained.py --verbose=3 \
           --disambig-symbol="$phone_disambig_symbol" \
         $dir/pocolm.arpa $dir/allowed_bigrams '>' $dir/unk_fst_orig.txt
  fi
else

  if [ $stage -le 1 ]; then
    echo "$0: using make_phone_lm.py to create $ngram_order-gram language-model FST"
    $cmd $dir/log/make_phone_lm.log \
         utils/sym2int.pl $dir/phones.txt $dir/training.txt '|' \
         utils/lang/make_phone_lm.py --verbose=2 \
         --phone-disambig-symbol=$phone_disambig_int \
         --num-extra-ngrams=$num_extra_ngrams \
         --ngram-order=$ngram_order '|' \
         utils/int2sym.pl -f 3-4 $dir/phones.txt '>'$dir/unk_fst_orig.txt
  fi
fi


sym_opts="--isymbols=$dir/phones.txt --osymbols=$dir/phones.txt"

if ! $position_dependent_phones; then
  if  [ $min_word_length == 1 ]; then
    echo "$0: no word-length constraint or word-position-dependency, so exiting."
    # There is no need to compose unk_fst_orig.txt with a separate FST: because of
    # the bigram constraints and because we ensure that there were no empty prons
    # in the dictionary (no empty lines in training.txt), the FST wouldn't allow
    # length-zero words anyway.
    cp $dir/unk_fst_orig.txt $dir/unk_fst.txt
    fstcompile $sym_opts <$dir/unk_fst.txt >$dir/unk.fst
    exit 0;
  else
    echo "$0: creating constraint_fst.txt for min-word-length=2 constraint."
    # min-word-length is 2; we need to apply that constraint.  A note on the FST
    # states: 0 is start state, 1 is "seen one phone", 2 is "seen two or more
    # phones".
    # We don't need to take into account the disambig symbol because we compose on
    # the right with this FST, and it doesn't appear on the output side.
    cat $dir/all_nonsil_phones | \
      awk '{ph[$1]=1} END{ for (p in ph) { print 0,1,p,p; print 1,2,p,p; print 2,2,p,p; }
                 print 2,0.0; }' > $dir/constraint_fst.txt
  fi
else
  echo "$0: creating constraint_fst.txt for min-word-length=$min_word_length constraint, plus word-position-dependency conversion."

  # Add constraints and convert phones without tags into phones with the _B, _E, _I and _S
  # tags (begin, end, internal, singleton).

  # States:
  # 0 is start state,
  # 1 is "seen initial phone (and maybe internal phones) of multi-phone word",
  # 2 is "seen final phone of multi-phone word".
  # 3 is "seen phone of single-phone word"; note, if --min-word-length is 2,
  #      then state 3 will not exist.

  cat $dir/all_nonsil_phones | \
    awk -v mwl=$min_word_length -v "disambig=$phone_disambig_symbol" \
 '{ph[$1]=1} END{ for (n=0; n<3; n++) print n,n,disambig,disambig;
                  for (p in ph) { printf("0 1 %s %s_B\n", p, p); printf("1 1 %s %s_I\n", p, p);
                                  printf("1 2 %s %s_E\n", p, p); if (mwl==1) printf("0 3 %s %s_S\n", p, p);  }
                 print 2,0.0; if (mwl==1) print 3,0.0; }' >$dir/constraint_fst.txt
fi


echo "$0: creating final FST via composition, etc."

fstcompile $sym_opts <$dir/constraint_fst.txt | fstarcsort > $dir/constraint.fst
fstcompile $sym_opts <$dir/unk_fst_orig.txt >$dir/unk_orig.fst

# The first 'fstproject' below projects on the input; it makes sure the
# disambiguation symbol appears on the output side also.
# The fstcompose actually applies the constraints and does the conversion, but
# after this the "correct" phones appear only on the output side.
# The second 'fstproject' copies the word-position-dependent phones to
# the input side.
# The 'fstpushspecial' pushes the weights, as the composition with the
#  constraint FST makes the FST quite non-stochastic [weights per state do not
#  sum up to one].
# The 'fstrmsymbols' command makes sure the disambiguation symbol appears only
# on the input side.
# 'fstminimizeencoded' combines states that are the same as far as their output
# arcs are concerned; in the case where --min-word-length is 1, this combines
# a lot of final-states that have no transitions out of them.
fstproject $dir/unk_orig.fst | \
  fstcompose - $dir/constraint.fst | \
  fstproject --project_output=true | \
  fstpushspecial | \
  fstminimizeencoded | \
  fstrmsymbols --remove-from-output=true <(echo $phone_disambig_int) >$dir/unk.fst

fstprint $sym_opts <$dir/unk.fst >$dir/unk_fst.txt


exit 0;
