#!/bin/bash

# Copyright FAU Erlangen-Nuremberg (Author: Axel Horndasch) 2016
#
# This script creates sub-languages models for all word classes which are
# listed in the file $wclass_list (e.g. data/local/wclass/wclass_list.txt).
#
# Apart from the word class list file the script also expects to find files
# which include
# word class label, a (corpus-dependent) count, word class entry, e.g.
# C=WEEKDAY 6588 MONDAY
# C=WEEKDAY 5662 TUESDAY
# C=WEEKDAY 4794 WEDNESDAY
# ...
# These files can for example be created using the SRI LM tool
# replace-words-with-classes (see also replace_wclass_entries_with_labels.sh).
#
# If the file $wclass_list indicates that SWUs should be used for OOV
# detection (find more details in the comments below), this script embeds
# an OOV sub-language model into the word-class sub-language model.
#
# The approach is also described in the paper "How to Add Word Classes
# to the Kaldi Speech Recognition Toolkit" (TSD 2016)

set -e

echo "$0 $@"  # Print the command line for logging
. ./path.sh

# begin configuration section
swu_ngram_size=1
# end configuration section

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: "
  echo "  $0 [options] <wclass-dir> <words-txt> <corpus-name>"
  echo "e.g.:"
  echo " $0 --swu-ngram-size=5 data/local/wclass data/lang_nosp_bd/words.txt wsj"
  echo "Options"
  echo "   --swu-ngram-size=<int>     # the n-gram size used for SWU-based OOV detection, default: 3"
  exit 1;
fi

wclass_dir=$1
words_txt=$2
corpus_name=$3

# Files which hopefully exist
wclass_list=$wclass_dir/wclass_list.txt
wclass_count_file=$wclass_dir/${corpus_name}_count.classes

# A local wclass directory for the sub-language models (to be created)
wclass_lm_dir=$wclass_dir/lm

# Some sanity checks regarding files we expect to exist
if [ ! -d $wclass_dir ]; then
  echo "The directory \"$wclass_dir\" does not exist, exiting ..." && exit 1;
fi

if [ ! -f $words_txt ]; then
  echo "The file containing the word -> symbol mapping \"$words_txt\" does not exist, exiting ..." && exit 1;
fi

if [ -z "$corpus_name" ]; then
  echo "Invalid corpus name \"$corpus_name\", exiting ..." && exit 1;
fi

if [ ! -f $wclass_list ]; then
  echo "The file containing the word class list \"$wclass_list\" does not exist, exiting ..." && exit 1;
fi

if [ ! -f $wclass_dir/$corpus_name.classes ]; then
  echo "The file containing the word -> class mapping \"$wclass_dir/$corpus_name.classes\" does not exist, exiting ..." && exit 1;
fi

if [ ! -f $wclass_dir/${corpus_name}_count.classes ]; then
  echo "The file containing the word class counts \"$wclass_dir/${corpus_name}_count.classes\" does not exist."
  echo "The file should have been created by replace_wclass_entries_with_labels.sh, exiting ..." && exit 1;
fi

# This is the new directory for the class-based sub-language models.
mkdir -p ${wclass_lm_dir}

# We have to create symbol mappings (word class label -> integer) so we can
# later replace them in embed_wclass_SLMs.sh using fstreplace. As word
# class labels we use C=CLASS_NAME to avoid clashes with actual words.

all_words_txt=$wclass_dir/all_words.txt

# Since words.txt already exists, we need to extract the current word count.
word_count=`tail -n 1 $words_txt | awk '{ print $2 }'`
 
# Creating a new file which contains word class label -> integer mappings
cat $wclass_list | \
    awk -v WC=$word_count '{
      printf("%s %d\n", $1, ++WC);
      if($2 > 0.0 && $2 < 1.0) {
	printf("%s_SWU %d\n", $1, ++WC);
      }
    }' > $wclass_lm_dir/wclass_words.txt || exit 1;

# For the FSTs we build later on, we need words.txt and the new mappings.
cat $words_txt $wclass_lm_dir/wclass_words.txt > $all_words_txt

echo "Creating word-class-based sub-language models ..."

# Create sub-language models for all available word classes; the class names
# are extracted from the file 'wclass_list.txt' which is assumed to have the
# following format:
#
# C=CLASS_NAME_1<TAB>0.2
# C=CLASS_NAME_2<TAB>0.0
# ...
#
# The second value is a a probability-like value for OOV detection; the higher
# the value, the more probable it is to go to the SWU-based OOV model for the
# word class.
for wclass_label in $( cat $wclass_list | awk '{ print $1 }' ); do
  wclass=`echo $wclass_label | sed 's/^C=//'`
  echo "Creating a sub-language model for word class $wclass ..."

  # determine_wclass_specific_counts.pl extracts the number of occurences for
  # the words in a certain word class (they have been counted already by
  # 'replace-words-with-classes', but the counts for all word classes are in
  # one file -> $wclass_count_file).

  # In case there is a probability p (0.0 < p <= 1.0) for ${wclass} in
  # wclass_list.txt determine_wclass_specific_counts.pl will also add an extra
  # entry "${wclass_label}_SWU" to the class-specific counts file (and thus the
  # word-class sub-language model); that extra entry is then replaced with an
  # SWU-based sub-language model for OOV detection below.
  local/wclass/determine_wclass_specific_counts.pl --replace-blanks-with-dash "true" $wclass $wclass_label \
    $wclass_dir/wclass_list.txt ${wclass_count_file} $wclass_lm_dir/counts.$wclass

  # now we can actually create the (in this case unigram) language models
  # for the word class (with or without OOV model)
  ngram-count -order 1 -text-has-weights -text $wclass_lm_dir/counts.$wclass -lm $wclass_lm_dir/lm.$wclass && gzip -f $wclass_lm_dir/lm.$wclass

  # The following sequence of calls converts the language model to an FST.
  # Two things need to be prevented:
  # - creating no output (<eps>) during decoding when entering and leaving
  #   the word class graph -> this job is done by convert_self-loop_to_two-state_fst.pl
  # - confusing different word classes -> this job is done by
  #   replace_BOS_and_EOS_with_disambig_symbol.pl (it adds disambiguation
  #   symbols when entering and leaving the word class graph)
  gunzip -c $wclass_lm_dir/lm.${wclass}.gz | \
  arpa2fst - | fstminimize | fstrmepsilon | fstprint | \
  local/wclass/convert_to_embeddable_fst.pl | \
  local/wclass/replace_BOS_and_EOS_with_disambig_symbol.pl \
             --bos-input-symbol "#$wclass" --bos-output-symbol "<eps>" \
	     --eos-input-symbol "#$wclass" --eos-output-symbol "<eps>" \
	     --remove-bos-weight "true" --remove-eos-weight "true" | \
  fstcompile -isymbols=$all_words_txt -osymbols=$all_words_txt --keep_osymbols=false > $wclass_lm_dir/$wclass.fst

  # In case there is OOV modeling based on sub-word units (SWUs) for this class
  # the call to determine_wclass_specific_counts.pl has added the entry
  # "${wclass_label}_SWU" to the class-specific counts file. If a grep for this
  # symbol is successful, we need to check for the SWU file, create an SWU FST
  # for the word class and embed it into the word-class sub-language model.
  if grep -q "${wclass_label}_SWU" $wclass_lm_dir/counts.$wclass; then
    echo OOV modeling for word class $wclass
    if [ ! -f $wclass_dir/$wclass.swu ]; then
      echo $wclass_dir/wclass_list.txt indicated OOV detection for class $wclass,
      echo but the SWU file $wclass_dir/$wclass.swu could not be found, exiting...
      exit 1;
    fi

    # Count occurences of the SWUs for the OOV model of the category
    ngram-count -text $wclass_dir/$wclass.swu -write $wclass_lm_dir/counts.${wclass}_SWU
    ngram-count -order $swu_ngram_size -read $wclass_lm_dir/counts.${wclass}_SWU -lm $wclass_lm_dir/lm.${wclass}_SWU && gzip -f $wclass_lm_dir/lm.${wclass}_SWU

    # Create an FST from class-specific SWU sub-language model for OOV detection
    gunzip -c $wclass_lm_dir/lm.${wclass}_SWU.gz | \
    arpa2fst - | fstminimize | fstrmepsilon | fstprint | \
    local/wclass/convert_to_embeddable_fst.pl --back-transition "#${wclass}_SWU_BACK" | \
    local/wclass/replace_BOS_and_EOS_with_disambig_symbol.pl \
               --bos-input-symbol "#${wclass}_SWU" --bos-output-symbol "#${wclass}_SWU" \
	       --eos-input-symbol "#${wclass}_SWU" --eos-output-symbol "#${wclass}_SWU" \
	       --remove-bos-weight "true" --remove-eos-weight "true" | \
    fstcompile -isymbols=$all_words_txt -osymbols=$all_words_txt --keep_osymbols=false > $wclass_lm_dir/${wclass}_SWU.fst

    # The ${wclass_label}_SWU label is an integer -> we need that number for 'fstreplace'
    # so we can embed the OOV model
    swu_model_label=`awk -v unk="${wclass_label}_SWU" '{ if ( $1 == unk ) { print $2 } }' $all_words_txt`
    echo "Extracted ID for word class label ${wclass_label}_SWU: \"${swu_model_label}\""

    # Embedding the sub-word unit-based OOV model in the word-class sub-language model
    if [ -n "$swu_model_label" ]; then
      fstreplace --epsilon_on_replace $wclass_lm_dir/$wclass.fst -1 $wclass_lm_dir/${wclass}_SWU.fst $swu_model_label |\
      fstrmepsilon |\
      fstminimizeencoded |\
      fstarcsort --sort_type=ilabel > $wclass_lm_dir/${wclass}_new.fst
      mv $wclass_lm_dir/${wclass}_new.fst $wclass_lm_dir/$wclass.fst
    else
      echo Could not extract word ID for ${wclass_label}_SWU, exiting... && exit 1;
    fi
  else
    echo No OOV modeling for word class $wclass
  fi
done

echo "Successfully created word-class-based sub-language models ..."

exit 0;
