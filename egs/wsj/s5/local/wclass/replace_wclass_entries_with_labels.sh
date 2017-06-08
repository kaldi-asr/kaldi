#!/bin/bash

# Copyright FAU Erlangen-Nuremberg (Author: Axel Horndasch) 2016
#
# Given a training text for a language model, a word class mapping
# in the 'classes-format' (see also
# http://www.speech.sri.com/projects/srilm/manpages/classes-format.5.html)
# and a corpus name, this script replaces all occurences of word class
# entries with word class labels.
#
# For example if there is a wsj.classes (excerpt):
# ...
# C=US_STATE ALABAMA
# ...
# after running 'replace_wclass_entries_with_labels.sh' the training text
# IN HUNTSVILLE ALABAMA
# becomes
# IN HUNTSVILLE C=US_STATE
#
# To keep track of all replacements, a file <corpus-name>_count.classes
# is created. This can be reused to estimate probabilities for the
# word-class sub-language model (see also create_wclass_SLMs.sh). A line
# in the <corpus-name>_count.classes file looks somehow like this:
# ...
# C=US_STATE 885 ALABAMA
# ...
#
# The original text file is replaced with the new text file that contains
# word class labels.

set -e

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "usage: replace_wclass_entries_with_labels.sh <text> <wlcass-dir> <corpus-name>" && exit 1;
fi

text=$1
wclass_dir=$2
corpus_name=$3

# some sanity checks regarding files we expect to exist
if [ ! -d $wclass_dir ]; then
  echo "The directory \"$wclass_dir\" does not exist, exiting ..." && exit 1;
fi

if [ ! -f $text ]; then
  echo "The training text for the language model \"$text\" does not exist, exiting ..." && exit 1;
fi

if [ -z "$corpus_name" ] ||
   [ ! -f $wclass_dir/$corpus_name.classes ]; then
  echo "Invalid corpus name \"$corpus_name\" or missing file \"$wclass_dir/$corpus_name.classes\", exiting ..." && exit 1;
fi

# This is the call to replace word class entries; 'normalize=0' is used
# to preserve replacement counts (otherwise they are normalized to
# probabilities between 0 and 1).
cat $text | replace-words-with-classes normalize=0 outfile=$wclass_dir/${corpus_name}_count.classes classes=$wclass_dir/$corpus_name.classes > ${text}_classes
mv ${text}_classes $text

exit 0;
