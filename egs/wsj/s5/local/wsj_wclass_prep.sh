#!/bin/bash
#
# Copyright 2010-2016 FAU Erlangen  (Author: Axel Horndasch).  Apache 2.0.

# To be run from one directory above this script (e.g. from run.sh)
#
# The main purpose of the script is to copy corpus-specific word class
# files from an external source ($wclass_src_dir) to the place in the
# Kaldi directory structure ($wclass_dst_dir -> usually data/local/wclass).
# The directory $wclass_dst_dir is created by this script.
#
# Obligatory files in $wclass_src_dir are
# - wsj.classes -> a file containing the mapping of words to classes
#                  in the 'classes-format', see also
#                  http://www.speech.sri.com/projects/srilm/manpages/classes-format.5.html
# - wclass_list.txt -> a file which contains all classes, including an OOV probability
#
# Optional files in $wclass_src_dir are
# - ${class}.swu -> one file per class which contains sub-word units (SWUs) for OOV detection,
#                   one SWU per line; these SWUs also need to be in the pronunciation lexicon

set -e

# Begin configuration section.
corpus_name=wsj
# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./utils/parse_options.sh

if [ $# != 2 ]; then
  echo "Usage: $0 <word-class-source-dir> <word-class-destination-dir>"
  exit 1
fi

[ -f path.sh ] && . ./path.sh;

wclass_src_dir=$1
wclass_dst_dir=$2

# the following two files are obligatory, if they don't exist -> exit
wclass_file=$wclass_src_dir/$corpus_name.classes
wclass_list=$wclass_src_dir/wclass_list.txt

if [ ! -f $wclass_file ] ||
   [ ! -f $wclass_list ]; then
   echo "Error: $0 requires a directory that contains the files ${corpus_name}.classes and wclass_list.txt"
   echo "The file ${corpus_name}.classes needs to be in the 'classes-format', see also"
   echo "http://www.speech.sri.com/projects/srilm/manpages/classes-format.5.html"
   exit 1
fi

mkdir -p $wclass_dst_dir

# copy the files which contain the word classes, the word class list and (optional) SWU files
cp $wclass_file $wclass_list $wclass_dst_dir

############# SWU files for OOV detection in word classes ####################
# Extracting the list of word class names from the wclass_list.txt file
# Please note: In wclass_list.txt the word class label is used which is
#              usually the prefix 'C=' + the word class name. The prefix
#              is removed here -> C=CITYNAME becomes CITYNAME
#
# The file wclass_list.txt has lines which consist of two components
# 1. The word class label (e.g. C=CITYNAME)
# 2. A floating point number which is a probablity-like value (in [0.0,1.0])
#    for OOV detection. The higher the value, the more probable it is to
#    recognize SWUs (-> and detect OOVs) in a given class.
#
# The values are separated by a TAB, e.g.
# C=CITYNAME	0.1
#
# For all classes which have an OOV probability > 0.0 there should be a file
# containing sub-word units (SWUs) for OOV detection, e.g. CITYNAME.swu

oov_word_classes=`cat $wclass_list |\
sed 's/C=//' |\
awk '{ if($2 > 0.0 && $2 < 1.0) { print $1 } }'`

# Only classes with an OOV probability > 0.0 need an SWU file
for wclass in $oov_word_classes; do
  # The (class-specific) SWU files are used for a sub-language model which
  # is embedded into the word class language model to do OOV detection
  wclass_swu_file=$wclass_src_dir/$wclass.swu
  if [ -f $wclass_swu_file ]; then
    cp $wclass_swu_file $wclass_dst_dir/$wclass.swu
  else
    echo "WARNING: no SWU file for OOV detection (word class $wclass)"
  fi
done

local/wclass/create_extra_wclass_disambig_syms.pl --remove-wclass-prefix "C=" $wclass_list > $wclass_dst_dir/wclass_disambig_syms.txt

echo $0 succeeded.
