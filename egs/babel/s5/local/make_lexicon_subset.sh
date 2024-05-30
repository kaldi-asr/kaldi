#!/usr/bin/env bash

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

transcriptions=$1
input_lexicon_file=$2
output_lexicon_file=$3

(
  #find $dev_data_dir/transcription/ -name "*.txt" | xargs egrep -vx '\[[0-9.]+\]'  |cut -f 2- -d ':' | perl -ape 's/ /\n/g;' 
  find $transcriptions -name "*.txt" | xargs egrep -vx '\[[0-9.]+\]'  |cut -f 2- -d ':' | perl -ape 's/ /\n/g;'
) | sort -u | awk ' 
  BEGIN {
      while(( getline line< ARGV[2] ) > 0 ) {
          split(line, e, "\t")
          LEXICON[ e[1] ]=line
      }
      FILENAME="-"
      i=0
    
      while(( getline word< ARGV[1] ) > 0 ) {
        if (word in LEXICON)
          print LEXICON[word]
      }
  }
' -  $input_lexicon_file | sort -u > $output_lexicon_file

