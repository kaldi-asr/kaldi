#!/bin/bash 

# Copyright 2012  Johns Hopkins University (Author: Yenda Trmal)
# Apache 2.0.

#Begin configuration
ignore_missing_txt=false  #If the reference transcript txt is missing, \
                          #shall we ignore it or treat it as a fatal error?
#End configuration
echo "$0 $@"  # Print the command line for logging

help_message="$0: create subset of the input directory (specified as the first directory).
                 The subset is specified by the second parameter.
                 The directory in which the subset should be created is the third parameter
             Example:
                 $0 <source-corpus-dir> <subset-descriptor-list-file> <target-corpus-subset-dir>"

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [[ "$#" -ne "3" ]] ; then
    echo -e "FATAL: wrong number of script parameters!\n\n"
    printf "$help_message\n\n"
    exit 1;
fi

input_data_dir=$1
input_data_list=$2
output_data_dir=$3

if [[ ! -d "$input_data_dir" ]] ; then
  echo "FATAL: input data directory does not exist"; 
  exit 1;
fi
if [[ ! -f "$input_data_list" ]] ; then 
  echo "FATAL: input data list file does not exist!"; 
  exit 1;
fi

mkdir -p $output_data_dir/transcription
mkdir -p $output_data_dir/audio

abs_src_dir=`readlink -f $input_data_dir` 
abs_tgt_dir=`readlink -f $output_data_dir`

echo "Making subset..."
for file_basename in `cat $input_data_list`; do
    if [[ -e $abs_src_dir/audio/$file_basename.sph ]] ; then
        ln -sf $abs_src_dir/audio/$file_basename.sph $abs_tgt_dir/audio || exit 1
    else 
      if [[ -e $abs_src_dir/audio/$file_basename.wav ]] ; then
        ln -sf $abs_src_dir/audio/$file_basename.wav $abs_tgt_dir/audio || exit 1
      else
        echo "File $abs_src_dir/audio/$file_basename.sph|wav does not exist!"
        exit 1
      fi
    fi

    if [[ -e $abs_src_dir/transcription/$file_basename.txt ]] ; then
        ln -sf $abs_src_dir/transcription/$file_basename.txt $abs_tgt_dir/transcription || exit 1
    else
        echo "File $abs_src_dir/transcription/$file_basename.txt does not exist!"

        if ! $ignore_missing_txt ; then
          exit 1;
        fi
    fi
done



