#!/bin/bash

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

input_data_dir=$1
input_filelist=$2
output_data_dir=$3

mkdir -p $output_data_dir/transcription
mkdir -p $output_data_dir/audio

abs_src_dir=`readlink -f $input_data_dir` 
abs_tgt_dir=`readlink -f $output_data_dir`

for file_basename in `cat $input_data_list`; do
    if [[ -e $abs_src_dir/audio/$file_basename.sph ]] ; then
        ln -sf $abs_src_dir/audio/$file_basename.sph $abs_tgt_dir/audio || exit 1
    else
        echo "File $abs_src_dir/audio/$file_basename.sph does not exist!"
        exit 1
    fi

    if [[ -e $abs_src_dir/transcription/$file_basename.txt ]] ; then
        ln -sf $abs_src_dir/transcription/$file_basename.txt $abs_tgt_dir/transcription || exit 1
    else
        echo "File $abs_src_dir/audio/$file_basename.sph does not exist!"
        exit 1
    fi
done



