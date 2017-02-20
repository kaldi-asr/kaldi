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

if [[ "$#" -lt "3" ]] ; then
    echo -e "FATAL: wrong number of script parameters!\n\n"
    printf "$help_message\n\n"
    exit 1;
fi

output_data_dir=${@: -1}  # last argument to the script
sources=( $@ )
unset sources[${#sources[@]}-1]  # 'pop' the last argument which is odir
num_src=${#sources[@]}  # number of systems to combine

if [ $(( $num_src % 2 )) -ne 0 ]; then
    echo -e "FATAL: wrong number of script parameters!"
    echo -e "     : The input directories are not in pairs!\n\n"
    printf "$help_message\n\n"
    exit 1;
fi

mkdir -p $output_data_dir/transcription
mkdir -p $output_data_dir/audio

num_warns_def=3;

rm -f $output_data_dir/filelist.list

for i in `seq 0 $(( $num_src / 2 - 1))` ; do
  num_warns=$num_warns_def;
  input_data_dir=${sources[ $[2 * $i] ]}
  input_data_list=${sources[ $((2 * $i + 1)) ]}

  abs_src_dir=`readlink -f $input_data_dir`
  abs_tgt_dir=`readlink -f $output_data_dir`

  if [[ ! -d "$input_data_dir" ]] ; then
    echo "FATAL: input data directory does not exist";
    exit 1;
  fi
  if [[ ! -f "$input_data_list" ]] ; then
    echo "FATAL: input data list file does not exist!";
    exit 1;
  fi

  idl=`basename $input_data_list`
  echo "Making subsets from $input_data_dir according to $idl"

  for file_basename in `cat $input_data_list`; do
      if [[ -e $abs_src_dir/audio/$file_basename.sph ]] ; then
          ln -sf $abs_src_dir/audio/$file_basename.sph $abs_tgt_dir/audio || exit 1
      else
        if [[ -e $abs_src_dir/audio/$file_basename.wav ]] ; then
          ln -sf $abs_src_dir/audio/$file_basename.wav $abs_tgt_dir/audio || exit 1
        else
          echo "File $abs_src_dir/audio/$file_basename.sph|wav does not exist!"  >&2
          exit 1
        fi
      fi

      if [[ -e $abs_src_dir/transcription/$file_basename.txt ]] ; then
          ln -sf $abs_src_dir/transcription/$file_basename.txt $abs_tgt_dir/transcription || exit 1
      else
          if ! $ignore_missing_txt ; then
            echo "File $abs_src_dir/transcription/$file_basename.txt does not exist!"
            exit 1;
          elif [ $num_warns -gt 0 ]; then
            echo "WARNING: File $file_basename.txt does not exist!"
            num_warns=$(($num_warns - 1))
          elif [ $num_warns -eq 0 ]; then
            echo "Not warning anymore"
            num_warns=$(($num_warns - 1))
          fi
      fi
  done
  cat $input_data_list >> $output_data_dir/filelist.list
done


