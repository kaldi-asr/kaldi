#!/bin/bash
# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely, Mirko Hannemann)
# Licensed under the Apache License, Version 2.0 (the "License")

#begin configuration section.
tmp_root=/mnt/scratch01/tmp/$USER # will be deleted after not using 30days.
#end configuration section.

#echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 1 ]; then
  echo "Usage: $0 [--tmp-root <scratch-dir>] <local-dir>"
  echo "options:"
  echo "  --tmp-root DIR  # root dir, where the tmpdirs get created"
  exit 1;
fi

local_dir=$1
[ -e $local_dir ] && echo "$0: exiting... (dir '$local_dir' already exists)" && exit 0

# create tmp dir,
[ ! -e $tmp_root ] && mkdir -p $tmp_root 
[ ! -e $tmp_root ] && echo "$0: cannot create dir '$tmp_root'" && exit 1;
tmp_dir=$(mktemp -d --tmpdir=$tmp_root kaldi_tmp_dir.$(basename $local_dir).XXXXXXXXXXXXX) || exit 1
chmod g+rX $tmp_dir # make it readable for group

# create parent of local dir
local_dir_parent=$(dirname $local_dir)
[ ! -e $local_dir_parent ] && mkdir -p $local_dir_parent

# make the symbolic link,
ln -s $tmp_dir $local_dir || exit 1

# make backlink, so the tmpdir is traceable to kaldi dir,
local_dir_abs=$local_dir; [ "${local_dir: 0 : 1}" != "/" ] && local_dir_abs=$PWD/$local_dir
echo "$local_dir_abs" > $tmp_dir/BACK_LINK_TO_KALDI

echo "$0: success! (linked : $local_dir <- $tmp_dir)"
