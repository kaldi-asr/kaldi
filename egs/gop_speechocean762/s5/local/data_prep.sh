#!/usr/bin/env bash

# Copyright 2020-2021  Xiaomi Corporation (Author: Junbo Zhang, Yongqing Wang)
# Apache 2.0

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 /home/storage07/zhangjunbo/data/speechocean762/test data/test"
  exit 1
fi

src=$1
dst=$2

[ ! -d $src ] && echo "$0: no such directory $src" && exit 1;
[ ! -d $src/../WAVE ] && echo "$0: no wav directory" && exit 1;

wavedir=`realpath $src/../WAVE`

[ -d $dst ] || mkdir -p $dst || exit 1;

cp -Rf $src/* $dst/ || exit 1;

sed -i.ori "s#WAVE#${wavedir}#" $dst/wav.scp || exit 1

utils/validate_data_dir.sh --no-feats $dst || exit 1;

echo "$0: successfully prepared data in $dst"

exit 0
