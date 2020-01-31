#!/usr/bin/env bash

# Copyright 2014 QCRI (author: Ahmed Ali)
# Copyright 2016 Johns Hopkins Univeersity (author: Jan "Yenda" Trmal)
# Copyright 2019 Johns Hopkins Univeersity (author: Jinyi Yang)
# Apache 2.0


echo $0 "$@"

tdtData=$(utils/make_absolute.sh "${@: -1}" );
wavedir=$tdtData/wav
mkdir -p $wavedir


length=$(($#-1))
args=${@:1:$length}

# Check if sph2pipe is installed
sph2pipe=`which sph2pipe` || sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
[ ! -x $sph2pipe ] && echo "Could not find the sph2pipe program at $sph2pipe" && exit 1;
set -e -o pipefail

for var in $args; do
  CD=$(basename $var)
  [ -d $wavedir/$CD ] && rm -rf $wavedir/$CD
  mkdir -p $wavedir/$CD
  find $var -type f -name *.sph | grep "MAN" | while read file; do
    f=$(basename $file)
    if [[ ! -L "$wavedir/$CD/$f" ]]; then
      ln -sf $file $wavedir/$CD/$f
    fi
  done
done

#figure out the proper sph2pipe command line
(
  for w in `find $wavedir -name *.sph` ; do
    base=`basename $w .sph`
    fullpath=`utils/make_absolute.sh $w`
    echo "$base $sph2pipe -f wav -p -c 1 $fullpath |"
  done
) | sort -u > $tdtData/wav.scp

#clean
rm -fr $tdtData/id$$ $tdtData/wav$$
echo "$0: data prep audio succeded"

exit 0

