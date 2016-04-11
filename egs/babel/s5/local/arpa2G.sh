#!/bin/bash
# Copyright 2013  Johns Hopkins University (authors: Yenda Trmal)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

#Simple utility script to convert the gzipped ARPA lm into a G.fst file



#no configuration here
#end configuration section.

echo $0 $@

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ]  && . ./cmd.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0  <arpa-lm-file> <lang-dir> <dest-dir>"
  exit 1;
fi

lmfile=$1
langdir=$2
destdir=$3

mkdir $destdir 2>/dev/null || true

gunzip -c $lmfile | \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$langdir/words.txt - $destdir/G.fst || exit 1
fstisstochastic $destdir/G.fst || true

exit 0
