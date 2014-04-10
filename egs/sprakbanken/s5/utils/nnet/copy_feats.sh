#!/bin/bash
# Copyright 2012  Brno University of Technology (author: Karel Vesely)

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

# This script resaves features to a specified directory,
# this is done to have the randomized data stored consecutivly,
# which improves the speed and reduces loads on disks.
#
# To make sure the temporary dir gets deleted upon exit of the calling script
# you can use something like:
#
# trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; rm -r $tmpdir" EXIT


echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
   echo "Usage: $0 <input.scp> <tmpdir> <output.scp>"
   echo " e.g.: $0 train_remote.scp /tmp/324nkjl train_local.scp"
   exit 1;
fi

scp_in=$1
tmpdir=$2
scp_out=$3

echo "Re-saving the features to tmpdir $tmpdir @ $(hostname)"
#divide the arks per 10k files
nj=$((1 + $(cat $scp_in | wc -l) / 10000))
for((n=0; n<nj; n++)); do
  copy-feats "scp:utils/split_scp.pl -j $nj $n $scp_in - |" ark,scp:$tmpdir/feats.$n.ark,$tmpdir/feats.$n.scp || exit 1
done
#assemble the scp file
for((n=0; n<nj; n++)); do
  cat $tmpdir/feats.$n.scp
done > $scp_out
#test we have all the data
l1=$(cat $scp_in | wc -l)
l2=$(cat $scp_out | wc -l)
[[ "$l1" != "$l2" ]] && echo "ERROR in data re-saving $l1 != $l2" && exit 1;
#notify it was copied ok
wc -l $scp_in $scp_out
echo Copied ok!

exit 0

