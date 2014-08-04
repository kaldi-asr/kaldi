#!/bin/bash

# Copyright (c) 2013, Ondrej Platek, Ufal MFF UK <oplatek@ufal.mff.cuni.cz>
#
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
# limitations under the License. #

expdir=$1; shift
tgtdir=$2; shift

tgtdir="$tgtdir/$name"
date="`date +%F_%T.%N`"

if [[ -d $tgtdir || -f $tgtdir ]] ; then
    tgtdir="$tgtdir/backup_$date"
fi


# This is EXAMPLE SCRIPT you are ENCOURAGED TO CHANGE IT!

mkdir -p "$tgtdir"
cp -rf $expdir "$tgtdir"

# Collect the results

local/results.py $EXP > "$tgtdir"/results.log
echo "Date: $date" >> "$tgtdir"/results.log
size=`du -hs "$tgtdir"`
echo "Size of backup: $size" >> "$tgtdir"/results.log

echo; echo "DATA successfully copied to $tgtdir"; echo
