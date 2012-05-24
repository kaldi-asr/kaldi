#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation

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


fake=false
if [ "$1" == "--fake" ]; then
    fake=true
    shift
fi

sphdir=$1 # e.g. /mnt/matylda2/data/RM
wavdir=$2 # e.g. /mnt/matylda6/jhu09/qpovey/kaldi_rm_wav
flistin=$3 # e.g. train_sph.flist, contains sph files in sphdir
flistout=$4 # e.g. train_wav.flist, contains wav files in wavdir


if [ $fake == false ]; then
    for x in `cat $flistin`; do 
        y=`echo $x | sed s:$sphdir:$wavdir: | sed s:.sph:.wav:`;
        mkdir -p `dirname $y`
        ../../tools/sph2pipe_v2.5/sph2pipe -f wav $x $y || exit 1;
    done 
fi

cat $flistin | sed s:$sphdir:$wavdir: | sed s:.sph:.wav: > $flistout || exit 1;

