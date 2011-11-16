#!/bin/bash

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

