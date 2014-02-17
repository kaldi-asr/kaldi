#!/bin/bash

# data location
PWD=`pwd`
exp_dir=$PWD
data_dir=$PWD/vystadial-sample-cs/test
decode_dir=$exp_dir/decode

# IO parameters
wav_scp=$data_dir/input_best.scp
# wav_scp=$data_dir/input.scp

gmm_latgen_faster_tra=$decode_dir/gmm-latgen-faster.tra
gmm_latgen_faster_tra_txt=${gmm_latgen_faster_tra}.txt

pykaldi_latgen_tra=$decode_dir/pykaldi-latgen.tra
pykaldi_latgen_tra_txt=${pykaldi_faster_tra}.txt
lattice=$decode_dir/lat.gz
# common configs
mfcc_config=mfcc.conf
decode_config=conf/decode.conf

wst=$exp_dir/words.txt
silent_phones=silence.csl

if [ -f tri2b.mat ] ; then
    # Note that $model has to be trained with LDA enabled if using LDA
    lda_matrix="$exp_dir/tri2b.mat"
    model=$exp_dir/tri2b.mdl
    hclg=$exp_dir/HCLG.fst
else
    # if no LDA matrix specified -> use delta + delta-delta
    model=$exp_dir/tri2a.mdl
    hclg=$exp_dir/HCLG.fst  # different HCLG.fst
    lda_matrix=""
fi

pykaldi_dir=`pwd`/../..
export LD_LIBRARY_PATH=$pykaldi_dir:$LD_LIBRARY_PATH
export PYTHONPATH=$pykaldi_dir:$pykaldi_dir/pyfst:$PYTHONPATH
