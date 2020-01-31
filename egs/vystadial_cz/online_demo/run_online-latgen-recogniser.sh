#!/usr/bin/env bash

# source the settings
. ./path.sh

. utils/parse_options.sh || exit 1


# cgdb -q -x .gdbinit_latgen --args \

wav_name=./data/vystadial-sample-cs/test/vad-2013-06-08-22-50-01.897179.wav
onl-rec-latgen-recogniser-test $wav_name \
    --verbose=0  --max-mem=500000000 --lat-lm-scale=15 --config=$MFCC \
    --beam=$beam --lattice-beam=$latbeam --max-active=$max_active \
    $AM $HCLG `cat $SILENCE` $MAT

echo; echo "Converting the lattice to svg picture ${wav_name}.svg" ; echo
fstdraw --portrait=true --osymbols=$WST ${wav_name}.fst | dot -Tsvg  > ${wav_name}.svg
