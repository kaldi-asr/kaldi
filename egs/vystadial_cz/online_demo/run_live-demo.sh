#!/bin/bash

# source the settings
. path.sh

batch_size=4560
beam=12.0
latbeam=6.0
max_active=2000

# cgdb -q -x .gdbinit_faster --args python \
python \
live-demo.py $batch_size $WST \
    --verbose=0 --lat-lm-scale=15 --config=$MFCC \
    --beam=$beam --lattice-beam=$latbeam --max-active=$max_active \
    $AM $HCLG `cat $SILENCE` $MAT
