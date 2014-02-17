#!/bin/bash

# source the settings
. path.sh

batch_size=4560
beam=10.0
latbeam=6.0
max_active=7000

python live-demo.py $batch_size $wst \
    --verbose=0 --lat-lm-scale=15 --config=$mfcc_config \
    --beam=$beam --lattice-beam=$latbeam --max-active=$max_active \
    $model $hclg 1:2:3:4:5 $lda_matrix
