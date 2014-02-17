#!/bin/bash

# source the settings
. path.sh

beam=16.0
latbeam=10.0
max_active=14000

. $decode_config

batch_size=4560

# Below, there are various commands for debugging, profiling and always
# running the python wrapper around decoder.
# Uncomment convenient prefix for you and put it just before the arguments.
#
# cgdb -q -x .gdbinit_latgen --args python \
# kernprof.py -l -v  \
# valgrind --tool=callgrind -v --dump-instr=yes --trace-jump=yes --callgrind-out-file=callgrind.log python \
python \
pykaldi-latgen-faster-decoder.py $wav_scp $batch_size $pykaldi_latgen_tra $wst \
    --verbose=0  --max-mem=500000000 --lat-lm-scale=15 --config=$mfcc_config \
    --beam=$beam --lattice-beam=$latbeam --max-active=$max_active \
    $model $hclg `cat silence.csl` $lda_matrix

# TODO use --word-penalty=0.0

# If using callgrind display the results by running kcachegrind
# kcachegrind callgrind.log

# reference is named based on wav_scp
./build_reference.py $wav_scp $decode_dir
reference=$decode_dir/`basename $wav_scp`.tra

echo; echo "Reference"; echo
cat $reference
echo; echo "Decoded"; echo
cat $pykaldi_latgen_tra

compute-wer --text --mode=present ark:$reference ark,p:$pykaldi_latgen_tra
