#!/bin/bash

# source the settings
. path.sh

. utils/parse_options.sh || exit 1

logname=b${beam}_lb${latbeam}_ma${max_active}_bs${batch_size}

# Below, there are various commands for debugging, profiling.
# Uncomment convenient prefix for you and put it just before the arguments.
#
# cgdb -q -x .gdbinit_latgen --args python \
# valgrind --tool=callgrind -v --dump-instr=yes --trace-jump=yes --callgrind-out-file=callgrind_${logname}.log python \
# kernprof.py -o kernprof_${logname}.log -l -v \
python \
  pykaldi-online-latgen-recogniser.py $wav_scp $batch_size $pykaldi_latgen_tra $WST \
    --verbose=0  --max-mem=500000000 --lat-lm-scale=15 --config=$MFCC \
    --beam=$beam --lattice-beam=$latbeam --max-active=$max_active \
    $AM $HCLG `cat $SILENCE` $MAT

# If using callgrind display the results by running kcachegrind
# kcachegrind callgrind_${logname}.log
# If using kernprof.py @profile decorators 
# to functions which should be profiled.

# reference is named based on wav_scp
./build_reference.py $wav_scp $decode_dir
reference=$decode_dir/`basename $wav_scp`.tra

echo; echo "Reference"; echo
cat $reference
echo; echo "Decoded"; echo
cat $pykaldi_latgen_tra
echo

compute-wer --text --mode=present ark:$reference ark,p:$pykaldi_latgen_tra
