#!/bin/bash
# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
# Apache 2.0

for f in HCLG.fst words.txt final.mdl fbank.16k.conf; do
        if [ ! -f $f ]; then
                echo "asr-demo: $f is missing."
                exit 1
        fi
done

max_num_utts=15
frames_per_chunk=20
extra_left_context_initial=0
acoustic_scale=0.1
min_active=200
max_active=7000
beam=11.0
lattice_beam=6.0

# your $PATH should be able to find $KALDI_ROOT/src/online2bin/online2-mic-asr-demo
online2-mic-asr-demo \
        --online=true \
        --feature-type=fbank --fbank-config=fbank.conf \
        --frames-per-chunk=${frames_per_chunk} \
        --extra-left-context-initial=${extra_left_context_initial} \
        --acoustic-scale=${acoustic_scale} \
        --min-active=${min_active} --max-active=${max_active} \
        --beam=${beam} --lattice-beam=${lattice_beam} \
        --word-symbol-table=words.txt \
        --do-endpointing=true --endpoint.silence-phones=1 \
        --max-num-utts=$max_num_utts \
        final.mdl HCLG.fst ark:/dev/null
