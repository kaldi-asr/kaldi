#!/bin/bash

. cmd.sh

id=`nvidia-smi | awk '{print $2}' | grep [0-9] | sort | uniq -c | sort -k1 -n | head -n 1 | awk '{print$2}'`

$KALDI_ROOT/tools/cuedrnnlm/rnnlm -device $id $@
