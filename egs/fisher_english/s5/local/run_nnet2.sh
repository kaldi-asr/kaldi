#!/bin/bash


# This shows what you can potentially run; you'd probably want to pick and choose.
# The ones with _gpu in their name are tuned for GPUs.


use_gpu=true

if $use_gpu; then
  local/nnet2/run_6c_gpu.sh  
fi

