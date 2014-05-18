#!/bin/bash


# This shows what you can potentially run; you'd probably want to pick and choose.

use_gpu=true

if $use_gpu; then
  local/nnet2/run_5a_gpu.sh  # 100h subset, on top of fMLLR.  
  local/nnet2/run_6a_gpu.sh  # as 5a but realign then train again.

  local/nnet2/run_5b_gpu.sh  # 100h subset, unadapted recipe with multiple VTLN warp factors.

  local/nnet2/run_5c_gpu.sh  # full training set, on top of fMLLR.

  local/nnet2/run_5d_gpu.sh  # 100h subset, on top of fMLLR, p-norm recipe.

  # local/nnet2/run_5e_gpu.sh  # 100h subset, on top of fMLLR, p-norm recipe, on top
  #                            # of raw-fMLLR features; you need to run local/run_raw_fmllr.sh
  #                            # first.  You probably don't want to do this; it's worse.
else
  local/nnet2/run_5a.sh # 100h subset, on to of fMLLR.
fi

