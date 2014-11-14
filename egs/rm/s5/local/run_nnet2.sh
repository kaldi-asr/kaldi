#!/bin/bash

# You don't have to run all these.  
# you can pick and choose.  Look at the RESULTS file..


use_gpu=true  

if $use_gpu; then
  # This example runs on top of "raw-fMLLR" features.
  # We don't have a GPU version of this script.
  #local/nnet2/run_4a_gpu.sh

  # This one is on top of filter-bank features (VTLN-perturbed), 
  # with only CMN.
  local/nnet2/run_4b_gpu.sh

  # This one is on top of 40-dim + fMLLR features.
  local/nnet2/run_4c.sh --use-gpu true

  # This one is for training pnorm nnets on top of 40-dim + fMLLR features
  # **THIS IS THE PRIMARY RECIPE**
  local/nnet2/run_4d3.sh --use-gpu true

  # this is the old version of the run_4d3.sh script, before
  # switching to more compact version of egs.
  #local/nnet2/run_4d.sh --use-gpu true

  # as above with 'perturbed training'.  A bit better results, a bit slower.
  local/nnet2/run_4d2.sh --use-gpu true
  
  # This is discriminative training on top of 4c.  (hardly helps)
  local/nnet2/run_5c_gpu.sh
  
  # This is discriminative training on top of 4d.
  local/nnet2/run_5d.sh --use-gpu true
else
  # This example runs on top of "raw-fMLLR" features;
  # you have to run local/run_raw_fmllr.sh first.
  local/nnet2/run_4a.sh

  # This one is on top of filter-bank features, with only CMN.
  local/nnet2/run_4b.sh

  # This one is on top of 40-dim + fMLLR features, it's a fairly
  # normal tanh system.
  local/nnet2/run_4c.sh --use-gpu false

  # **THIS IS THE PRIMARY RECIPE (40-dim + fMLLR + p-norm neural net)**
  local/nnet2/run_4d.sh --use-gpu false

  # as above with 'perturbed training'.  A bit better results, a bit slower.
  local/nnet2/run_4d2.sh --use-gpu false

  # This is discriminative training on top of 4c.
  local/nnet2/run_5c.sh

  # This is discriminative training on top of 4d.
  local/nnet2/run_5d.sh --use-gpu false

  # This is p-norm on top of raw-fMLLR.
  #local/nnet2/run_4e.sh

fi
  

