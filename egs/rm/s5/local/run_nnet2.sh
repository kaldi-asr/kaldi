#!/bin/bash


# This example runs on top of "raw-fMLLR" features:
local/nnet2/run_4a.sh


# This one is on top of filter-bank features, with only CMN.
local/nnet2/run_4b.sh

# This one is on top of 40-dim + fMLLR features
local/nnet2/run_4c.sh

