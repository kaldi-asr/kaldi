#!/bin/bash
# when we train the model by nnet3 or chain, we can use this script to show the log property or accuracy
steps/nnet3/report/generate_plots.py --is-chain false exp/nnet3/tdnn_test_kws  report
