#!/bin/bash

# Copyright 2017  University of Sheffield (Author: Ning Ma)
#           2017  Ruhr-University Bochum (Author: Hendrik Meutzner)
#
# Apache 2.0.

# This script calls run_dnn.sh with different feature configurations.

# When running run_dnn.sh, it is possible to skip some stages by providing an additional argument:
# --stage <stage>       # 0: no skipping (default)
#                       # 1: skip data preparation
#                       # 2: skip language preparation
#                       # 3: skip feature extraction
#                       # 4: skip training
#                       # 5: skip decoding
#                       # 6: skip DNN experiments

# Run experiments
./run_dnn.sh mfcc  	# MFCC features only
./run_dnn.sh video 	# Video features only
./run_dnn.sh av    	# MFCC + video features (using early integration)

./run_dnn.sh fbank  # Filterbank features only
./run_dnn.sh av2    # Filterbank + video features (using early integration)

# Summarize results in a table
python ./local/summarize_scores.py --exp "./proc/exp"
#python ./local/summarize_scores.py --exp "./proc/exp" --models "dnn-v1,dnn-v1 (it5)"
