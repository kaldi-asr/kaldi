#!/bin/bash

# Copyright   2019  Idiap Research Institute (Author: Srikanth Madikeri).  Apache 2.0.
# This script is the equivalent of get_successful_models function in the python library.
# It takes a list of models and returns either the best model (the deafult) or a list of
# models to average.

models_to_average=false
difference_threshold=1.0
output=output


# echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -le 1 ]; then
    echo "Usage: $0: [options] <model-1-log> <model-2-log> .... <model-N-log>"
    echo "where <model-n> is one of the n models to choose from."
    echo ""
    echo "--models-to-average: when true, returns the models to be averaged rather than the single best model"
    echo "--difference-threshold: used to reject models. models with objf < max-value - difference_threshold are rejected"
    echo "--output: the objf of the this output layer is used for model selection"
    echo ""
    exit 1;
fi

if ! $models_to_average; then
    if [ $# -eq 1 ]; then
        basename $1 | tr '.' ' ' | awk '{ print $(NF-1) }'
        exit 0;
    fi
    model_log_list=$(for arg in $*; do echo $arg; done)
    first_log=$1
    log_line=`fgrep -m 1 "Overall average objective function for '$output' is" $first_log`
    colno=`echo $log_line | cut -d '=' -f1 | wc -w`
    ((colno+=2))
    filename=$(fgrep -m 1 "Overall average objective function for '$output' is" $model_log_list | \
        cut -d ' ' -f1,$colno | tr ':' ' ' | \
        awk '{print $1,$3}' | \
        sort -k2,2 -g | tail -1 | cut -d ' ' -f1)
    basename $filename | tr '.' ' ' | awk '{ print $(NF-1) }'
fi
