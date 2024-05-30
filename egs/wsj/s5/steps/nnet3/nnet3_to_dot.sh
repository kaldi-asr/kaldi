#!/usr/bin/env bash

# script showing use of nnet3_to_dot.py
# Copyright 2015  Johns Hopkins University (Author: Vijayaditya Peddinti).

# Begin configuration section.
component_attributes="name,type"
node_prefixes=""
info_bin=nnet3-am-info
echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [opts] <nnet3-mdl-file> <output-dot-file> <output-png-file>"
  echo " e.g.: $0 exp/sdm1/nnet3/lstm_sp/0.mdl lstm.dot lstm.png"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --info-bin <nnet3-am-info|nnet3-info>        # Name of the binary to generate the nnet3 file"
  echo "  --component-attributes <string|name,type>     # attributes to be printed in nnet3 components"
  echo "  --node-prefixes <string|Lstm1,Lstm2>          # list of prefixes. Nnet3 components/component-nodes with the same prefix"
  echo "                                                # will be clustered together in the dot-graph"


  exit 1;
fi

model=$1
dot_file=$2
output_file=$3

attr=${node_prefixes:+ --node-prefixes "$node_prefixes"}
$info_bin $model | \
  steps/nnet3/dot/nnet3_to_dot.py \
    --component-attributes "$component_attributes" \
    $attr $dot_file
echo "Generated the dot file $dot_file"

command -v dot >/dev/null 2>&1 || { echo >&2 "This script requires dot but it's not installed. Please compile $dot_file with dot"; exit 1; }
dot -Tpdf $dot_file -o $output_file
