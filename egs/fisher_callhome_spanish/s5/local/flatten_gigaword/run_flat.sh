#!/usr/bin/env bash
set -e

. ./path_venv.sh

# Path to Gigaword corpus with all data files decompressed.
GIGAWORDPATH=$1
# The directory to write output to
OUTPUTDIR=$2
file=$(basename ${GIGAWORDPATH})
if [ ! -e ${OUTPUTDIR}/${file}.flat ]; then
    echo "flattening to ${OUTPUTDIR}/${file}.flat"
    python local/flatten_gigaword/flatten_one_gigaword.py --gigaword-path ${GIGAWORDPATH} --output-dir ${OUTPUTDIR}
else
    echo "skipping ${file}.flat"
fi

