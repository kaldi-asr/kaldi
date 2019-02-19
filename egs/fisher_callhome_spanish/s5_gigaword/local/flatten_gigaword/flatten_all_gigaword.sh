#!/usr/bin/env bash
set -e

# Path to Gigaword corpus with all data files decompressed.
export GIGAWORDDIR=$1
# The directory to write output to
export OUTPUTDIR=$2
# The number of jobs to run at once
export NUMJOBS=$3

echo "Flattening Gigaword with ${NUMJOBS} processes..."
mkdir -p $OUTPUTDIR
find ${GIGAWORDDIR}/data/*/* -type f -print -exec local/flatten_gigaword/run_flat.sh {} ${OUTPUTDIR} \;
echo "Combining the flattened files into one..."
cat ${OUTPUTDIR}/*.flat > ${OUTPUTDIR}/flattened_gigaword.txt
