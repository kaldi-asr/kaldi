#!/usr/bin/env bash
# Copyright 2017 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

. ./cmd.sh
. ./path.sh

# Config:

if [ $# != 1 ]; then
   echo "Wrong #arguments ($#, expected 1)"
   echo "Usage: local/write_se_results.sh <enhancement-method>"
   exit 1;
fi

enhancement=$1

echo -e "PESQ ($enhancement) \t dt05_simu=$(cat exp/compute_pesq_$enhancement/pesq_dt05) \t et05_simu=$(cat exp/compute_pesq_$enhancement/pesq_et05)"
echo -e "STOI ($enhancement) \t dt05_simu=$(cat exp/compute_stoi_estoi_sdr_$enhancement/${enhancement}_dt05_STOI) \t et05_simu=$(cat exp/compute_stoi_estoi_sdr_$enhancement/${enhancement}_et05_STOI)"
echo -e "eSTOI ($enhancement) \t dt05_simu=$(cat exp/compute_stoi_estoi_sdr_$enhancement/${enhancement}_dt05_eSTOI) \t et05_simu=$(cat exp/compute_stoi_estoi_sdr_$enhancement/${enhancement}_et05_eSTOI)"
echo -e "SDR ($enhancement) \t dt05_simu=$(cat exp/compute_stoi_estoi_sdr_$enhancement/${enhancement}_dt05_SDR) \t et05_simu=$(cat exp/compute_stoi_estoi_sdr_$enhancement/${enhancement}_et05_SDR)"
echo ""
