#!/bin/bash

# This script provides normalized scoring for the various slam dataset languages.
# This is especially important for Farsi as this script scores using various normalizations
# on things like Farsi vs Arabic yeh.
# Usage e.g.:
# local/normalized_scoring/normalized_scoring.sh output exp_yomdle_farsi/chain/cnn_e2eali_1b/decode_test/scoring_kaldi/penalty_1.0/7.txt farsi

OUTDIR=$1
HYP_FILE=$2
LANG=$3

cat ${HYP_FILE} | cut -d' ' -f1 > ids
cat ${HYP_FILE} | cut -d' ' -f2- | python3 local/bidi.py > hyp
paste -d' ' ids hyp | python3 local/normalized_scoring/convert2snor.py > hyp_file.txt

local/normalized_scoring/score.sh ${OUTDIR} hyp_file.txt ${LANG}
