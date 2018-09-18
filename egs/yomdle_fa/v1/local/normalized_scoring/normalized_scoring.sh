#!/bin/bash

# This script provides normalized scoring for the various slam dataset languages.
# This is especially important for Farsi as this script scores using various normalizations
# on things like Farsi vs Arabic yeh.
# Usage e.g.:
# local/normalized_scoring/normalized_scoring.sh output exp_yomdle_farsi/chain/cnn_e2eali_1b/decode_test/scoring_kaldi/penalty_1.0/7.txt data_yomdle_farsi/test/text.old farsi

OUTDIR=$1
HYP_FILE=$2
REF_FILE=$3
LANG=$4

mkdir -p $OUTDIR
cat ${HYP_FILE} | cut -d' ' -f1 > $OUTDIR/hyp_ids
cat ${HYP_FILE} | cut -d' ' -f2- > $OUTDIR/hyp_utt
paste -d' ' $OUTDIR/hyp_ids $OUTDIR/hyp_utt | python3 local/normalized_scoring/convert2snor.py > $OUTDIR/hyp_file.txt

cat ${REF_FILE} | cut -d' ' -f1 > $OUTDIR/ref_ids
cat ${REF_FILE} | cut -d' ' -f2- | python3 local/bidi.py > $OUTDIR/ref_utt
paste -d' ' $OUTDIR/ref_ids $OUTDIR/ref_utt | python3 local/normalized_scoring/convert2snor.py > $OUTDIR/ref_file.txt

local/normalized_scoring/score.sh ${OUTDIR} $OUTDIR/hyp_file.txt $OUTDIR/ref_file.txt ${LANG}
