#!/bin/bash
OUTDIR=$1
HYP_FILE=$2
LANG=$3

cat ${HYP_FILE} | cut -d' ' -f1 > ids
cat ${HYP_FILE} | cut -d' ' -f2- | python3 local/bidi.py > hyp
paste -d' ' ids hyp | python3 local/normalized_scoring/convert2snor.py > hyp_file.txt

local/normalized_scoring/score.sh ${OUTDIR} hyp_file.txt ${LANG}

#rm ids hyp hyp_file.txt
