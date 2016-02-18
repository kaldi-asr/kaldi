#!/bin/bash
# Copyright (c) 2015, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -o nounset                              # Treat unset variables as an error


if [ ! -x  results ] ; then
  data=$(readlink -f ./local)
  data=$(dirname $data)
  mkdir -p $data/results
  ln -s $data/results results
fi

if [ ! -e ./RESULTS ] ; then
  p=$(basename `readlink -f lang.conf`)
  p=${p##.*}
  filename=results.${p}.${USER}.$(date --iso-8601=seconds)
  echo "#Created on $(date --iso-8601=seconds) by $0" >> results/$filename
  ln -sf  results/$filename RESULTS
fi


(
  echo -e "#\n# STT Task performance (WER), evaluated on $(date --iso-8601=seconds)"
  for f in exp/*; do
    find $f -path "*dev10h.pem*" -name "*.sys" -not -name "*char*" | xargs grep Avg | utils/best_wer.sh
  done
  for f in exp_bnf/*; do
    find $f -path "*dev10h.pem*" -name "*.sys" -not -name "*char*" | xargs grep Avg | utils/best_wer.sh
  done
)  >> RESULTS

(
  ls exp/tri5/decode*dev10h*/score_*/*char*sys >/dev/null 2>&1  || exit 0
  echo -e "#\n# STT Task performance (CER), evaluated on $(date --iso-8601=seconds)"
  for f in exp/*; do
    find $f -path "*dev10h.pem*" -name "*.sys" -name "*char*" | xargs grep Avg | utils/best_wer.sh
  done
  for f in exp_bnf/*; do
    find $f -path "*dev10h.pem*" -name "*.sys" -name "*char*" | xargs grep Avg | utils/best_wer.sh
  done
)  >> RESULTS

