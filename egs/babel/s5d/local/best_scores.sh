#!/usr/bin/env bash
# Copyright (c) 2015, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -o nounset                              # Treat unset variables as an error


if [ ! -x  results ] ; then
  data=$(utils/make_absolute.sh ./local)
  data=$(dirname $data)
  mkdir -p $data/results
  ln -s $data/results results
fi

if [ ! -e ./RESULTS ] ; then
  p=$(basename `utils/make_absolute.sh lang.conf`)
  p=${p##.*}
  filename=results.${p}.${USER}.$(date --iso-8601=seconds)
  echo "#Created on $(date --iso-8601=seconds) by $0" >> results/$filename
  ln -sf  results/$filename RESULTS
fi


set -f
export mydirs=( `find exp/ exp_bnf/ exp_psx/ -name "decode*dev10h.pem*" -type d | sed  's/it[0-9]/*/g;s/epoch[0-9]/*/g' | sort -u` )
set +f
(
  echo -e "#\n# STT Task performance (WER), evaluated on $(date --iso-8601=seconds) by user `whoami` on `hostname -f`"
  for f in  "${mydirs[@]}"; do
    find $f -name "*.sys" -not -name "*char*" | xargs grep Avg | utils/best_wer.sh
  done | column -t
)  >> RESULTS

(
  ls exp/tri5/decode*dev10h*/score_*/*char*sys >/dev/null 2>&1  || exit 0
  echo -e "#\n# STT Task performance (CER), evaluated on $(date --iso-8601=seconds) by user `whoami` on `hostname -f`"
  for f in  "${mydirs[@]}"; do
    find $f -name "*.sys" -name "*char*" | xargs grep Avg | utils/best_wer.sh
  done | column -t
)  >> RESULTS

