#!/bin/bash
# Copyright (c) 2020, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
cmd=run.pl
debug=false
# End configuration section
. ./utils/parse_options.sh
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

if [ $# -eq 2 ] ; then
  AUDIO_DIR=$1
  OUTPUT=$2
elif [ $# -eq 3 ] ;  then
  AUDIO_DIR=$1
  TEXTS_DIR=$2
  OUTPUT=$3
else
  echo >&2  "Unexpected number of arguments -- only 2 or three arguments are ok"
fi

mkdir -p $OUTPUT
while IFS= read -r line ; do
  base=$(basename $line .flac)
  path=$(dirname $line)
  base=$(echo $base | sed 's/_mixed//g')
  base=$(echo $base | sed 's/_dev//g')
  wavid="${base}"
  echo "$wavid flac -s -c -d $line | sox - -b 16 -t wav -r 16000 - |"
done < <(find -L ${AUDIO_DIR} -name "*.flac" ) | sort -u > $OUTPUT/wav.scp

if [  "${TEXTS_DIR:-x}" == "x"  ]; then
  exit 0
fi

while IFS= read -r line ; do
  base=$(basename $line .tsv)
  path=$(dirname $line)
  wavid=$base
  wavid="${base}_xxxxxxx"
  wavid="${wavid:0:35}"  #40 found by experimenation
  echo "$wavid $line"
done < <(find -L ${TEXTS_DIR} -name "*tsv" ) | sort -u > $OUTPUT/texts.scp


./local/safet_parse_texts.py $OUTPUT/texts.scp > $OUTPUT/transcripts

exit 0
