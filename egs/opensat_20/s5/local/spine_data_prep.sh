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
  base=$(basename $line .sph)
  base=$(basename $base .SPH)
  path=$(dirname $line)
  wavid="${base}xxxxxxxxxxxxxxxxxxxxx"
  wavid=$(echo $wavid |  tr '[:upper:]' '[:lower:]')
  wavid=$(echo $wavid |  sed 's/_unpr//g' | sed 's/_pr//g')
  wavid="${wavid:0:20}"  #35 found by experimenation
  echo "${wavid}_A sph2pipe -f wav -p  -c 1 $line|"
  echo "${wavid}_B sph2pipe -f wav -p  -c 2 $line|"
done < <(find -L ${AUDIO_DIR} -iname "*.sph" -and -not -iname '*_unpr*' ) | sort -u > $OUTPUT/wav.scp

if [  "${TEXTS_DIR:-x}" == "x"  ]; then
  exit 0
fi

while IFS= read -r line ; do
  base=$(basename $line .typ)
  base=$(basename $base .txt)
  path=$(dirname $line)
  wavid=$base
  wavid=$(echo "$wavid" | sed 's/_unpr//g' |  tr '[:upper:]' '[:lower:]')
  wavid=$(echo $wavid |  sed 's/_pr//g')
  wavid="${wavid}xxxxxxxxxxxxxxxxxxxxx"
  wavid="${wavid:0:20}"  #40 found by experimenation
  echo "$wavid $line"
done < <(find -L ${TEXTS_DIR} \( -name "*typ" -or -name '*.txt' \) ) | sort -u > $OUTPUT/texts.scp


./local/spine_parse_texts.py $OUTPUT/texts.scp > $OUTPUT/transcripts


exit 0
