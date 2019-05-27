#!/bin/bash -u

# Copyright 2012  Arnab Ghoshal

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

set -o errexit

function read_dirname () {
  local dir_name=`expr "X$1" : '[^=]*=\(.*\)'`;
  [ -d "$dir_name" ] || { echo "Argument '$dir_name' not a directory" >&2; \
    exit 1; }
  local retval=`cd $dir_name 2>/dev/null && pwd || $dir_name`;
  echo $retval
}

PROG=`basename $0`;
usage="Usage: $PROG <arguments> [options]\n
Converts GlobalPhone audio files from shorten to WAV with error checking.\n
(Must have shorten and sox on PATH).\n\n
Required arguments:\n
  --input-list=FILE\tList of shorten-compressed files to process.\n
  --output-dir=DIR\tDirectory to write the WAV files to.\n
Options:\n
  --output-list=FILE\tWrite list of converted files.\n
  --help\t\t\tPrint this help and exit.\n
";

if [ $# -lt 2 ]; then
  echo -e $usage; exit 1;
fi

while [ $# -gt 0 ];
do
  case "$1" in
  --help) echo -e $usage; exit 0 ;;
  --input-list=*)
  INLIST=`expr "X$1" : '[^=]*=\(.*\)'`; 
  [ -f "$INLIST" ] || { echo "Argument '$INLIST' not a file" >&2; exit 1; }; 
  shift ;;
  --output-dir=*) 
  ODIR=`read_dirname $1`; shift ;;
  --output-list=*)
  OLIST=`expr "X$1" : '[^=]*=\(.*\)'`; shift ;;
  *)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
  esac
done
OLIST=${OLIST:-/dev/null}  # Default for output list

# Checking for shorten and sox. Since 'errexit' option is set, the script will
# terminate if shorten and sox are not found.
which shorten > /dev/null
which sox > /dev/null

tmpdir=$(mktemp -d);
trap 'rm -rf "$tmpdir"' EXIT

mkdir -p $tmpdir/raw $ODIR
shnerr=$tmpdir/shnerr;
soxerr=$tmpdir/soxerr;
nshnerr=0;
nsoxerr=0;

while read line; do
  [[ "$line" =~ ^.*/.*\.adc.shn$ ]] || { echo "Bad line: '$line'"; exit 1; }
  set +e  # Don't want script to die if conversion fails.
  b=`basename $line .adc.shn`; 
  shorten -x $line $tmpdir/raw/${b}.raw;
  if [ $? -ne 0 ]; then
    echo "$line" >> $shnerr;
    let "nshnerr+=1"
  else
    sox -t raw -r 16000 -e signed-integer -b 16 $tmpdir/raw/${b}.raw \
      -t wav $ODIR/${b}.wav
    if [ $? -ne 0 ]; then
      echo "$tmpdir/raw/${b}.raw: exit status = $?" >> $soxerr;
      let "nsoxerr+=1"
    else
      # Just in case there are empty files! Setting the cutoff at 1000 samples,
      # which, assuming 16KHz sampling, is 0.0625 seconds.
      nsamples=`soxi -s "$ODIR/${b}.wav"`;
      if [[ "$nsamples" -gt 1000 ]]; then 
	echo "$ODIR/${b}.wav" >> $OLIST;
      else
	echo "$tmpdir/raw/${b}.raw: #samples = $nsamples" >> $soxerr;
	let "nsoxerr+=1"
      fi
    fi
  fi
  set -e
done < "$INLIST"

[[ "$nshnerr" -gt 0 ]] && \
  echo "shorten: error converting following $nshnerr file(s):" >&2
[ -f "$shnerr" ] && cat "$shnerr" >&2
[[ "$nsoxerr" -gt 0 ]] && \
  echo "sox: error converting following $nsoxerr file(s):" >&2
[ -f "$soxerr" ] && cat "$soxerr" >&2

exit 0;
