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

function error_exit () {
  echo -e "$@" >&2; exit 1;
}

function read_dirname () {
  local dir_name=`expr "X$1" : '[^=]*=\(.*\)'`;
  [ -d "$dir_name" ] || error_exit "Argument '$dir_name' not a directory";
  local retval=`cd $dir_name 2>/dev/null && pwd || exit 1`
  echo $retval
}

PROG=`basename $0`;
usage="Usage: $PROG <arguments>\n
Prepare train, dev, eval file lists for a language.\n
e.g.: $PROG --config-dir=conf --corpus-dir=corpus --languages=\"GE PO SP\"\n\n
Required arguments:\n
  --config-dir=DIR\tDirecory containing the necessary config files\n
  --corpus-dir=DIR\tDirectory for the GlobalPhone corpus\n
  --languages=STR\tSpace separated list of two letter language codes\n
";

if [ $# -lt 3 ]; then
  error_exit $usage;
fi

while [ $# -gt 0 ];
do
  case "$1" in
  --help) echo -e $usage; exit 0 ;;
  --config-dir=*)
  CONFDIR=`read_dirname $1`; shift ;;
  --corpus-dir=*)
  GPDIR=`read_dirname $1`; shift ;;
  --languages=*)
  LANGUAGES=`expr "X$1" : '[^=]*=\(.*\)'`; shift ;;
  *)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
  esac
done

# (1) check if the config files are in place:
pushd $CONFDIR > /dev/null
[ -f dev_spk.list ] || error_exit "$PROG: Dev-set speaker list not found.";
[ -f eval_spk.list ] || error_exit "$PROG: Eval-set speaker list not found.";
[ -f lang_codes.txt ] || error_exit "$PROG: Mapping for language name to 2-letter code not found.";

popd > /dev/null
[ -f path.sh ] && . ./path.sh  # Sets the PATH to contain necessary executables

# (2) get the various file lists (for audio, transcription, etc.) for the
# specified language.
printf "Preparing file lists ... "
for L in $LANGUAGES; do
  mkdir -p data/$L/local/data
  local/gp_prep_flists.sh --corpus-dir=$GPDIR --dev-spk=$CONFDIR/dev_spk.list \
    --eval-spk=$CONFDIR/eval_spk.list --lang-map=$CONFDIR/lang_codes.txt \
    --work-dir=data $L >& data/$L/prep_flists.log & 
  # Running these in parallel since this does audio conversion (to figure out
  # which files cannot be processed) and takes some time to run. 
done
wait;
echo "Done"

# (3) Normalize the transcripts.
for L in $LANGUAGES; do
  printf "Language - ${L}: normalizing transcripts ... "
  for x in train dev eval; do
    local/gp_norm_trans_${L}.pl -i data/$L/local/data/${x}_${L}.trans1 \
      > data/$L/local/data/${x}_${L}.txt;
  done
  echo "Done"
done

# (4) Create a directories to contain files needed in training and testing:
for L in $LANGUAGES; do
  printf "Language - ${L}: formatting train/test data ... "
  for x in train dev eval; do
    mkdir -p data/$L/$x
    cp data/$L/local/data/${x}_${L}_wav.scp data/$L/$x/wav.scp
    cp data/$L/local/data/${x}_${L}.txt data/$L/$x/text
    cp data/$L/local/data/${x}_${L}.spk2utt data/$L/$x/spk2utt
    cp data/$L/local/data/${x}_${L}.utt2spk data/$L/$x/utt2spk
  done
  echo "Done"
done


echo "Finished data preparation."
