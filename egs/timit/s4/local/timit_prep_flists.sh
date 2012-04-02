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
set -o pipefail

function read_dirname () {
  local dir_name=`expr "X$1" : '[^=]*=\(.*\)'`;
  [ -d "$dir_name" ] || { echo "Argument '$dir_name' not a directory" >&2; \
    exit 1; }
  local retval=`cd $dir_name 2>/dev/null && pwd || exit 1`
  echo $retval
}

PROG=`basename $0`;
usage="Usage: $PROG <arguments>\n
Prepare train, dev, test file lists for TIMIT.\n\n
Required arguments:\n
  --corpus-dir=DIR\tDirectory for the TIMIT corpus\n
  --dev-spk=FILE\tDevelopment set speaker list\n
  --test-spk=FILE\tCore test set speaker list\n
  --work-dir=DIR\t\tPlace to write the files (in a subdirectory with the 2-letter language code)\n
";

if [ $# -lt 3 ]; then
  echo -e $usage; exit 1;
fi

while [ $# -gt 0 ];
do
  case "$1" in
  --help) echo -e $usage; exit 0 ;;
  --corpus-dir=*) 
  CORPUS=`read_dirname $1`; shift ;;
  --dev-spk=*)
  DEVSPK=`expr "X$1" : '[^=]*=\(.*\)'`; shift ;;
  --test-spk=*)
  TESTSPK=`expr "X$1" : '[^=]*=\(.*\)'`; shift ;;
  --work-dir=*)
  WDIR=`read_dirname $1`; shift ;;
  *)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
  esac
done

if [ ! -d "$CORPUS/train" -a ! -d "$CORPUS/TRAIN" ]; then
  echo "Expecting directory $CORPUS/train or $CORPUS/TRAIN to exist."
  exit 1;
fi

tmpdir=$(mktemp -d);
trap 'rm -rf "$tmpdir"' EXIT

# Get the list of speakers. The list of speakers in the 24-speaker core test 
# set and the 50-speaker development set must be supplied to the script. All
# speakers in the 'train' directory are used for training.
tr '[:upper:]' '[:lower:]' < $DEVSPK > $tmpdir/dev_spk    # Just in case!
tr '[:upper:]' '[:lower:]' < $TESTSPK > $tmpdir/test_spk  # Just in case!

ls -d "$CORPUS"/train/dr*/* | sed -e "s:^.*/::" > $tmpdir/train_spk


ODIR=$WDIR/local  # Directory to write file lists & transcripts
mkdir -p $ODIR

for x in train dev test; do
  # First, find the list of audio files (use only si & sx utterances).
  # Note: train & test sets are under different directories, but doing find on 
  # both and grepping for the speakers will work correctly.
  find $CORPUS/{train,test} -not \( -name 'sa*' \) -name '*.wav' \
    | grep -f $tmpdir/${x}_spk > $ODIR/${x}_sph.flist
  sed -e 's:.*/\(.*\)/\(.*\).wav$:\1_\2:' $ODIR/${x}_sph.flist \
    > $tmpdir/${x}_sph.uttids
  paste $tmpdir/${x}_sph.uttids $ODIR/${x}_sph.flist \
    | sort -k1,1 > $ODIR/${x}_sph.scp

  # Now, get the transcripts: each line of the output contains an utterance 
  # ID followed by the transcript.
  find $CORPUS/{train,test} -not \( -name 'sa*' \) -name '*.phn' \
    | grep -f $tmpdir/${x}_spk > $tmpdir/${x}_phn.flist
  sed -e 's:.*/\(.*\)/\(.*\).phn$:\1_\2:' $tmpdir/${x}_phn.flist \
    > $tmpdir/${x}_phn.uttids
  while read line; do
    [ -f $line ] || error_exit "Cannot find transcription file '$line'";
    cut -f3 -d' ' "$line" | tr '\n' ' ' | sed -e 's: *$:\n:'
  done < $tmpdir/${x}_phn.flist > $tmpdir/${x}_phn.trans
  paste $tmpdir/${x}_phn.uttids $tmpdir/${x}_phn.trans \
    | sort -k1,1 > $ODIR/${x}.trans

  # # Intersect the set of utterances with transcripts with the set of those
  # # with valid audio.
  # cut -f1 $tmpdir/${x}.trans \
  #   | join $tmpdir/${x}_basenames_wav2 - > $tmpdir/${x}_basenames
  # # Get the common set of WAV files and transcripts.
  # join $tmpdir/${x}_basenames $tmpdir/${x}_wav.scp \
  #   > $ODIR/${x}_wav.scp
  # join $tmpdir/${x}_basenames $tmpdir/${x}.trans \
  #   > $ODIR/${x}.trans

  awk '{printf("%s sph2pipe -f wav %s |\n", $1, $2);}' < $ODIR/${x}_sph.scp \
    > $ODIR/${x}_wav.scp

  sed -e 's:_.*$::' $tmpdir/${x}_sph.uttids \
    | paste -d' ' $tmpdir/${x}_sph.uttids - | sort -k1,1 \
    > $ODIR/${x}.utt2spk
  utt2spk_to_spk2utt.pl $ODIR/${x}.utt2spk \
    > $ODIR/${x}.spk2utt;
done
