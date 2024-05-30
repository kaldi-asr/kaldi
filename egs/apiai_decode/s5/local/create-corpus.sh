#!/usr/bin/env bash

# Checking arguments
if [ $# -le 1 ]; then
  echo "Use $0 <datadir> test1.wav [test2.wav] ..."
  echo "  $0 data/test-corpus test1.wav test2.wav"
  exit 0;
fi

CORPUS=$1
shift
for file in "$@"; do
  if [[ "$file" != *.wav ]]; then
    echo "Expecting .wav files, got $file"
    exit 1;
  fi

  if [ ! -f "$file" ]; then
    echo "$file not found";
    exit 1;
  fi
done;


echo "Initilizing $CORPUS"
if [ ! -d "$CORPUS" ]; then
  echo "Creating $CORPUS directory"
  mkdir -p "$CORPUS" || ( echo "Unable to create data dir" && exit 1 )
fi;

wav_scp="$CORPUS/wav.scp"
spk2utt="$CORPUS/spk2utt"
utt2spk="$CORPUS/utt2spk"
text="$CORPUS/text"

#nulling files
cat </dev/null >$wav_scp
cat </dev/null >$spk2utt
cat </dev/null >$utt2spk
cat </dev/null >$text
rm $CORPUS/feats.scp 2>/dev/null;
rm $CORPUS/cmvn.scp  2>/dev/null;

for file in "$@"; do
  id=$(echo $file | sed -e 's/ /_/g')
  echo "$id $file" >>$wav_scp
  echo "$id $id" >>$spk2utt
  echo "$id $id" >>$utt2spk
  echo "$id NO_TRANSRIPTION" >>$text
done;
