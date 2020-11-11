#!/usr/bin/env bash
if [[ $1 && $2 ]]; then

  local=`pwd`/local

  mkdir -p data data/local data/$1 data/$2

  echo "Preparing train and test data"
  echo "make wav.scp for $1 $2"

  rm -rf data/mfcc data/log

  cd asr_swahili/data

  pushd $1
  cp wav.scp $local/../data/$1/wav.scp
  popd

  pushd $2/wav5
  ls */*.wav |  sed 's/^/asr_swahili\/data\/test\/wav5\//g' > tutu1
  cat tutu1 | sed "s/\//#/g" | awk 'BEGIN{FS="#"} {print $6}' | sed "s/\.wav//g" > tutu2
  paste tutu2 tutu1 > $local/../data/$2/wav.scp
  rm tutu1 tutu2
  popd

  echo "copy spk2utt, utt2spk, text for $1 $2"

  for x in $1 $2; do
    cp $x/spk2utt $local/../data/$x/.
    cp $x/utt2spk $local/../data/$x/.
    cp $x/text $local/../data/$x/.
  done

  pushd $local/../data/local
  if [ ! -f  "swahili.arpa" ]; then
    cd $local/../asr_swahili/LM
    unzip swahili.arpa.zip -d $local/../data/local/
  fi
  popd

  echo "Preparing data OK."

  cd ../..
else
  echo "ERROR: Preparing train and test data failed !"
  echo "You must have forgotten to precise train test directories"
  echo "Usage: ./prepare_data.sh train test"
fi
