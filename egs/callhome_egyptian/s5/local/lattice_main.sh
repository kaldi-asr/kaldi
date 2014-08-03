#!/usr/bin/env bash

outDir=exp/lat
mkdir -p $outDir

stage=1

if [ $stage -lt 1 ]; then

  # First convert all lattices into the pruned, minimized version
  decodeDir=exp/tri5a/decode_dev
  acousticScale=0.8333
  local/latconvert.sh $outDir $decodeDir $acousticScale

  decodeDir=exp/tri5a/decode_test
  acousticScale=0.8333
  local/latconvert.sh $outDir $decodeDir $acousticScale

fi

if [ $stage -lt 2 ]; then
  # Get oracles
  latticeDir=exp/tri5a/decode_dev
  textFile=data/dev/text
  symTable=exp/tri5a/graph/words.txt
  local/get_oracle.sh $latticeDir $symTable $textFile

  latticeDir=exp/tri5a/decode_test
  textFile=data/test/text
  symTable=exp/tri5a/graph/words.txt
  local/get_oracle.sh $latticeDir $symTable $textFile
fi
