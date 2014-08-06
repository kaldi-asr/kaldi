#!/usr/bin/env bash

outDir=exp/lat
mkdir -p $outDir

stage=2

if [ $stage -lt 1 ]; then

  # First convert all lattices into the pruned, minimized version
  decodeDir=exp/tri5a/decode_dev
  acousticScale=0.08333
  local/latconvert.sh $outDir $decodeDir $acousticScale

  decodeDir=exp/tri5a/decode_test
  acousticScale=0.08333
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

if [ $stage -lt 3 ]; then
  # Create a script lexicon if it does not exist
  if [ ! -f data/local/dict/lexicon_script.txt ]; then
    local/callhome_prepare_script_dict.py /export/corpora/LDC/LDC99L22/ \
      exp/tri5a/graph/words.txt data/local/dict/lexicon_script.txt
  fi

  # Now get the n-best files from the lattices
  decodeDir=exp/tri5a/decode_dev
  acousticScale=0.08333
  local/get_nbest.sh $outDir $decodeDir $acousticScale dev

  decodeDir=exp/tri5a/decode_test
  acousticScale=0.08333
  local/get_nbest.sh $outDir $decodeDir $acousticScale test
fi
