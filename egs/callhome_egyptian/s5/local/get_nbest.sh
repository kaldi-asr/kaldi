#!/usr/bin/env bash
# Author : Gaurav Kumar, Johns Hopkins University 
# Creates n-best lists from Kaldi lattices
# This script needs to be run from one level above this directory

. path.sh

if [ $# -lt 3 ]; then
  echo "Enter the latdir (where the n-best will be put), the decode dir containing lattices and the acoustic scale"
  exit 1
fi

noNBest=100
maxProcesses=10

latdir=$1
decode_dir=$2
acoustic_scale=$3
partition=$4
symTable=exp/tri5a/graph/words.txt
scriptSymTable=data/local/dict/lexicon_script.txt

stage=0

if [ -d $decode_dir ]
then
  allNBest=$latdir/$partition.all.nbest
  runningProcesses=0

  for l in $decode_dir/lat.*.gz
  do	
    (
    # Extract file name and unzip the file first
    bname=${l##*/}
    bname="$latdir/$partition.${bname%.gz}"
    gunzip -c $l > "$bname.bin"

    if [ $stage -le 0 ]; then

      # Extract n-best from the lattices
      lattice-to-nbest --acoustic-scale=$acoustic_scale --n=$noNBest \
        ark:$bname.bin ark:$bname.nbest

      #Convert the n-best lattice to linear word based sentences
      nbest-to-linear ark,t:$bname.nbest ark,t:$bname.ali ark,t:$bname.words \
        ark,t:$bname.lmscore ark,t:$bname.acscore

      #Convert the int to word for each sentence
      cat $bname.words | utils/int2sym.pl -f 2- \
        $symTable >> $allNBest.roman
      #Use the script version on the symbol table too
      cat $bname.words | utils/int2sym.pl -f 2- \
        $scriptSymTable >> $allNBest.script
    fi

    echo "Done getting n-best"
    ) &	
    runningProcesses=$((runningProcesses+1))
    echo "#### Processes running = " $runningProcesses " ####"
    if [ $runningProcesses -eq $maxProcesses ]; then
      echo "#### Waiting for slot ####"
      wait
      runningProcesses=0
      echo "#### Done waiting ####"
    fi
  done
  wait
fi
