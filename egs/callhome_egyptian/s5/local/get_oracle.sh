#!/usr/bin/env bash

# Gets lattice oracles

if [ $# -lt 3 ]; then
    echo "Specify lattice dir, symbol table and text file for partition"
    exit 1;
fi

latticeDir=$1
textFile=$3
symTable=$2
oracleDir=$latticeDir/oracle

echo $latticeDir
echo $oracleDir

. path.sh

if [ ! -f $textFile -o ! -f $symTable -o ! -d $latticeDir ]; then
    echo "Required files not found"
    exit 1;
fi

mkdir -p $oracleDir

# Since the lexicon is built from the LDC lexicon, there are words in the dataset
# that do not appear in the lexicon. These have to marked as OOV. 
# Removing [hes] symbols as well. This is not consistent with the scoring scheme used
# while scoring 1-best. 
cat $textFile | sed 's:\[laughter\]::g' | sed 's:\[noise\]::g' | sed 's:\[hes\]::g' | \
    utils/sym2int.pl --map-oov [oov] -f 2- $symTable | \
    $KALDI_ROOT/src/latbin/lattice-oracle --word-symbol-table=$symTable "ark:gunzip -c $latticeDir/lat.*.gz|" ark:- ark,t:$oracleDir/oracle.tra 2>$oracleDir/oracle.log

sort -k1,1 -u $oracleDir/oracle.tra -o $oracleDir/oracle.tra
