#!/usr/bin/env bash

# Gets lattice oracles
# Copyright 2014  Gaurav Kumar.   Apache 2.0

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

cat $textFile | sed 's:\[laughter\]::g' | sed 's:\[noise\]::g' | \
    utils/sym2int.pl --map-oov [oov] -f 2- $symTable | \
    $KALDI_ROOT/src/latbin/lattice-oracle --word-symbol-table=$symTable "ark:gunzip -c $latticeDir/lat.*.gz|" ark:- ark,t:$oracleDir/oracle.tra 2>$oracleDir/oracle.log

sort -k1,1 -u $oracleDir/oracle.tra -o $oracleDir/oracle.tra
