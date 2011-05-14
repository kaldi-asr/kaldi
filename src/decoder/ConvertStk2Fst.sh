#!/bin/bash
#usage: ConvertStk2Fst STKnetwork kaldiMMF

#OpenFST paths
export PATH=~../../../openfst-1.2/bin/:$PATH
export LD_LIBRARY_PATH=~../../../openfst-1.2/lib:$LD_LIBRARY_PATH

echo "text file conversion"
./KaldiNet2WfstRight.sh $1
./KaldiModelNumbers.sh $2
echo "compile binary fst"
fstcompile --isymbols=$2.sym --osymbols=reconet.words --keep_isymbols --keep_osymbols reconet.txt reconet.fst