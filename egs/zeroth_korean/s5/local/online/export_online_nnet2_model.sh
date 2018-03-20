#!/bin/bash

# Copyright 2017 Lucas Jo (Atlas Guide)
# Apache 2.0

if [ $# -ne "1" ]; then
	echo "Usage: $0 <export_dir>"
	echo "e.g.: $0 ./export"
	exit 1
fi

tardir=$1
srcdir=exp/nnet2_online/nnet_ms_a_online
graphdir=exp/tri5b/graph_tgsmall
oldlang=data/lang_test_tgsmall
newlang=data/lang_test_fglarge
oldlm=$oldlang/G.fst
newlm=$newlang/G.carpa
symtab=$newlang/words.txt

for f in $srcdir/final.mdl $symtab $graphdir/HCLG.fst $srcdir/conf/mfcc.conf \
	$srcdir/conf/ivector_extractor.conf $oldlm $newlm; do
	[ ! -f $f ] && echo "export_model.sh: no such file $f" && exit 1;
done

mkdir -p $tardir/conf
cp -rpf $srcdir/final.mdl $tardir/final.mdl	# acoustic  model
cp -rpf $symtab $tardir/words.txt			# word symbol table
cp -rpf $graphdir/HCLG.fst $tardir/HCLG.fst	# HCLG 
cp -rpf $srcdir/conf/mfcc.conf $tardir/conf/mfcc.conf
cp -rpf $srcdir/conf/ivector_extractor.conf $tardir/conf/ivector_extractor.conf
cp -rpf $oldlm $tardir/G.fst
cp -rpf $newlm $tardir/G.carpa
