#!/bin/bash 
tmpdir=data/local/tmp

. ./path.sh || exit 1;

mkdir -p data/lang_ug
cp \
    -vr \
    data/lang/* \
    data/lang_ug

rm -vrf data/lang_ug/tmp

echo "computing costs"
local/compute_ug_costs.pl < \
			  data/train_semi_supervised/text > \
			  $tmpdir/costs.txt || exit 1;

echo "compiling fst"
fstcompile \
    --isymbols=data/lang/words.txt \
    --osymbols=data/lang/words.txt \
    --keep_isymbols=false \
    --keep_osymbols=false < \
    $tmpdir/costs.txt > \
    $tmpdir/ug_compiled.fst || exit 1;

echo "sorting fst arcs"
fstarcsort \
    --sort_type=ilabel > \
    data/lang_ug/G.fst < \
    $tmpdir/ug_compiled.fst || exit 1;

echo "Checking that G is stochastic [note, it wouldn't be for an Arpa]"
fstisstochastic \
    data/lang_ug/G.fst || echo Error: G is not stochastic

echo "Checking that G.fst is determinizable."
fstdeterminize \
    data/lang_ug/G.fst \
    /dev/null || echo Error determinizing G.

echo " Checking that L_disambig.fst is determinizable."
fstdeterminize \
    data/lang_ug/L_disambig.fst \
    /dev/null || echo Error determinizing L.

echo " Checking that disambiguated lexicon times G is determinizable"
fsttablecompose \
    data/lang_ug/L_disambig.fst \
    data/lang_ug/G.fst | \
    fstdeterminize > \
		   /dev/null || echo Error

echo " Checking that LG is stochastic:"
fsttablecompose \
    data/lang_ug/L.fst \
    data/lang_ug/G.fst | \
    fstisstochastic || echo Error: LG is not stochastic.

echo " Checking that L_disambig.G is stochastic:"
fsttablecompose \
    data/lang_ug/L_disambig.fst \
    data/lang_ug/G.fst | \
    fstisstochastic || echo Error: LG is not stochastic.
