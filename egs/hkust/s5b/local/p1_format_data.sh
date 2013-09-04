#!/bin/bash 
#

if  [ $# -ne 3 ]
then
  echo $0 input_lang_dir output_lang_test_dir gzip_arpa_lm
  exit
fi

if [ -f path.sh ]; then . path.sh; fi

indir=$1
outdir=$2

silprob=0.5
mkdir -p ${outdir} data/train


arpa_lm=$3
[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;

cp -r ${indir}/* ${outdir}

# grep -v '<s> <s>' etc. is only for future-proofing this script.  Our
# LM doesn't have these "invalid combinations".  These can cause 
# determinization failures of CLG [ends up being epsilon cycles].
# Note: remove_oovs.pl takes a list of words in the LM that aren't in
# our word list.  Since our LM doesn't have any, we just give it
# /dev/null [we leave it in the script to show how you'd do it].
gunzip -c "$arpa_lm" | \
   grep -v '<s> <s>' | \
   grep -v '</s> <s>' | \
   grep -v '</s> </s>' | \
   arpa2fst - | fstprint | \
   utils/remove_oovs.pl /dev/null | \
   utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=${outdir}/words.txt \
     --osymbols=${outdir}/words.txt  --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon > ${outdir}/G.fst
  fstisstochastic ${outdir}/G.fst


echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic ${outdir}/G.fst 

## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
fstprint   --isymbols=${indir}/phones.txt --osymbols=${indir}/words.txt ${indir}/L.fst  | head

echo Performing further checks

# Checking that G.fst is determinizable.
fstdeterminize ${outdir}/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize ${outdir}/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
# Note: we do this with fstdeterminizestar not fstdeterminize, as
# fstdeterminize was taking forever (presumbaly relates to a bug
# in this version of OpenFst that makes determinization slow for
# some case).
fsttablecompose ${outdir}/L_disambig.fst ${outdir}/G.fst | \
   fstdeterminizestar >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose ${indir}/L_disambig.fst ${outdir}/G.fst | \
   fstisstochastic || echo LG is not stochastic


echo p1_format_data succeeded.

