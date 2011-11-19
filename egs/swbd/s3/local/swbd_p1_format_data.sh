#!/bin/bash 
#

if [ -f path.sh ]; then . path.sh; fi

silprob=0.5
for x in lang lang_test train; do
  mkdir -p data/$x
done

arpa_lm=data/local/lm/3gram-mincount/lm_unpruned.gz

# Copy stuff into its final location:



for x in train; do 
  cp data/local/$x.spk2utt data/$x/spk2utt || exit 1;
  cp data/local/$x.utt2spk data/$x/utt2spk || exit 1;
  # Don't call it wav.scp because that's reserved for the wav file
  # that's one per utterance.
  cp data/local/${x}_wav.scp data/$x/wav_sides.scp || exit 1;
  cp data/local/${x}.txt data/$x/text || exit 1;
  cp data/local/segments* data/$x || exit 1;
done


cp data/local/words.txt data/lang/words.txt
cp data/local/phones.txt data/lang/phones.txt

silphones="SIL SPN NSN LAU"; # This would in general be a space-separated list of 
     # all silence/nonspeech phones. 
     # Generate colon-separated lists of silence and non-silence phones.
scripts/silphones.pl data/lang/phones.txt "$silphones" data/lang/silphones.csl \
  data/lang/nonsilphones.csl
# we map OOVs in the training data to this symbol.
echo "[VOCALIZED-NOISE]" > data/lang/oov.txt

# For monophone systems and tree roots-- share the roots of the different
# position-dependent versions of the "real" phones, and also  the different
# forms of silence.  This is all as for WSJ, except we have one more silence
# phone, and we introduce one more state into the prototype silence model
# in conf/topo.proto, to give it more parameters to model laughter.
local/make_shared_phones.sh < data/lang/phones.txt > data/lang/phonesets_mono.txt
grep -v -w SIL data/lang/phonesets_mono.txt > data/lang/phonesets_cluster.txt
local/make_extra_questions.sh < data/lang/phones.txt > data/lang/extra_questions.txt

( # Creating the "roots file" for building the context-dependent systems...
  # we share the roots across all the versions of each real phone.  We also
  # share the states of the 3 forms of silence.  "not-shared" here means the
  # states are distinct p.d.f.'s... normally we would automatically split on
  # the HMM-state but we're not making silences context dependent.
  echo 'not-shared not-split SIL SPN NSN LAU';
  cat data/lang/phones.txt | grep -v -E 'eps|SIL|SPN|NSN|LAU' | awk '{print $1}' | \
    perl -e 'while(<>){ m:([A-Za-z]+)(\d*)(_.)?: || die "Bad line $_"; 
            $phone=$1; $stress=$2; $position=$3;
      if($phone eq $curphone){ print " $phone$stress$position"; }
      else { if(defined $curphone){ print "\n"; } $curphone=$phone; 
            print "shared split $phone$stress$position";  }} print "\n"; '
) > data/lang/roots.txt


silphonelist=`cat data/lang/silphones.csl | sed 's/:/ /g'`
nonsilphonelist=`cat data/lang/nonsilphones.csl | sed 's/:/ /g'`
cat conf/topo.proto | sed "s:NONSILENCEPHONES:$nonsilphonelist:" | \
   sed "s:SILENCEPHONES:$silphonelist:" > data/lang/topo

ndisambig=`scripts/add_lex_disambig.pl data/local/lexicon.txt data/local/lexicon_disambig.txt`
ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST.
scripts/add_disambig.pl --include-zero data/lang/phones.txt $ndisambig > data/lang_test/phones_disambig.txt


scripts/make_lexicon_fst.pl data/local/lexicon.txt $silprob SIL  | \
  fstcompile --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt \
   --keep_isymbols=false --keep_osymbols=false | \
   fstarcsort --sort_type=olabel > data/lang/L.fst

for x in topo L.fst words.txt phones.txt silphones.csl nonsilphones.csl; do
  cp data/lang/$x data/lang_test
done



# Create L_align.fst, which is as L.fst but with alignment symbols (#1 and #2 at the
# beginning and end of words, on the input side)... needed to discover the 
# word boundaries in alignments, when we need to create ctm-format output.

cat data/local/lexicon.txt | \
 awk '{printf("%s #1 ", $1); for (n=2; n <= NF; n++) { printf("%s ", $n); } print "#2"; }' | \
 scripts/make_lexicon_fst.pl - $silprob SIL | \
 fstcompile --isymbols=data/lang_test/phones_disambig.txt --osymbols=data/lang_test/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
 fstarcsort --sort_type=olabel > data/lang_test/L_align.fst


# Make lexicon with disambiguation symbols.  We need to
# add self-loops to "pass through" the #0 symbol from the 
# backoff language model.
phone_disambig_symbol=`grep \#0 data/lang_test/phones_disambig.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 data/lang_test/words.txt | awk '{print $2}'`

scripts/make_lexicon_fst.pl data/local/lexicon_disambig.txt $silprob SIL '#'$ndisambig | \
   fstcompile --isymbols=data/lang_test/phones_disambig.txt --osymbols=data/lang_test/words.txt \
   --keep_isymbols=false --keep_osymbols=false | \
   fstaddselfloops  "echo $phone_disambig_symbol |" "echo $word_disambig_symbol |" | \
   fstarcsort --sort_type=olabel \
    > data/lang_test/L_disambig.fst

# Copy into data/lang/ also, where it will be needed for discriminative training.
cp data/lang_test/L_disambig.fst data/lang/


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
   scripts/remove_oovs.pl /dev/null | \
   scripts/eps2disambig.pl | scripts/s2eps.pl | fstcompile --isymbols=data/lang_test/words.txt \
     --osymbols=data/lang_test/words.txt  --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon > data/lang_test/G.fst
  fstisstochastic data/lang_test/G.fst



echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic data/lang_test/G.fst 

## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
fstprint   --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt data/lang/L.fst  | head

echo Performing further checks

# Checking that G.fst is determinizable.
fstdeterminize data/lang_test/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize data/lang_test/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
# Note: we do this with fstdeterminizestar not fstdeterminize, as
# fstdeterminize was taking forever (presumbaly relates to a bug
# in this version of OpenFst that makes determinization slow for
# some case).
fsttablecompose data/lang_test/L_disambig.fst data/lang_test/G.fst | \
   fstdeterminizestar >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose data/lang/L.fst data/lang_test/G.fst | \
   fstisstochastic || echo Error: LG is not stochastic.


echo swbd_p1_format_data succeeded.

