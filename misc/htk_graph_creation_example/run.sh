# **THIS SCRIPT IS NOT WORKING YET-- wait a few hours**

# This is a fairly simple recipe for model conversion and graph creation.

rootdir=../..
D=`dirname "$rootdir"`
B=`basename "$rootdir"`
rootdir="`cd \"$D\" 2>/dev/null && pwd || echo \"$D\"`/$B"
export PATH=$PATH:$rootdir/tools/openfst/bin/:$rootdir/src/fstbin/:$rootdir/src/lm/:$rootdir/egs/wsj/s1/scripts/:$rootdir/src/bin/

workdir=convert_basic

../htk_conversion/convert_htk.sh $rootdir /mnt/matylda5/jhu09/setup/CH1/English/exp/xwrd.R0_800_TB500/hmm164/MMF /mnt/matylda5/jhu09/setup/CH1/English/exp/xwrd.R0_800_TB500/hmm10_800_500/cluster.trees $workdir
# the above creates kaldi.tree and kaldi.mdl

cd $workdir
export LC_ALL=C
(echo "<s>    sil";  echo "</s>    sil"; ) |
cat /homes/eva/q/qthomas/Workshop/Interp/callhome_gigaword_switchboard_web_64k/dict_callhome_gigaword_switchboard_web - \
 > lexicon.txt


# Make the words symbol-table; add the disambiguation symbol #0 (we use this in place of epsilon
# in the grammar FST).
cat lexicon.txt | awk '{print $1}' | sort | uniq  | \
 awk 'BEGIN{print "<eps> 0";} {printf("%s %d\n", $1, NR);} END{printf("#0 %d\n", NR+1);} ' \
  > words.txt

# Make lexicon fst.


make_lexicon_fst.pl lexicon.txt 0.5 sil  | fstcompile --isymbols=phones.txt --osymbols=words.txt --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > L.fst

ndisambig=`add_lex_disambig.pl lexicon.txt lexicon_disambig.txt`
echo $ndisambig > lex_ndisambig
# Next, create a phones.txt file that includes the disambiguation symbols.
# the --include-zero includes the #0 symbol we pass through from the grammar.
add_disambig.pl --include-zero phones.txt $ndisambig > phones_disambig.txt

phone_disambig_symbol=`grep \#0 phones_disambig.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 words.txt | awk '{print $2}'`

make_lexicon_fst.pl lexicon_disambig.txt 0.5 sil  | \
   fstcompile --isymbols=phones_disambig.txt --osymbols=words.txt \
   --keep_isymbols=false --keep_osymbols=false |   \
   fstaddselfloops  "echo $phone_disambig_symbol |" "echo $word_disambig_symbol |" | \
   fstarcsort --sort_type=olabel > L_disambig.fst

# Now make the LM FST.
cat /homes/eva/q/qthomas/Workshop/Interp/callhome_gigaword_switchboard_web_64k/lm_callhome_gigaword_switchboard_web_2gram.arpa | \
    grep -v '<s> <s>' | \
    grep -v '</s> <s>' | \
    grep -v '</s> </s>' | \
  arpa2fst - | fstprint | \
  eps2disambig.pl | fstcompile --isymbols=words.txt --osymbols=words.txt \
     --keep_isymbols=false --keep_osymbols=false > G.fst
# For diagnostics...
fstisstochastic G.fst

# Randomly generating string, as follows, for test:
# fstrandgen --select=log_prob  G.fst | fstprint --isymbols=words.txt --osymbols=words.txt

# Doing the same with the lexicon involved:
# fstrandgen --select=log_prob  G.fst | fstcompose L_disambig.fst - | fstrandgen | fstprint --isymbols=phones_disambig.txt --osymbols=words.txt


reorder=true # Dan-style, make false for Mirko+Lukas's decoder.
  # Actually, not applicable for this model (has no effect)-- due to the
  # ergodic silence.

loopscale=0.1
tscale=1.0


fsttablecompose L_disambig.fst G.fst | fstdeterminizestar --use-log=true | \
    fstminimizeencoded > LG.fst

fstisstochastic LG.fst # diagnostic.

fstcomposecontext ilabels < LG.fst >CLG.fst

  echo "Example string from LG.fst: "
  echo
  fstrandgen --select=log_prob LG.fst | fstprint --isymbols=phones_disambig.txt --osymbols=words.txt -

grep '#' phones_disambig.txt | awk '{print $2}' > disambig_phones.list

fstcomposecontext \
    --read-disambig-syms=disambig_phones.list \
    --write-disambig-syms=disambig_ilabels.list \
    ilabels < LG.fst >CLG.fst

 # for debugging:                                                                                                                              
    fstmakecontextsyms phones.txt ilabels > context_syms.txt
    echo "Example string from CLG.fst: "
    echo
    fstrandgen --select=log_prob CLG.fst | fstprint --isymbols=context_syms.txt --osymbols=words.txt -


# diagnostic.
fstisstochastic CLG.fst



make-ilabel-transducer --write-disambig-syms=disambig_ilabels_remapped.list \
  ilabels kaldi.tree kaldi.mdl ilabels.remapped > ilabel_map.fst

# Reduce size of CLG by remapping symbols...                                                                                                   
fstcompose ilabel_map.fst CLG.fst  | fstdeterminizestar --use-log=true \
  | fstminimizeencoded > CLG2.fst

make-h-transducer --disambig-syms-out=disambig_tstate.list \
   --transition-scale=$tscale  ilabels.remapped kaldi.tree kaldi.mdl > Ha.fst

fsttablecompose Ha.fst CLG2.fst | fstdeterminizestar --use-log=true \
 | fstrmsymbols disambig_tstate.list | fstrmepslocal  | fstminimizeencoded > HCLGa.fst

# Diagnostic
fstisstochastic HCLGa.fst

add-self-loops --self-loop-scale=$loopscale --reorder=$reorder kaldi.mdl < HCLGa.fst > HCLG.fst

#The next five lines are debug.                                                                                                                
# The last two lines of this block print out some alignment info.                                                                              
fstrandgen --select=log_prob HCLG.fst |  fstprint --osymbols=words.txt > rand.txt
cat rand.txt | awk 'BEGIN{printf("0  ");} {if(NF>=3 && $3 != 0){ printf ("%d ",$3); }} END {print ""; }' > rand_align.txt
show-alignments phones.txt kaldi.mdl ark:rand_align.txt
cat rand.txt | awk ' {if(NF>=4 && $4 != "<eps>"){ printf ("%s ",$4); }} END {print ""; }'
