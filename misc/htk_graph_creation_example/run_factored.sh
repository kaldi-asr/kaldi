# This is a recipe involving factoring, where we make the topology linear (remove
# the extra transitions from silence; this is an option to the conversion script.

rootdir=../..
D=`dirname "$rootdir"`
B=`basename "$rootdir"`
rootdir="`cd \"$D\" 2>/dev/null && pwd || echo \"$D\"`/$B"

workdir=convert_factored

../htk_conversion/convert_htk.sh --linear-topology $rootdir /mnt/matylda5/jhu09/setup/CH1/English/exp/xwrd.R0_800_TB500/hmm164/MMF /mnt/matylda5/jhu09/setup/CH1/English/exp/xwrd.R0_800_TB500/hmm10_800_500/cluster.trees $workdir

cd $workdir
export LC_ALL=C
(echo "<s>    sil";  echo "</s>    sil"; ) |
cat /homes/eva/q/qthomas/Workshop/Interp/callhome_gigaword_switchboard_web_64k/dict_callhome_gigaword_switchboard_web - \
 > lexicon.txt
echo "<eps> 0" > words.txt
cat lexicon.txt | awk '{print $1}' | sort | uniq | awk '{printf("%s %d\n", $1, NR);}' >> words.txt

# Make lexicon fst.

$rootdir/rm_recipe_2/scripts/make_lexicon_fst.pl lexicon.txt 0.5 sil  | fstcompile --isymbols=phones.txt --osymbols=words.txt --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > L.fst


$rootdir/src/lm/arpa2fst /homes/eva/q/qthomas/Workshop/Interp/callhome_gigaword_switchboard_web_64k/lm_callhome_gigaword_switchboard_web_2gram.arpa | fstprint | fstcompile --isymbols=words.txt --osymbols=words.txt --keep_isymbols=false --keep_osymbols=false > G.fst

# Randomly generating string, as follows, for test:
# fstrandgen --select=log_prob  G.fst | fstprint --isymbols=words.txt --osymbols=words.txt

# Doing the same with the lexicon involved:
# fstrandgen --select=log_prob  G.fst | fstcompose L.fst - | fstrandgen | fstprint --isymbols=phones.txt --osymbols=words.txt


reorder=true # Dan-style, make false for Mirko+Lukas's decoder.
  # Actually, not applicable for this model (has no effect)-- due to the
  # ergodic silence.
tscale=0.1 # transition-prob scale.

export PATH=$PATH:$rootdir/src/fstbin/:$rootdir/src/bin/
fsttablecompose L.fst G.fst | fstoptimize > LG.fst

fstcomposecontext ilabels < LG.fst >CLG.fst

 # for debugging:
 fstmakecontextsyms phones.txt ilabels > context_syms.txt
 echo "Example string from CLG.fst: "
 echo 
 fstrandgen --select=log_prob CLG.fst | fstprint --isymbols=context_syms.txt --osymbols=words.txt -


 $rootdir/src/bin/make-h-transducer --symbol-type=transition-states ilabels kaldi.tree kaldi.mdl > Ha.fst
 fsttablecompose Ha.fst CLG.fst | fstoptimize > HCLGa.fst
 fstfactor HCLGa.fst HCLGa1.fst HCLG2.fst

 $rootdir/src/bin/add-self-loops --transition-scale=$tscale --reorder=$reorder kaldi.mdl < HCLGa1.fst > HCLG1.fst

 fsttablecompose HCLG1.fst HCLG2.fst > HCLG.fst

