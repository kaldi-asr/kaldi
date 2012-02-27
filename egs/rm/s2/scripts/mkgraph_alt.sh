#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# This version of mkgraph.sh creates the C fst explicitly.

reorder=true # Dan-style, make false for Mirko+Lukas's decoder.

for x in 1 2 3; do 
  if [ $1 == "--mono" ]; then
    monophone_opts="--context-size=1 --central-position=0"
    shift;
  fi

  if [ $1 == "--noreorder" ]; then 
    reorder=false # we set this for the Kaldi decoder.
    shift;
  fi
done

if [ $# != 3 ]; then
   echo "Usage: scripts/mkgraph.sh  <tree> <model> <graphdir>"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi


tree=$1
model=$2
dir=$3

mkdir -p $dir

tscale=1.0
loopscale=0.1

fsttablecompose data/L_disambig.fst data/G.fst | fstdeterminizestar --use-log=true | \
  fstminimizeencoded  > $dir/LG.fst

fstisstochastic $dir/LG.fst || echo "warning: LG not stochastic."

echo "Example string from LG.fst: "
echo 
fstrandgen --select=log_prob $dir/LG.fst | fstprint --isymbols=data/phones_disambig.txt --osymbols=data/words.txt -

grep '#' data/phones_disambig.txt | awk '{print $2}' > $dir/disambig_phones.list
subseq_sym=`tail -1 data/phones_disambig.txt | awk '{print $2+1;}'`
cp data/phones_disambig.txt $dir/phones_disambig_subseq.txt
echo '$' $subseq_sym >> $dir/phones_disambig_subseq.txt

fstmakecontextfst --read-disambig-syms=$dir/disambig_phones.list \
 --write-disambig-syms=$dir/disambig_ilabels.list data/phones.txt $subseq_sym \
   $dir/ilabels | fstarcsort --sort_type=olabel > $dir/C.fst

fstaddsubsequentialloop $subseq_sym $dir/LG.fst | \
 fsttablecompose $dir/C.fst - > $dir/CLG.fst


 # for debugging:
 fstmakecontextsyms data/phones.txt $dir/ilabels > $dir/context_syms.txt
 echo "Example string from CLG.fst: "
 echo 
 fstrandgen --select=log_prob $dir/CLG.fst | fstprint --isymbols=$dir/context_syms.txt --osymbols=data/words.txt -

fstisstochastic $dir/CLG.fst || echo "warning: CLG not stochastic."

make-ilabel-transducer --write-disambig-syms=$dir/disambig_ilabels_remapped.list $dir/ilabels $tree $model $dir/ilabels.remapped > $dir/ilabel_map.fst

# Reduce size of CLG by remapping symbols...
fstcompose $dir/ilabel_map.fst $dir/CLG.fst  | fstdeterminizestar --use-log=true \
  | fstminimizeencoded > $dir/CLG2.fst


cat $dir/CLG2.fst |  fstisstochastic  || echo "warning: CLG2 is not stochastic."

make-h-transducer --disambig-syms-out=$dir/disambig_tstate.list \
  --transition-scale=$tscale $dir/ilabels.remapped $tree $model > $dir/Ha.fst


fsttablecompose $dir/Ha.fst $dir/CLG2.fst | fstdeterminizestar --use-log=true \
 | fstrmsymbols $dir/disambig_tstate.list | fstrmepslocal | fstminimizeencoded > $dir/HCLGa.fst

fstisstochastic $dir/HCLGa.fst || echo "HCLGa is not stochastic"

add-self-loops --self-loop-scale=$loopscale --reorder=$reorder $model < $dir/HCLGa.fst > $dir/HCLG.fst

if [ $tscale == 1.0 -a $loopscale == 1.0 ]; then
  # No point doing this test if transition-scale not 1, as it is bound to fail. 
  fstisstochastic $dir/HCLG.fst || echo "Final HCLG is not stochastic."
fi

fstisstochastic $dir/HCLG.fst || echo "Final HCLG is not stochastic."


#The next five lines are debug.
# The last two lines of this block print out some alignment info.
fstrandgen --select=log_prob $dir/HCLG.fst |  fstprint --osymbols=data/words.txt > $dir/rand.txt
cat $dir/rand.txt | awk 'BEGIN{printf("0  ");} {if(NF>=3 && $3 != 0){ printf ("%d ",$3); }} END {print ""; }' > $dir/rand_align.txt
show-alignments data/phones.txt $model ark:$dir/rand_align.txt
cat $dir/rand.txt | awk ' {if(NF>=4 && $4 != "<eps>"){ printf ("%s ",$4); }} END {print ""; }'

