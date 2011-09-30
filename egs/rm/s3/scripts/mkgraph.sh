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


N=3
P=1
clean=false

for x in 1 2 3; do 
  if [ $1 == "--mono" ]; then
    N=1;
    P=0;
    shift;
  fi
  if [ $1 == "--clean" ]; then
    clean=true
    shift;
  fi

done

if [ $# != 3 ]; then
   echo "Usage: scripts/mkgraph.sh <test-lang-dir> <model-dir> <graphdir>"
   echo "e.g.: scripts/mkgraph.sh data/lang_test exp/tri1/ exp/tri1/graph"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

lang=$1
tree=$2/tree
model=$2/final.mdl
dir=$3

if $clean; then rm -r $lang/tmp; fi

mkdir -p $dir

tscale=1.0
loopscale=0.1

# If $lang/tmp/LG.fst does not exist or is older than its sources, make it...
# (note: the [[ ]] brackets make the || type operators work (inside [ ], we
# would have to use -o instead),  -f means file exists, and -ot means older than).

mkdir -p $lang/tmp
if [[ ! -f $lang/tmp/LG.fst || $lang/tmp/LG.fst -ot $lang/G.fst || \
      $lang/tmp/LG.fst -ot $lang/L_disambig.fst ]]; then
  fsttablecompose $lang/L_disambig.fst $lang/G.fst | fstdeterminizestar --use-log=true | \
    fstminimizeencoded  > $lang/tmp/LG.fst || exit 1;
  fstisstochastic $lang/tmp/LG.fst || echo "warning: LG not stochastic."
fi

if [ ! -f $lang/phones_disambig.txt ]; then
  echo "No such file $lang/phones_disambig.txt (supplied a training lang/ directory?)"
  exit 1;
fi

grep '#' $lang/phones_disambig.txt | awk '{print $2}' > $lang/tmp/disambig_phones.list


clg=$lang/tmp/CLG_${N}_${P}.fst

if [[ ! -f $clg || $clg -ot $lang/tmp/LG.fst ]]; then
  fstcomposecontext --context-size=$N --central-position=$P \
   --read-disambig-syms=$lang/tmp/disambig_phones.list \
   --write-disambig-syms=$lang/tmp/disambig_ilabels_${N}_${P}.list \
    $lang/tmp/ilabels_${N}_${P} < $lang/tmp/LG.fst >$clg
  fstisstochastic $clg  || echo "warning: CLG not stochastic."
fi

if [[ ! -f $dir/Ha.fst || $dir/Ha.fst -ot $model ]]; then
  make-h-transducer --disambig-syms-out=$dir/disambig_tid.list \
    --transition-scale=$tscale $lang/tmp/ilabels_${N}_${P} $tree $model \
     > $dir/Ha.fst  || exit 1;
fi

if [[ ! -f $dir/HCLGa.fst || $dir/HCLGa.fst -ot $dir/Ha.fst || \
      $dir/HCLGa.fst -ot $clg ]]; then
  fsttablecompose $dir/Ha.fst $clg | fstdeterminizestar --use-log=true \
    | fstrmsymbols $dir/disambig_tid.list | fstrmepslocal | \
     fstminimizeencoded > $dir/HCLGa.fst || exit 1;
  fstisstochastic $dir/HCLGa.fst || echo "HCLGa is not stochastic"
fi

if [[ ! -f $dir/HCLG.fst || $dir/HCLG.fst -ot $dir/HCLGa.fst ]]; then
  add-self-loops --self-loop-scale=$loopscale --reorder=true \
    $model < $dir/HCLGa.fst > $dir/HCLG.fst || exit 1;

  if [ $tscale == 1.0 -a $loopscale == 1.0 ]; then
    # No point doing this test if transition-scale not 1, as it is bound to fail. 
    fstisstochastic $dir/HCLG.fst || echo "Final HCLG is not stochastic."
  fi
fi


# to make const fst:
# fstconvert --fst_type=const $dir/HCLG.fst $dir/HCLG_c.fst

