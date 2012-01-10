#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation  Arnab Ghoshal

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


# tri2a is another epoch of triphone training, starting from the system in tri1.
# The data subset (3500 utterances from si-84) is the same as in tri1, which
# means we can re-use its graphs.  Starting with a smallish number of Gaussians
# so we can use this same setup when training transforms and it will be
# reasonably efficient.
# See tri3a for a full-size (si-284) system that builds from this.

if [ -f path.sh ]; then . path.sh; fi

dir=exp/reduce_pdf_count
srcdir=exp/tri3a
srcmodel=$srcdir/final.mdl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

numleaves=500 # <= PDF count (reducing from original 2000 PDFs)

numgauss=2000 # initial num-gauss smallish so that transform-training
              # code (when we modify this script) is a bit faster.

silphonelist=`cat data/silphones.csl`

mkdir -p $dir
cp $srcdir/train.scp $dir
cp $srcdir/train.tra $dir

# Make graph + align on 3 cpus.  
scripts/split_scp.pl $dir/train.scp  $dir/train{1,2,3}.scp
scripts/split_scp.pl $dir/train.tra  $dir/train{1,2,3}.tra

# also see featspart below, used for sub-parts of the features;
# try to keep them in sync.
feats="ark:add-deltas --print-args=false scp:$dir/train.scp ark:- |"
for n in 1 2 3; do
   featspart[$n]="ark:add-deltas --print-args=false scp:$dir/train${n}.scp ark:- |"
done

cp $srcdir/topo $dir

# Align all training data using old model (and old graphs, since we
# use the same data-subset as last time). 

echo "Aligning all training data"

rm -f $dir/.error
for n in 1 2 3; do
   gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $srcmodel \
       "ark:gunzip -c $srcdir/graphs${n}.fsts.gz|" "${featspart[$n]}" \
       "ark:|gzip -c >$dir/0.${n}.ali.gz" \
           2> $dir/align.0.${n}.log || touch $dir/.error &
done
wait;
[ -f $dir/.error ] &&  echo align error RE old system && exit 1

( 
 # Put a few graphs in human-readable form
 # for easier debugging.
  mkdir -p $dir/graph_egs
  n=5
  head -$n $dir/train.tra | awk '{printf("%s '$dir'/graph_egs/%s.fst\n", $1, $1); }' > $dir/some_graphs.scp
  compile-train-graphs $srcdir/tree $srcmodel  data/L.fst "ark:head -$n $dir/train.tra|" \
    "scp:$dir/some_graphs.scp"  2>$dir/compile_some_graphs.log || exit 1 
  for filename in `cat $dir/some_graphs.scp | awk '{print $2;}'`; do
     fstprint --osymbols=data/words.txt $filename > $filename.txt
  done
)


acc-tree-stats  --ci-phones=$silphonelist $srcmodel "$feats" "ark:gunzip -c $dir/0.?.ali.gz|" $dir/treeacc 2> $dir/acc.tree.log  || exit 1;


# The next few commands are involved with making the questions
# for tree clustering.  The extra complexity vs. the RM recipe has
# to do with the desire to ask questions about the "real" phones
# ignoring things like stress and position-in-word, and ask questions
# separately about stress and position-in-word.

# Don't include silences as things to be clustered -> --nosil option.
scripts/make_shared_phones.sh --nosil | scripts/sym2int.pl data/phones.txt > $dir/phone_sets.list
cluster-phones $dir/treeacc $dir/phone_sets.list $dir/questions.txt 2> $dir/cluster_phones.log || exit 1;
scripts/int2sym.pl data/phones.txt < $dir/questions.txt > $dir/questions_syms.txt
scripts/make_extra_questions.sh | cat $dir/questions_syms.txt - > $dir/questions_syms_all.txt
scripts/sym2int.pl data/phones.txt < $dir/questions_syms_all.txt > $dir/questions_all.txt

compile-questions $dir/topo $dir/questions_all.txt $dir/questions.qst 2>$dir/compile_questions.log || exit 1;

scripts/make_roots.sh > $dir/roots_syms.txt
scripts/sym2int.pl --ignore-oov data/phones.txt  < $dir/roots_syms.txt > $dir/roots.txt


build-tree --verbose=1 --max-leaves=$numleaves \
    $dir/treeacc $dir/roots.txt \
    $dir/questions.qst $dir/topo $dir/tree  2> $dir/train_tree.log || exit 1;

gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $dir/topo $dir/1.mdl 2> $dir/init_model.log || exit 1;

gmm-mixup --mix-up=$numgauss $dir/1.mdl $dir/1.occs $dir/1.mdl \
    2>$dir/mixup.log || exit 1;


rm $dir/treeacc $dir/1.occs


# Convert alignments generated from previous model, to use as initial alignments.
echo "Converting alignments"
for n in 1 2 3; do
  convert-ali  $srcmodel $dir/1.mdl $dir/tree \
      "ark:gunzip -c $dir/0.$n.ali.gz|" \
      "ark:|gzip -c > $dir/cur$n.ali.gz" 2>$dir/convert.$n.log || exit 1;
done
rm $dir/0.?.ali.gz

echo "Alignments converted..."
 
( cd $dir; rm final.mdl 2>/dev/null; ln -s 1.mdl final.mdl )
