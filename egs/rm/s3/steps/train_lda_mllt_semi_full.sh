#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation  Arnab Ghoshal
#                     Univ. Erlangen-Nuremberg  Korbinian Riedhammer

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

# To be run from ..
# Triphone model training, using delta-delta features and cepstral
# mean normalization.  It starts from an existing directory (e.g.
# exp/mono), supplied as an argument, which is assumed to be built using
# the same type of features.

if [[ $# != 9 ]]; then
   echo "Usage: steps/train_lda_mllt_semi_full.sh <data-dir> <lang-dir> <ali-dir> <ubm> <exp-dir> <num-tree-leaves> <smooth-type> <tau> <rho>"
   echo " e.g.: steps/train_lda_mllt_semi_full.sh data/train data/lang exp/tri2b_ali exp/ubm3f/final.ubm exp/tiedfull3f 2500 1 35 0.2"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
lang=$2
alidir=$3
ubm=$4
dir=$5
max_leaves=$6   # target num-leaves in tree building.
smooth_type=$7  # (0, regular Interpolate1) (1, preserve-counts, Interpolate2)
tau=$8          # set to > 0 to activate prop/interp of suff. stats. (weights only)
rho=$9          # set to > 0 to actiavte smoothing of new model with prior 
                # model (weights only)

if [ ! -f $ubm ]; then
  echo "No such file $ubm"
  exit 1;
fi

if [ ! -f $alidir/final.mdl -o ! -f $alidir/ali -o ! -f $alidir/final.mat ]; then
  echo "Error: alignment dir $alidir does not contain final.mdl and ali and final.mat"
  exit 1;
fi

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
realign_iters="5 10 15 20";  
silphonelist=`cat $lang/silphones.csl`
numiters=25     # Number of iterations of training


inter=""           # will be filled, if parameter smoothing was requested
preserve_counts="" # will be filled, if preserve-counts


# inter-iteration smoothing?
if [ "$rho" != "0" ]; then
  inter="--interpolation-weight=$rho --interpolate-weights"
fi

# preserve counts for intra-interation smoothing?
if [ "$smooth_type" == "0" ]; then
  preserve_counts=""
elif [ "$smooth_type" == "1" ]; then
  preserve_counts="--preserve-counts"
else
  echo "Invalid smoothing type $smooth_type"
  exit 1
fi

mkdir -p $dir

# write out the params we're using, just for the record
echo "Training started: `date`
$0 $@
max_leaves=$max_leaves
smooth_type=$smooth_type
tau=$tau
rho=$rho
intra=$intra
preserve_counts=$preserve_counts" > $dir/.params

cp $alidir/final.mat $dir/

# regular features, with mean normalization
feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$alidir/cmvn.ark scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"

echo "Accumulating tree stats"
acc-tree-stats  --ci-phones=$silphonelist $alidir/final.mdl "$feats" \
   ark:$alidir/ali $dir/treeacc 2> $dir/acc.tree.log  || exit 1;

echo "Computing questions for tree clustering"
cat $lang/phones.txt | awk '{print $NF}' | grep -v -w 0 > $dir/phones.list
cluster-phones $dir/treeacc $dir/phones.list $dir/questions.txt 2> $dir/questions.log || exit 1;
scripts/int2sym.pl $lang/phones.txt < $dir/questions.txt > $dir/questions_syms.txt
compile-questions $lang/topo $dir/questions.txt $dir/questions.qst 2>$dir/compile_questions.log || exit 1;

# Have to make silence root not-shared because we will not split it.
scripts/make_roots.pl --separate $lang/phones.txt $silphonelist shared split \
    > $dir/roots.txt 2>$dir/roots.log || exit 1;

# build the tree, but disable the post-clustering of the leaves by setting --cluster-thresh=0
echo "Building tree"
build-tree --verbose=1 --cluster-thresh=0 \
    --max-leaves=$max_leaves \
    $dir/treeacc $dir/roots.txt \
    $dir/questions.qst $lang/topo $dir/tree 2> $dir/train_tree.log || exit 1;

# generate dummy tree.map to map all leaves to the single codebook
echo -n "[ " > $dir/tree.map
for i in `seq 1 $max_leaves`; do
  echo -n "0 " >> $dir/tree.map
done
echo "]" >> $dir/tree.map

echo "Initializing model"
tied-full-gmm-init-model $dir/tree $lang/topo $ubm $dir/1.mdl 2> $dir/init_model.log || exit 1;

rm $dir/treeacc

# Convert alignments generated from cont/triphone model, to use as initial alignments.

convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree ark:$alidir/ali ark:$dir/cur.ali 2>$dir/convert.log 
  # Debug step only: convert back and check they're the same.
  convert-ali $dir/1.mdl $alidir/final.mdl $alidir/tree ark:$dir/cur.ali ark:- \
   2>/dev/null | cmp - $alidir/ali || exit 1; 

# Make training graphs
echo "Compiling training graphs"
compile-train-graphs $dir/tree $dir/1.mdl  $lang/L.fst \
  "ark:scripts/sym2int.pl --ignore-first-field $lang/words.txt < $data/text |" \
   "ark:|gzip -c >$dir/graphs.fsts.gz"  2>$dir/compile_graphs.log  || exit 1;

# Make gselect objects (for speed)

( gmm-gselect --n=50 "fgmm-global-to-gmm $ubm - |" "$feats" ark:- | \
  fgmm-gselect --n=10 --gselect=ark:- $ubm "$feats" \
  "ark,t:|gzip -c >$dir/gselect.gz" )  2>$dir/gselect.log || exit 1;

gselect_opt="--gselect=ark,s,cs:gunzip -c $dir/gselect.gz|"

x=1
while [ $x -lt $numiters ]; do
   echo Pass $x
   if echo $realign_iters | grep -w $x >/dev/null; then
     echo "Aligning data"
     tied-full-gmm-align-compiled $scale_opts --beam=8 \
       --retry-beam=40 $dir/$x.mdl "$gselect_opt" \
       "ark:gunzip -c $dir/graphs.fsts.gz|" "$feats" \
        ark:$dir/cur.ali 2> $dir/align.$x.log || exit 1;
   fi

   tied-full-gmm-acc-stats-ali "$gselect_opt" --binary=false \
       $dir/$x.mdl "$feats" ark:$dir/cur.ali $dir/$x.acc \
     2> $dir/acc.$x.log  || exit 1;

   # suff. stats smoothing?
   if [ "$tau" != "0" ]; then
     smooth-stats-full $preserve_counts --tau=$tau $dir/tree $dir/tree.map $dir/$x.acc $dir/$x.acc.tmp \
       2> $dir/smooth.$x.err > $dir/smooth.$x.out || exit 1;
     mv $dir/$x.acc.tmp $dir/$x.acc
   fi

   tied-full-gmm-est $inter --write-occs=$dir/$[$x+1].occs $dir/$x.mdl $dir/$x.acc $dir/$[$x+1].mdl \
     2> $dir/update.$x.log || exit 1;
   
   rm $dir/$x.mdl $dir/$x.acc
   rm $dir/$x.occs
   x=$[$x+1];
done

( cd $dir; rm final.mdl 2>/dev/null; ln -s $x.mdl final.mdl; ln -s $x.occs final.occs )

echo Done
