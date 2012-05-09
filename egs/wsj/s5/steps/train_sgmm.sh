#!/bin/bash

# Copyright 2012  Daniel Povey.  Apache 2.0.

# SGMM training, with speaker vectors.  This script would normally be called on
# top of fMLLR features obtained from a conventional system, but it also works
# on top of any type of speaker-independent features (based on
# deltas+delta-deltas or LDA+MLLT).  For more info on SGMMs, see the paper "The
# subspace Gaussian mixture model--A structured model for speech recognition".
# (Computer Speech and Language, 2011).

# Begin configuration section.
nj=4
cmd=scripts/run.pl
stage=-5
context_opts= # e.g. set it to "--context-width=5 --central-position=2"  for a
# quinphone system.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
num_iters=25   # Total number of iterations
num_iters_alimdl=3 # Number of iterations for estimating alignment model.
maxiterinc=15 # Last iter to increase #substates on.
realign_iters="5 10 15"; # Iters to realign on. 
spkvec_iters="5 8 12 17" # Iters to estimate speaker vectors on.
add_dim_iters="6 8 10 12"; # Iters on which to increase phn dim and/or spk dim,
rand_prune=0.1 # Randomized-pruning parameter for posteriors, to speed up training.
phn_dim=  # You can use this to set the phonetic subspace dim. [default: feat-dim+1]
spk_dim=  # You can use this to set the speaker subspace dim. [default: feat-dim]


# End configuration section.

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 7 ]; then
  echo "Usage: steps/train_sgmm.sh <num-leaves> <num-substates> <data> <lang> <ali-dir> <ubm-dir> <exp-dir>"
  echo " e.g.: steps/train_sgmm.sh 3500 10000 data/train_si84 data/lang \\"
  echo "                      exp/tri3b_ali_si84 exp/ubm4a/final.ubm exp/sgmm4a"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --silence-weight <sil-weight>                    # weight for silence (e.g. 0.5 or 0.0)"
  echo "  --num-iters <#iters>                             # Number of iterations of E-M"
  exit 1;
fi


num_leaves=$1
totsubstates=$2
data=$3
lang=$4
alidir=$5
ubm=$6
dir=$7

# Check some files.
for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl $ubm/final.ubm; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done



cp $transform_dir/final.mat $dir/final.mat || exit 1;

# Set some variables.
oov=`cat $lang/oov.int`
silphonelist=`cat $lang/phones/silence.csl`
numsubstates=$num_leaves # Initial #-substates.
incsubstates=$[($totsubstates-$numsubstates)/$maxiterinc] # per-iter increment for #substates
feat_dim=`gmm-info $alidir/final.model | awk '/feature dimension/{print $NF}'` || exit 1;
[ $feat_dim -eq $feat_dim ] || exit 1; # make sure it's numeric.
[ -z $phn_dim ] && phn_dim=$[$feat_dim+1]
[ -z $spk_dim ] && spk_dim=$feat_dim
nj=`cat $alidir/num_jobs` || exit 1;

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;


for n in `get_splits.pl $nj`; do
  # Initially don't have speaker vectors, but change this after we estimate them.
  spkvecs_opt[$n]=
  gselect_opt[$n]="--gselect=ark,s,cs:gunzip -c $dir/$n.gselect.gz|"
done


n1=`get_splits.pl $nj | awk '{print $1}'`
[ -f $transform_dir/$n1.trans ] && echo "Using speaker transforms from $transform_dir"

for n in `get_splits.pl $nj`; do
  featspart[$n]="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$transform_dir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
  if [ -f $transform_dir/$n1.trans ]; then
    featspart[$n]="${featspart[$n]} transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$transform_dir/$n.trans ark:- ark:- |"
  fi
done


if [ ! -f $ubm ]; then
  echo "No UBM in $ubm"
  exit 1;
fi


if [ $stage -le -5 ]; then
  # This stage assumes we won't need the context of silence, which
  # assumes something about $lang/roots.txt, but it seems pretty safe.
  echo "Accumulating tree stats"
  rm $dir/.error 2>/dev/null
  for n in `get_splits.pl $nj`; do
    $cmd $dir/log/acc_tree.$n.log \
    acc-tree-stats  $context_opts --ci-phones=$silphonelist $alidir/final.mdl "${featspart[$n]}" \
      "ark:gunzip -c $alidir/$n.ali.gz|" $dir/$n.treeacc || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo Error accumulating tree stats && exit 1;
  sum-tree-stats $dir/treeacc $dir/*.treeacc 2>$dir/log/sum_tree_acc.log || exit 1;
  rm $dir/*.treeacc
fi

if [ $stage -le -4 ]; then
  echo "Computing questions for tree clustering"
  # preparing questions, roots file...
  sym2int.pl $lang/phones.txt $lang/phonesets_cluster.txt > $dir/phonesets.txt || exit 1;
  cluster-phones $context_opts $dir/treeacc $dir/phonesets.txt $dir/questions.txt 2> $dir/log/questions.log || exit 1;
  sym2int.pl $lang/phones.txt $lang/extra_questions.txt >> $dir/questions.txt
  compile-questions $context_opts $lang/topo $dir/questions.txt $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;
  sym2int.pl --ignore-oov $lang/phones.txt $lang/roots.txt > $dir/roots.txt

  echo "Building tree"
  $cmd $dir/log/train_tree.log \
    build-tree $context_opts --verbose=1 --max-leaves=$num_leaves \
      $dir/treeacc $dir/roots.txt \
      $dir/questions.qst $lang/topo $dir/tree || exit 1;

  # The next line is a bit of a hack to work out the feature dim.  The program
  # feat-to-len returns the #rows of each matrix, which for the transform matrix,
  # is the feature dim.
  featdim=`feat-to-len "scp:echo foo $transform_dir/final.mat|" ark,t:- 2>/dev/null | awk '{print $2}'`
  
  # Note: if phn_dim and/or spk_dim are higher than you can initialize with,
  # sgmm-init will just make them as high as it can (later we'll increase)

  $cmd $dir/log/init_sgmm.log \
    sgmm-init --phn-space-dim=$phn_dim --spk-space-dim=$spk_dim $lang/topo $dir/tree $ubm \
      $dir/0.mdl || exit 1;

fi

rm $dir/.error 2>/dev/null

if [ $stage -le -3 ]; then
  echo "Doing Gaussian selection"
  for n in `get_splits.pl $nj`; do
    $cmd $dir/log/gselect$n.log \
      sgmm-gselect $dir/0.mdl "${featspart[$n]}" "ark,t:|gzip -c > $dir/$n.gselect.gz" \
     || touch $dir/.error &
  done
  wait;
  [ -f $dir/.error ] && echo "Error doing Gaussian selection" && exit 1;
fi

if [ $stage -le -2 ]; then
  echo "Compiling training graphs"
  for n in `get_splits.pl $nj`; do
    $cmd $dir/log/compile_graphs$n.log \
      compile-train-graphs $dir/tree $dir/0.mdl  $lang/L.fst  \
       "ark:sym2int.pl --map-oov $oov --ignore-first-field $lang/words.txt < $data/split$nj/$n/text |" \
       "ark:|gzip -c >$dir/$n.fsts.gz" || touch $dir/.error &
  done
  wait;
  [ -f $dir/.error ] && echo "Error compiling training graphs" && exit 1;
fi


if [ $stage -le -1 ]; then
  echo "Converting alignments"  # don't bother parallelizing; very fast.
  for n in `get_splits.pl $nj`; do
    convert-ali $alidir/final.mdl $dir/0.mdl $dir/tree "ark:gunzip -c $alidir/$n.ali.gz|" \
       "ark:|gzip -c >$dir/$n.ali.gz" 2>$dir/log/convert$n.log 
  done
fi

x=0
while [ $x -lt $num_iters ]; do
   echo "Pass $x ... "
   if echo $realign_iters | grep -w $x >/dev/null; then
      if [ $stage -le $x ]; then
        echo "Aligning data"
        for n in `get_splits.pl $nj`; do
          $cmd $dir/log/align.$x.$n.log  \
            sgmm-align-compiled ${spkvecs_opt[$n]} $scale_opts "${gselect_opt[$n]}" \
               --utt2spk=ark:$data/split$nj/$n/utt2spk --beam=8 --retry-beam=40 \
               $dir/$x.mdl "ark:gunzip -c $dir/$n.fsts.gz|" "${featspart[$n]}" \
               "ark:|gzip -c >$dir/$n.ali.gz" || touch $dir/.error &
        done
        wait;
        [ -f $dir/.error ] && echo "Error realigning data on iter $x" && exit 1;
      fi
   fi
   if [ $spk_dim -gt 0 ] && echo $spkvec_iters | grep -w $x >/dev/null; then
     for n in `get_splits.pl $nj`; do
       if [ $stage -le $x ]; then
         $cmd $dir/log/spkvecs.$x.$n.log \
           ali-to-post "ark:gunzip -c $dir/$n.ali.gz|" ark:- \| \
             weight-silence-post 0.01 $silphonelist $dir/$x.mdl ark:- ark:- \| \
             sgmm-est-spkvecs --spk2utt=ark:$data/split$nj/$n/spk2utt \
               ${spkvecs_opt[$n]} "${gselect_opt[$n]}" $dir/$x.mdl \
            "${featspart[$n]}" ark,s,cs:- ark:$dir/tmp$n.vecs  \
           && mv $dir/tmp$n.vecs $dir/$n.vecs || touch $dir/.error &
       fi
       spkvecs_opt[$n]="--spk-vecs=ark:$dir/$n.vecs"
     done
     wait;
     [ -f $dir/.error ] && echo "Error computing speaker vectors on iter $x" && exit 1;     
   fi  
   if [ $x -eq 0 ]; then
     flags=vwcSt # On first iter, don't update M or N.
   elif [ $spk_dim -gt 0 -a $[$x%2] -eq 1 -a $x -ge `echo $spkvec_iters | awk '{print $1}'` ]; then 
     # Update N if we have spk-space and x is odd, and we're at least at 1st spkvec iter.
     flags=vNwcSt
   else # Else update M but not N.
     flags=vMwcSt
   fi

   if [ $stage -le $x ]; then
     for n in `get_splits.pl $nj`; do
       $cmd $dir/log/acc.$x.$n.log \
         sgmm-acc-stats ${spkvecs_opt[$n]} --utt2spk=ark:$data/split$nj/$n/utt2spk \
           --update-flags=$flags "${gselect_opt[$n]}" --rand-prune=$randprune \
           $dir/$x.mdl "${featspart[$n]}" "ark,s,cs:ali-to-post 'ark:gunzip -c $dir/$n.ali.gz|' ark:-|" \
           $dir/$x.$n.acc || touch $dir/.error &
     done
     wait;
     [ -f $dir/.error ] && echo "Error accumulating stats on iter $x" && exit 1;     
   fi

   add_dim_opts=
   if echo $add_dim_iters | grep -w $x >/dev/null; then
     add_dim_opts="--increase-phn-dim=$phn_dim --increase-spk-dim=$spk_dim"
   fi

   if [ $stage -le $x ]; then
     $cmd $dir/log/update.$x.log \
       sgmm-est --update-flags=$flags --split-substates=$numsubstates $add_dim_opts \
         --write-occs=$dir/$[$x+1].occs $dir/$x.mdl "sgmm-sum-accs - $dir/$x.*.acc|" \
       $dir/$[$x+1].mdl || exit 1;

     rm $dir/$x.mdl $dir/$x.*.acc
     rm $dir/$x.occs 
   fi
   if [ $x -lt $maxiterinc ]; then
     numsubstates=$[$numsubstates+$incsubstates]
   fi
   x=$[$x+1];
done

( cd $dir; rm final.mdl final.occs 2>/dev/null; 
  ln -s $x.mdl final.mdl; 
  ln -s $x.occs final.occs )

if [ $spk_dim -gt 0 ]; then
  # If we have speaker vectors, we need an alignment model.
  # The point of this last phase of accumulation is to get Gaussian-level
  # alignments with the speaker vectors but accumulate stats without
  # any speaker vectors; we re-estimate M, w, c and S to get a model
  # that's compatible with not having speaker vectors.

  # We do this for a few iters, in this recipe.
  cur_alimdl=$dir/$x.mdl
  y=0;
  while [ $y -lt $num_iters_alimdl ]; do
    echo "Pass $y of building alignment model"
    if [ $y -eq 0 ]; then
      flags=MwcS # First time don't update v...
    else
      flags=vMwcS # don't update transitions-- will probably share graph with normal model.
    fi
    if [ $stage -le $[$y+100] ]; then
      for n in `get_splits.pl $nj`; do
        $cmd $dir/log/acc_ali.$y.$n.log \
          ali-to-post "ark:gunzip -c $dir/$n.ali.gz|" ark:- \| \
            sgmm-post-to-gpost ${spkvecs_opt[$n]} "${gselect_opt[$n]}" \
              --utt2spk=ark:$data/split$nj/$n/utt2spk $dir/$x.mdl "${featspart[$n]}" ark,s,cs:- ark:- \| \
            sgmm-acc-stats-gpost --update-flags=$flags  $cur_alimdl "${featspart[$n]}" \
              ark,s,cs:- $dir/$y.$n.aliacc || touch $dir/.error &
      done
      wait;
      [ -f $dir/.error ] && echo "Error accumulating stats for alignment model on iter $y" && exit 1;
      $cmd $dir/log/update_ali.$y.log \
         sgmm-est --update-flags=$flags --remove-speaker-space=true $cur_alimdl \
         "sgmm-sum-accs - $dir/$y.*.aliacc|" $dir/$[$y+1].alimdl || exit 1;
      rm $dir/$y.*.aliacc || exit 1;
      [ $y -gt 0 ]  && rm $dir/$y.alimdl
    fi
    cur_alimdl=$dir/$[$y+1].alimdl
    y=$[$y+1]
  done
  (cd $dir; rm final.alimdl 2>/dev/null; ln -s $y.alimdl final.alimdl )
fi


# Print out summary of the warning messages.
for x in $dir/log/*.log; do 
  n=`grep WARNING $x | wc -l`; 
  if [ $n -ne 0 ]; then echo $n warnings in $x; fi; 
done

echo Done
