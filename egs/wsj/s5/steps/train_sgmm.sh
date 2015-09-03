#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# SGMM training, with speaker vectors.  This script would normally be called on
# top of fMLLR features obtained from a conventional system, but it also works
# on top of any type of speaker-independent features (based on
# deltas+delta-deltas or LDA+MLLT).  For more info on SGMMs, see the paper "The
# subspace Gaussian mixture model--A structured model for speech recognition".
# (Computer Speech and Language, 2011).

# Begin configuration section.
nj=4
cmd=run.pl
stage=-6
context_opts= # e.g. set it to "--context-width=5 --central-position=2"  for a
# quinphone system.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
num_iters=25   # Total number of iterations
num_iters_alimdl=3 # Number of iterations for estimating alignment model.
max_iter_inc=15 # Last iter to increase #substates on.
realign_iters="5 10 15"; # Iters to realign on. 
spkvec_iters="5 8 12 17" # Iters to estimate speaker vectors on.
increase_dim_iters="6 8"; # Iters on which to increase phn dim and/or spk dim;
   # rarely necessary, and if it is, only the 1st will normally be necessary.
rand_prune=0.1 # Randomized-pruning parameter for posteriors, to speed up training.
phn_dim=  # You can use this to set the phonetic subspace dim. [default: feat-dim+1]
spk_dim=  # You can use this to set the speaker subspace dim. [default: feat-dim]
power=0.25 # Exponent for number of gaussians according to occurrence counts
beam=8
retry_beam=40
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 7 ]; then
  echo "Usage: steps/train_sgmm.sh <num-leaves> <num-substates> <data> <lang> <ali-dir> <ubm> <exp-dir>"
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
for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl $ubm; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


# Set some variables.
oov=`cat $lang/oov.int`
silphonelist=`cat $lang/phones/silence.csl`
numsubstates=$num_leaves # Initial #-substates.
incsubstates=$[($totsubstates-$numsubstates)/$max_iter_inc] # per-iter increment for #substates
feat_dim=`gmm-info $alidir/final.mdl 2>/dev/null | awk '/feature dimension/{print $NF}'` || exit 1;
[ $feat_dim -eq $feat_dim ] || exit 1; # make sure it's numeric.
[ -z $phn_dim ] && phn_dim=$[$feat_dim+1]
[ -z $spk_dim ] && spk_dim=$feat_dim
nj=`cat $alidir/num_jobs` || exit 1;
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
cp $alidir/splice_opts $dir 2>/dev/null
cp $alidir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

spkvecs_opt=  # Empty option for now, until we estimate the speaker vectors.
gselect_opt="--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|"

## Set up features.
if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ -f $alidir/trans.1 ]; then
  echo "$0: using transforms from $alidir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$alidir/trans.JOB ark:- ark:- |"
fi
##


if [ $stage -le -6 ]; then
  echo "$0: accumulating tree stats"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
    acc-tree-stats  --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
    "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1;
  [ "`ls $dir/*.treeacc | wc -w`" -ne "$nj" ] && echo "$0: Wrong #tree-stats" && exit 1;
  sum-tree-stats $dir/treeacc $dir/*.treeacc 2>$dir/log/sum_tree_acc.log || exit 1;
  rm $dir/*.treeacc
fi

if [ $stage -le -5 ]; then
  echo "$0: Getting questions for tree clustering."
  # preparing questions, roots file...
  cluster-phones $dir/treeacc $lang/phones/sets.int $dir/questions.int 2> $dir/log/questions.log || exit 1;
  cat $lang/phones/extra_questions.int >> $dir/questions.int
  compile-questions $lang/topo $dir/questions.int $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;

  echo "$0: Building the tree"
  $cmd $dir/log/build_tree.log \
    build-tree --verbose=1 --max-leaves=$num_leaves \
    --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
    $dir/questions.qst $lang/topo $dir/tree || exit 1;
fi

if [ $stage -le -4 ]; then
  echo "$0: Initializing the model"  
  # Note: if phn_dim > feat_dim+1 or spk_dim > feat_dim, these dims
  # will be truncated on initialization.
  $cmd $dir/log/init_sgmm.log \
    sgmm-init --phn-space-dim=$phn_dim --spk-space-dim=$spk_dim $lang/topo \
    $dir/tree $ubm $dir/0.mdl || exit 1;
fi

if [ $stage -le -3 ]; then
  echo "$0: doing Gaussian selection"
  $cmd JOB=1:$nj $dir/log/gselect.JOB.log \
    sgmm-gselect $dir/0.mdl "$feats" \
    "ark,t:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
fi

if [ $stage -le -2 ]; then
  echo "$0: compiling training graphs"
  text="ark:sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata/JOB/text|"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs $dir/tree $dir/0.mdl  $lang/L.fst  \
    "$text" "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

if [ $stage -le -1 ]; then
  echo "$0: Converting alignments" 
  $cmd JOB=1:$nj $dir/log/convert_ali.JOB.log \
    convert-ali $alidir/final.mdl $dir/0.mdl $dir/tree "ark:gunzip -c $alidir/ali.JOB.gz|" \
    "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

x=0
while [ $x -lt $num_iters ]; do
   echo "$0: training pass $x ... "
   if echo $realign_iters | grep -w $x >/dev/null && [ $stage -le $x ]; then
     echo "$0: re-aligning data"
     $cmd JOB=1:$nj $dir/log/align.$x.JOB.log  \
       sgmm-align-compiled $spkvecs_opt $scale_opts "$gselect_opt" \
       --utt2spk=ark:$sdata/JOB/utt2spk --beam=$beam --retry-beam=$retry_beam \
       $dir/$x.mdl "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
       "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
   fi
   if [ $spk_dim -gt 0 ] && echo $spkvec_iters | grep -w $x >/dev/null; then
     if [ $stage -le $x ]; then
       $cmd JOB=1:$nj $dir/log/spkvecs.$x.JOB.log \
         ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:- \| \
         weight-silence-post 0.01 $silphonelist $dir/$x.mdl ark:- ark:- \| \
         sgmm-est-spkvecs --rand-prune=$rand_prune --spk2utt=ark:$sdata/JOB/spk2utt \
         $spkvecs_opt "$gselect_opt" $dir/$x.mdl "$feats" ark,s,cs:- \
         ark:$dir/tmp_vecs.JOB '&&' mv $dir/tmp_vecs.JOB $dir/vecs.JOB || exit 1;
     fi
     spkvecs_opt[$n]="--spk-vecs=ark:$dir/vecs.JOB"
   fi  
   if [ $x -eq 0 ]; then
     flags=vwcSt # on the first iteration, don't update projections M or N
   elif [ $spk_dim -gt 0 -a $[$x%2] -eq 1 -a $x -ge `echo $spkvec_iters | awk '{print $1}'` ]; then 
     # Update N if we have speaker-vector space and x is odd,
     # and we've already updated the speaker vectors...
     flags=vNwcSt
   else
     # otherwise update M.
     flags=vMwcSt
   fi
   
   if [ $stage -le $x ]; then
     $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
       sgmm-acc-stats $spkvecs_opt --utt2spk=ark:$sdata/JOB/utt2spk \
       --update-flags=$flags "$gselect_opt" --rand-prune=$rand_prune \
       $dir/$x.mdl "$feats" "ark,s,cs:gunzip -c $dir/ali.JOB.gz | ali-to-post ark:- ark:-|" \
       $dir/$x.JOB.acc || exit 1;
   fi

   # The next option is needed if the user specifies a phone or speaker sub-space
   # dimension that's higher than the "normal" one.
   increase_dim_opts=
   if echo $increase_dim_iters | grep -w $x >/dev/null; then
     increase_dim_opts="--increase-phn-dim=$phn_dim --increase-spk-dim=$spk_dim"
     # Note: the command below might have a null effect on some iterations.
     if [ $spk_dim -gt $feat_dim ]; then 
       cmd JOB=1:$nj $dir/log/copy_vecs.$x.JOB.log \
         copy-vector --print-args=false --change-dim=$spk_dim \
         ark:$dir/vecs.JOB ark:$dir/vecs_tmp.$JOB '&&' \
         mv $dir/vecs_tmp.JOB $dir/vecs.JOB || exit 1;
     fi
   fi

   if [ $stage -le $x ]; then
     $cmd $dir/log/update.$x.log \
       sgmm-est --update-flags=$flags --split-substates=$numsubstates $increase_dim_opts \
         --power=$power --write-occs=$dir/$[$x+1].occs $dir/$x.mdl "sgmm-sum-accs - $dir/$x.*.acc|" \
       $dir/$[$x+1].mdl || exit 1;
     rm $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs 2>/dev/null
   fi
   
   if [ $x -lt $max_iter_inc ]; then
     numsubstates=$[$numsubstates+$incsubstates]
   fi
   x=$[$x+1];
done

rm $dir/final.mdl $dir/final.occs 2>/dev/null
ln -s $x.mdl $dir/final.mdl
ln -s $x.occs $dir/final.occs

if [ $spk_dim -gt 0 ]; then
  # We need to create an "alignment model" that's been trained
  # without the speaker vectors, to do the first-pass decoding with.
  # in test time.

  # We do this for a few iters, in this recipe.
  final_mdl=$dir/$x.mdl
  cur_alimdl=$dir/$x.mdl
  while [ $x -lt $[$num_iters+$num_iters_alimdl] ]; do
    echo "$0: building alignment model (pass $x)"
    if [ $x -eq $num_iters ]; then # 1st pass of building alimdl.
      flags=MwcS # don't update v the first time.  Note-- we never update transitions.
      # they wouldn't change anyway as we use the same alignment as previously.
    else
      flags=vMwcS
    fi
    if [ $stage -le $x ]; then
      $cmd JOB=1:$nj $dir/log/acc_ali.$x.JOB.log \
        ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:- \| \
        sgmm-post-to-gpost $spkvecs_opt "$gselect_opt" \
        --utt2spk=ark:$sdata/JOB/utt2spk $final_mdl "$feats" ark,s,cs:- ark:- \| \
        sgmm-acc-stats-gpost --rand-prune=$rand_prune --update-flags=$flags \
          $cur_alimdl "$feats" ark,s,cs:- $dir/$x.JOB.aliacc || exit 1;
      $cmd $dir/log/update_ali.$x.log \
        sgmm-est --update-flags=$flags --remove-speaker-space=true --power=$power $cur_alimdl \
        "sgmm-sum-accs - $dir/$x.*.aliacc|" $dir/$[$x+1].alimdl || exit 1;
      rm $dir/$x.*.aliacc || exit 1;
      [ $x -gt $num_iters ]  && rm $dir/$x.alimdl
    fi
    cur_alimdl=$dir/$[$x+1].alimdl
    x=$[$x+1]
  done
  rm $dir/final.alimdl 2>/dev/null 
  ln -s $x.alimdl $dir/final.alimdl
fi

utils/summarize_warnings.pl $dir/log

echo Done
