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

# To be run from ..
# This does mixing-up, three iterations of model training, realignment,
# and two more iterations of model training.
# It's intended to be used for experiments with LDA+MLLT or LDA+MLLT+SAT
# models where you increase the number of mixtures and see if it helps.

nj=4
cmd=scripts/run.pl
stage=0
increasephonedim=0


for x in 1 2; do
  if [ "$1" == "--num-jobs" ]; then
    shift
    nj=$1
    shift
  fi
  if [ "$1" == "--increase-phone-dim" ]; then
    shift
    increasephonedim=$1
    shift
  fi
  if [ "$1" == "--cmd" ]; then
    shift
    cmd=$1
    shift
  fi  
  if [ $1 == "--stage" ]; then # stage to start training from, typically same as the iter you have a .mdl file;
     shift                     # in case it failed part-way.  
     stage=$1
     shift
  fi  
done

if [ $# != 5 ]; then
   echo "Usage: steps/mixup_sgmm_lda_etc.sh <num-substates> <data-dir> <old-exp-dir> <alignment-dir> <exp-dir>"
   echo "Note: <alignment-dir> is provided so we can get the CMVN and transform data from there."
   echo " e.g.: steps/mixup_sgmm_lda_etc.sh 35000 data/train_si284 exp/sgmm4c exp/tri3b_ali_si284_20 exp/sgmm4c_35k"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

numsubstates=$1
data=$2
olddir=$3
alidir=$4 # only needed for CMVN and possibly transform data.
dir=$5

for f in $data/feats.scp $olddir/final.mdl $olddir/final.mat; do
  [ ! -f $f ] && echo "mixup_lda_etc.sh: no such file $f" && exit 1;
done

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
randprune=0.1
numiters_alimdl=2 # Number of iterations for estimating alignment model. [note,
   # it's 3 in the script train_sgmm_lda_etc.sh, but a look at the objf impr's
   # suggests this was overkill.

mkdir -p $dir/log
cp $olddir/final.mat $olddir/tree $dir/

if [ ! -d $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  echo "Splitting data-dir $data into $nj pieces, but watch out: we require #jobs"  \
      "to be matched with $olddir"
  split_data.sh $data $nj
fi

for n in `get_splits.pl $nj`; do
  sifeatspart[$n]="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
  featspart[$n]="${sifeatspart[$n]}"
done

# Adjust the features to reflect any transforms we may have in $alidir.
first=`get_splits.pl $nj | awk '{print $1}'`
if [ -f $alidir/$first.trans ]; then
  echo Using transforms in $alidir
  for n in `get_splits.pl $nj`; do
    featspart[$n]="${sifeatspart[$n]} transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.trans ark:- ark:- |"
  done
else
  echo "No transforms in $alidir, assuming you are not using fMLLR."
fi


echo Mixing up old model to $numsubstates sub-states
[ $increasephonedim -gt 0 ] && echo "[and increasing phone subspace dimension to $increasephonedim]"

$cmd $dir/log/mixup.log \
 sgmm-mixup --increase-phn-dim=$increasephonedim --read-occs=$olddir/final.occs \
   --split-substates=$numsubstates $olddir/final.mdl $dir/0.mdl || exit 1;

rm $dir/.error 2>/dev/null

if [ -f $olddir/$first.vecs ]; then
  have_vecs=true
  ln.pl $olddir/*.vecs $dir  # Link speaker vectors
  echo "Using speaker vectors from $olddir"
  for n in `get_splits.pl $nj`; do spkvecs_opt[$n]="--spk-vecs=ark:$dir/$n.vecs"; done
else
  have_vecs=false
  echo "Not using speaker vectors as not present in $olddir"
fi

for n in `get_splits.pl $nj`; do 
  gselect_opt[$n]="--gselect=ark,s,cs:gunzip -c $dir/$n.gselect.gz|";
done

ln.pl $olddir/*.fsts.gz $dir  # Link FSTs.
ln.pl $olddir/*.gselect.gz $dir # Link gselect info.

dir_for_alignments=$olddir # This is where we find the alignments...
      # after we realign, on iter 3, we'll use the ones in $dir
niters=4
for x in `seq 0 $niters`; do  # Do five iterations of E-M; on 3rd iter, realign.
  echo Iteration $x
  if [ $stage -le $x ]; then
    if [ $x -eq 2 ]; then
      echo Realigning data on iteration $x
      for n in `get_splits.pl $nj`; do
        [ ! -f $dir/$n.fsts.gz ] && echo Expecting FSTs to exist: no such file $dir/$n.fsts.gz \
          && exit 1;
        $cmd $dir/log/align.$x.$n.log \
          sgmm-align-compiled ${spkvecs_opt[$n]} $scale_opts "${gselect_opt[$n]}" \
            --utt2spk=ark:$data/split$nj/$n/utt2spk --beam=8 --retry-beam=40 $dir/$x.mdl \
            "ark:gunzip -c $dir/$n.fsts.gz|" "${featspart[$n]}" \
            "ark:|gzip -c >$dir/$n.ali.gz" || touch $dir/.error &
      done
      wait
      [ -f $dir/.error ] && echo "Error computing alignments" && exit 1;
      dir_for_alignments=$dir
    fi

    if $have_vecs && [ $[$x%2] -eq 1 ]; then  # Update N if we have spk-space and x is odd
      # Note: $have_vecs is "true" or "false", which is why we put it outside the "test"
      # expression..
      flags=vNwcSt
    else # Else update M but not N.
      flags=vMwcSt
    fi
    echo "Accumulating statistics"
    for n in `get_splits.pl $nj`; do  
       $cmd $dir/log/acc.$x.$n.log \
        sgmm-acc-stats-ali ${spkvecs_opt[$n]} --utt2spk=ark:$data/split$nj/$n/utt2spk \
           --update-flags=$flags "${gselect_opt[$n]}" --rand-prune=$randprune \
          $dir/$x.mdl "${featspart[$n]}" "ark,s,cs:gunzip -c $dir_for_alignments/$n.ali.gz|" \
          $dir/$x.$n.acc || touch $dir/.error &
    done
    wait;
    [ -f $dir/.error ] && echo "Error accumulating stats on iteration $x" && exit 1;
    $cmd $dir/log/update.$x.log \
      sgmm-est --update-flags=$flags --write-occs=$dir/$[$x+1].occs $dir/$x.mdl \
        "sgmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
  else
    echo "[Skipping this stage]"
  fi
  rm $dir/$x.mdl $dir/$x.*.acc
  rm $dir/$x.occs  2>/dev/null
done
x=$[$niters+1]
rm $dir/final.mdl $dir/final.occs 2>/dev/null
ln -s $x.mdl $dir/final.mdl
ln -s $x.occs $dir/final.occs

if $have_vecs; then
  # If we have speaker vectors, we need an alignment model.
  # The point of this last phase of accumulation is to get Gaussian-level
  # alignments with the speaker vectors but accumulate stats without
  # any speaker vectors; we re-estimate M, w, c and S to get a model
  # that's compatible with not having speaker vectors.

  cur_alimdl=$dir/$x.mdl
  y=0;
  while [ $y -lt $numiters_alimdl ]; do
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
    else
      echo "[Skipping this stage]"
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
