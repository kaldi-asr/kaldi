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

# This is Subspace Gaussian Mixture Model (SGMM) training--
# see "The subspace Gaussian mixture model--A structured model for speech recognition"
# by D. Povey et al, Computer Speech and Language, 2011.

function error_exit () {
  echo -e "$@" >&2; exit 1;
}

function readint () {
  local retval=${1/#*=/};  # In case --switch=ARG format was used
  retval=${retval#0*}      # Strip any leading 0's
  [[ "$retval" =~ ^-?[1-9][0-9]*$ ]] \
    || error_exit "Argument \"$retval\" not an integer."
  echo $retval
}

nj=4       # Default number of jobs
stage=-4   # Default starting stage (start with tree building)
qcmd=""    # Options for the submit_jobs.sh script
sjopts=""  # Options for the submit_jobs.sh script

PROG=`basename $0`;
usage="Usage: $PROG [options] <num-substates> <phone-dim> <spk-dim> <data-dir> <lang-dir> <ali-dir> <ubm>\n
e.g.: $PROG 10000 40 39 data/train data/lang exp/tri2a_ali exp/ubm3c/final.ubm exp/sgmm3c\n\n
Options:\n
  --help\t\tPrint this message and exit\n
  --num-jobs INT\tNumber of parallel jobs to run (default=$nj).\n
  --qcmd STRING\tCommand for submitting a job to a grid engine (e.g. qsub) including switches.\n
  --sjopts STRING\tOptions for the 'submit_jobs.sh' script\n
  --stage INT\tStarting stage (e.g. -4 for SGMM init; 2 for iter 2; default=$stage)\n
";

while [ $# -gt 0 ]; do
  case "${1# *}" in  # ${1# *} strips any leading spaces from the arguments
    --help) echo -e $usage; exit 0 ;;
    --num-jobs) 
      shift; nj=`readint $1`;
      [ $nj -lt 1 ] && error_exit "--num-jobs arg '$nj' not positive.";
      shift ;;
    --qcmd)
      shift; qcmd=" --qcmd=${1}"; shift ;;
    --sjopts)
      shift; sjopts="$1"; shift ;;
    --stage)
      shift; stage=`readint $1`; shift ;;
    -*)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
    *)   break ;;   # end of options: interpreted as num-leaves
  esac
done

if [ $# != 8 ]; then
  error_exit $usage;
fi

[ -f path.sh ] && . path.sh

# This is SGMM with speaker vectors, on top of LDA+[something] features.
# Any speaker-specific transforms are obtained from the alignment directory.
# To be run from ..

totsubstates=$1
phndim=$2
spkdim=$3
data=$4
lang=$5
alidir=$6
ubm=$7
dir=$8

mkdir -p $dir || exit 1;

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

numiters=25   # Total number of iterations
numiters_alimdl=3 # Number of iterations for estimating alignment model.
maxiterinc=15 # Last iter to increase #substates on.
realign_iters="5 10 15"; 
spkvec_iters="5 8 12 17"
add_dim_iters="6 8 10 12"; # Iters on which to increase phn dim and/or spk dim,
   # if necessary, In most cases, either none of these or only the first of these 
   # will have any effect (we increase in increments of [feature dim])

oov_sym=`cat $lang/oov.txt`
silphonelist=`cat $lang/silphones.csl`

numsubstates=`cat $dir/numleaves`  # Initial #-substates.
# per-iter increment for #substates
incsubstates=$[($totsubstates-$numsubstates)/$maxiterinc]

# Initially don't have speaker vectors, but change this after we estimate them.
spkvecs_opt=
gselect_opt="--gselect=ark,s,cs:gunzip -c $dir/TASK_ID.gselect.gz|"

randprune=0.1
mkdir -p $dir/log 

featspart="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/TASK_ID/utt2spk ark:$alidir/TASK_ID.cmvn scp:$data/split$nj/TASK_ID/feats.scp ark:- | add-deltas ark:- ark:- |"

if [ ! -f $ubm ]; then
  echo "No UBM in $ubm"
  exit 1;
fi

if [ $stage -le -4 ]; then
  submit_jobs.sh "$qcmd" --log=$dir/log/init_sgmm.log $sjopts \
    sgmm-init --phn-space-dim=$phndim --spk-space-dim=$spkdim $lang/topo \
      $dir/tree $ubm $dir/0.mdl || error_exit "SGMM init failed."
fi

if [ $stage -le -3 ]; then
# Make training graphs (this is split in $nj parts).
  echo "Compiling training graphs"
  submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/log/compile_graphsTASK_ID.log \
    $sjopts compile-train-graphs $dir/tree $dir/0.mdl $lang/L.fst \
      "ark:sym2int.pl --map-oov '$oov_sym' --ignore-first-field $lang/words.txt < $data/split$nj/TASK_ID/text |" \
      "ark:|gzip -c >$dir/TASK_ID.fsts.gz" \
      || error_exit "Error compiling training graphs"
fi

if [ $stage -le -2 ]; then
  echo "Doing Gaussian selection"
  submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/log/gselectTASK_ID.log \
    $sjopts sgmm-gselect $dir/0.mdl "$featspart" "ark,t:|gzip -c > $dir/TASK_ID.gselect.gz" \
    || error_exit "Error doing Gaussian selection"
fi


if [ $stage -le -1 ]; then
  echo "Converting alignments"  # don't bother parallelizing; very fast.
  for n in `seq 1 $nj`; do
    convert-ali $alidir/final.mdl $dir/0.mdl $dir/tree \
      "ark:gunzip -c $alidir/$n.ali.gz|" "ark:|gzip -c >$dir/$n.ali.gz" \
      2>$dir/log/convert.$n.log 
  done
fi

x=0
while [ $x -lt $numiters ]; do
  if [ $x -eq 0 ]; then
    flags=vwcSt  # On first iter, don't update M or N.
  elif [ $spkdim -gt 0 -a $[$x%2] -eq 1 -a \
         $x -ge `echo $spkvec_iters | awk '{print $1}'` ]; then 
  # Update N on odd iterations after 1st spkvec iter, if we have spk-space.
    flags=vNwcSt
  else  # Else update M but not N.
    flags=vMwcSt
  fi

  if [ $stage -le $x ]; then
    echo "Pass $x: update flags = '$flags' "
    if echo $realign_iters | grep -w $x >/dev/null; then
      echo "Aligning data"
      submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/log/align.$x.TASK_ID.log  \
	$sjopts sgmm-align-compiled $spkvecs_opt $scale_opts "$gselect_opt" \
	--utt2spk=ark:$data/split$nj/TASK_ID/utt2spk --beam=8 --retry-beam=40 \
	$dir/$x.mdl "ark:gunzip -c $dir/TASK_ID.fsts.gz|" "$featspart" \
	"ark:|gzip -c >$dir/TASK_ID.ali.gz" \
	|| error_exit "Error realigning data on iter $x"
    fi

    if [ $spkdim -gt 0 ] && echo $spkvec_iters | grep -w $x >/dev/null; then
      submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/log/spkvecs.$x.TASK_ID.log \
	$sjopts ali-to-post "ark:gunzip -c $dir/TASK_ID.ali.gz|" ark:- \| \
        weight-silence-post 0.01 $silphonelist $dir/$x.mdl ark:- ark:- \| \
        sgmm-est-spkvecs --spk2utt=ark:$data/split$nj/TASK_ID/spk2utt \
          $spkvecs_opt "$gselect_opt" --rand-prune=$randprune $dir/$x.mdl \
          "$featspart" ark,s,cs:- ark:$dir/tmpTASK_ID.vecs \
	|| error_exit "Error computing speaker vectors on iter $x"
      for n in `seq 1 $nj`; do
        mv $dir/tmp${n}.vecs $dir/${n}.vecs;
      done
      spkvecs_opt="--spk-vecs=ark:$dir/TASK_ID.vecs"
    fi

    submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/log/acc.$x.TASK_ID.log \
      $sjopts sgmm-acc-stats --utt2spk=ark:$data/split$nj/TASK_ID/utt2spk \
	--update-flags=$flags --rand-prune=$randprune $spkvecs_opt \
	"$gselect_opt" $dir/$x.mdl "$featspart" \
	"ark,s,cs:ali-to-post 'ark:gunzip -c $dir/TASK_ID.ali.gz|' ark:-|" \
        $dir/$x.TASK_ID.acc || error_exit "Error accumulating stats on iter $x"

    add_dim_opts=
    if echo $add_dim_iters | grep -w $x >/dev/null; then
      add_dim_opts="--increase-phn-dim=$phndim --increase-spk-dim=$spkdim"
    fi

    submit_jobs.sh "$qcmd" --log=$dir/log/update.$x.log $sjopts \
      sgmm-est --update-flags=$flags --split-substates=$numsubstates \
	$add_dim_opts --write-occs=$dir/$[$x+1].occs $dir/$x.mdl \
	"sgmm-sum-accs - $dir/$x.*.acc|" $dir/$[$x+1].mdl \
	|| error_exit "Error in pass $x estimation."

    rm -f $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs
  fi

  if [ $x -lt $maxiterinc ]; then
    numsubstates=$[$numsubstates+$incsubstates]
  fi
  x=$[$x+1];
done

( cd $dir; rm final.mdl final.occs 2>/dev/null; 
  ln -s $x.mdl final.mdl; 
  ln -s $x.occs final.occs )

if [ $spkdim -gt 0 ]; then
  # If we have speaker vectors, we need an alignment model.
  # The point of this last phase of accumulation is to get Gaussian-level
  # alignments with the speaker vectors but accumulate stats without
  # any speaker vectors; we re-estimate M, w, c and S to get a model
  # that's compatible with not having speaker vectors.

  # We do this for a few iters, in this recipe.
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
      submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/log/acc_ali.$y.TASK_ID.log \
	$sjopts ali-to-post "ark:gunzip -c $dir/TASK_ID.ali.gz|" ark:- \| \
          sgmm-post-to-gpost $spkvecs_opt "$gselect_opt" \
          --utt2spk=ark:$data/split$nj/TASK_ID/utt2spk $dir/$x.mdl \
	  "$featspart" ark,s,cs:- ark:- \| \
          sgmm-acc-stats-gpost --update-flags=$flags $cur_alimdl "$featspart" \
          ark,s,cs:- $dir/$y.TASK_ID.aliacc \
	|| error_exit "Error accumulating stats for alignment model on iter $y"

      submit_jobs.sh "$qcmd" --log=$dir/log/update_ali.$y.log $sjopts \
	sgmm-est --update-flags=$flags --remove-speaker-space=true \
	  $cur_alimdl "sgmm-sum-accs - $dir/$y.*.aliacc|" $dir/$[$y+1].alimdl \
	|| error_exit "Error estimating alignment model on iter $y";
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
