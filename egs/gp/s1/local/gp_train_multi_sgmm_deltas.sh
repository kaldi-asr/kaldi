#!/bin/bash -u

# Copyright 2012  Arnab Ghoshal
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
#  retval=${retval#0*}      # Strip any leading 0's
  [[ "$retval" =~ ^-?[0-9][0-9]*$ ]] \
    || error_exit "Argument \"$retval\" not an integer."
  echo $retval
}

function est_alimodel () {
# If we have speaker vectors, we need an alignment model. This function gets 
# the Gaussian-level alignments with the speaker vectors but accumulates stats 
# without any speaker vectors; we re-estimate M, w, c and S to get a model
# that's compatible with not having speaker vectors. Note that the transitions
# are not updated since the decoding graph will be shared with the normal model.
  local lx=$1
  for L in $LANGUAGES; do
    wdir=$dir/$L
    local lspkdim=`sgmm-info $wdir/$lx.mdl | grep speaker | awk '{print $NF}'`
    if [ "$lspkdim" -le 0 ]; then
      echo "est_alimodel: No speaker space in model '$wdir/$lx.mdl'. Returning."
      return
    fi
  done

  local y=0;
  local lflags=MwcS  # First time don't update v
  while [ $y -lt $numiters_alimdl ]; do
    [ $y -gt 0 ] && lflags=vMwcS
    echo "Pass $y of building alignment model, flags = '$lflags'"
    local lmulti_est_opts=''  # model, acc, model-out, occs-out tuples
    for L in $LANGUAGES; do 
    (
      data=data/$L/train
      lang=data/$L/lang
      wdir=$dir/$L
      local cur_alimdl=$wdir/tmp$y.alimdl
      [ $y -eq 0 ] && cur_alimdl=$wdir/$lx.mdl
      feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/TASK_ID/utt2spk ark:$wdir/TASK_ID.cmvn scp:$data/split$nj/TASK_ID/feats.scp ark:- | add-deltas ark:- ark:- |"
      gselect_opt="--gselect=ark,s,cs:gunzip -c $wdir/TASK_ID.gselect.gz|"
      spkvecs_opt="--spk-vecs=ark:$wdir/TASK_ID.vecs"

      submit_jobs.sh "$qcmd" --njobs=$nj --log=$wdir/log/acc_ali${lx}_$y.TASK_ID.log \
	$sjopts ali-to-post "ark:gunzip -c $wdir/TASK_ID.ali.gz|" ark:- \| \
          sgmm-post-to-gpost $spkvecs_opt "$gselect_opt" \
          --utt2spk=ark:$data/split$nj/TASK_ID/utt2spk $wdir/$lx.mdl \
	  "$feats" ark,s,cs:- ark:- \| \
          sgmm-acc-stats-gpost --update-flags=$lflags $cur_alimdl "$feats" \
          ark,s,cs:- $wdir/$y.TASK_ID.aliacc \
	|| { touch $dir/err; \
	  error_exit "$L; Align model iter $y: Error accumulating stats"; }

      # Summing accs is quite fast; run locally
      sgmm-sum-accs $wdir/sum.aliacc $wdir/$y.*.aliacc || \
	{ touch $dir/err; \
	  error_exit "$L; Align model iter $y: Error summing stats"; }
    )&  # Accumulate in parallel for different languages
      wdir=$dir/$L
      local cur_alimdl=$wdir/tmp$y.alimdl
      [ $y -eq 0 ] && cur_alimdl=$wdir/$lx.mdl
      lmulti_est_opts="$lmulti_est_opts $cur_alimdl $wdir/sum.aliacc $wdir/tmp$[$y+1].alimdl $wdir/tmp$[$y+1].occs"
    done
    wait

    submit_jobs.sh "$qcmd" --log=$dir/log/update_ali.$y.log $sjopts \
      sgmm-est-multi --update-flags=$lflags --remove-speaker-space=true \
	$lmulti_est_opts \
      || error_exit "Error estimating alignment models on iter $y";

    rm -f $dir/??/$y.*.aliacc $dir/??/sum.aliacc || exit 1;
    [ $y -gt 0 ]  && rm $dir/??/tmp$y.{alimdl,occs} 
    y=$[$y+1]
  done

  for L in $LANGUAGES; do
    mv $dir/$L/tmp$y.alimdl $dir/$L/$lx.alimdl
  done
}

nj=4       # Default number of jobs
stage=-5   # Default starting stage (start with tree building)
qcmd=""    # Options for the submit_jobs.sh script
sjopts=""  # Options for the submit_jobs.sh script
LANGUAGES='GE PO SP SW'  # Languages processed

PROG=`basename $0`;
usage="Usage: $PROG [options] <phone-dim> <spk-dim> <ubm> <out-dir>\n
e.g.: $PROG 40 39 exp/ubm3c/final.ubm exp/sgmm3c\n\n
Options:\n
  --help\t\tPrint this message and exit\n
  --lang STR\tList of languages to process (default = '$LANGUAGES')\n
  --num-jobs INT\tNumber of parallel jobs to run (default=$nj).\n
  --qcmd STR\tCommand for submitting a job to a grid engine (e.g. qsub) including switches.\n
  --sjopts STR\tOptions for the 'submit_jobs.sh' script\n
  --stage INT\tStarting stage (e.g. -4 for SGMM init; 2 for iter 2; default=$stage)\n
";

echo "$PROG $@"
while [ $# -gt 0 ]; do
  case "${1# *}" in  # ${1# *} strips any leading spaces from the arguments
    --help) echo -e $usage; exit 0 ;;
    --lang) LANGUAGES="$2"; shift 2 ;;
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

if [ $# != 4 ]; then
  error_exit $usage;
fi

[ -f path.sh ] && . path.sh

# This is SGMM with speaker vectors, on top of LDA+[something] features.
# Any speaker-specific transforms are obtained from the alignment directory.
# To be run from ..

phndim=$1
spkdim=$2
ubm=$3
dir=$4

[ -f $ubm ] || error_exit "UBM file '$ubm' does not exist"
mkdir -p $dir/log || error_exit "Cannot create '$dir/log'"

# (1): Model initialization; training graph and initial alignment generation.
for L in $LANGUAGES; do
(
  data=data/$L/train
  lang=data/$L/lang
  alidir=exp/$L/tri2a_ali
  wdir=$dir/$L
  oov_sym=`cat $lang/oov.txt`
  mkdir -p $wdir/log || error_exit "Cannot create working directory '$wdir'"

  # Initialize the model (removed the --spk-space-dim option)
  if [ $stage -le -5 ]; then
    echo "$L: Initializing model"
    submit_jobs.sh "$qcmd" --log=$wdir/log/init_sgmm.log $sjopts \
      sgmm-init --phn-space-dim=$phndim $lang/topo $wdir/tree $ubm \
	$wdir/0.mdl || { touch $dir/err; error_exit "$L: SGMM init failed."; }
  fi

  # Make training graphs
  if [ $stage -le -4 ]; then
    echo "$L: Compiling training graphs"
    submit_jobs.sh "$qcmd" --njobs=$nj --log=$wdir/log/mkgraphs.TASK_ID.log \
      $sjopts compile-train-graphs $wdir/tree $wdir/0.mdl $lang/L.fst \
	"ark:sym2int.pl --map-oov '$oov_sym' --ignore-first-field $lang/words.txt < $data/split$nj/TASK_ID/text |" \
	"ark:|gzip -c >$wdir/TASK_ID.fsts.gz" \
      || { touch $dir/err; error_exit "$L: Error compiling training graphs"; }
  fi

  if [ $stage -le -3 ]; then
    echo "$L: Converting alignments"
    submit_jobs.sh "$qcmd" --njobs=$nj --log=$wdir/log/convert.TASK_ID.log \
      $sjopts convert-ali $alidir/final.mdl $wdir/0.mdl $wdir/tree \
	"ark:gunzip -c $alidir/TASK_ID.ali.gz|" \
	"ark:|gzip -c >$wdir/TASK_ID.ali.gz" \
      || { touch $dir/err; error_exit "$L: Convert alignment failed."; }
  fi

  if [ $stage -le -2 ]; then
    echo "$L: Computing cepstral mean and variance statistics"
    submit_jobs.sh "$qcmd" --njobs=$nj $sjopts --log=$wdir/log/cmvn.TASK_ID.log \
      compute-cmvn-stats --spk2utt=ark:$data/split$nj/TASK_ID/spk2utt \
	scp:$data/split$nj/TASK_ID/feats.scp ark:$wdir/TASK_ID.cmvn \
      || { touch $dir/err; error_exit "$L: Computing CMN/CVN stats failed."; }
  fi

  feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/TASK_ID/utt2spk ark:$wdir/TASK_ID.cmvn scp:$data/split$nj/TASK_ID/feats.scp ark:- | add-deltas ark:- ark:- |"

  if [ $stage -le -1 ]; then
    echo "$L: Doing Gaussian selection"
    submit_jobs.sh "$qcmd" --njobs=$nj --log=$wdir/log/gselectTASK_ID.log \
      $sjopts sgmm-gselect $wdir/0.mdl "$feats" "ark,t:|gzip -c > $wdir/TASK_ID.gselect.gz" \
      || { touch $dir/err; error_exit "$L: Error doing Gaussian selection"; }
  fi
)&  # Run the language-specific initializations in parallel
done
wait
[ -f $dir/err ] && { rm $dir/err; error_exit "Error initializing models."; }

# Language independent constants
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
numiters_alimdl=3 # Number of iterations for estimating alignment model.
incsub_interval=8   # increase substates every 8 iterations
# total substates after each such increment
total_substates=( 5000 7000 9000 12000 16000 20000 25000 30000 35000 40000 )
# For a given number of substates, iterate for $incsub_interval iterations
numiters=$[(${#total_substates[@]}+1)*$incsub_interval]
realign_interval=4  # realign every 4 iterations
spkvec_start=8      # use speaker subspace *after* 8 iterations
spkvec_interval=2   # reestimate the speaker vectors every 2 iterations
randprune=0.1

# Initially don't have speaker vectors, but change this after we estimate them.
spkvecs_gen=0

x=0
while [ $x -lt $numiters ]; do
  if [ $x -eq 0 ]; then
    flags=v  # On first iter, don't update M or N.
  elif [ $spkdim -gt 0 -a $[$x%2] -eq 0 -a $x -gt $spkvec_start ]; then 
  # Update N on odd iterations after 1st spkvec iter, if we have spk-space.
    flags=NwSvct
  else  # Else update M but not N.
    flags=MwSvct
  fi

  if [ $stage -le $x ]; then
    echo "Pass $x: update flags = '$flags' "
    multi_est_opts=''  # Will contain model, acc, model-out, occs-out tuples
    for L in $LANGUAGES; do 
    (
      data=data/$L/train
      lang=data/$L/lang
      wdir=$dir/$L
      feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/TASK_ID/utt2spk ark:$wdir/TASK_ID.cmvn scp:$data/split$nj/TASK_ID/feats.scp ark:- | add-deltas ark:- ark:- |"
      gselect_opt="--gselect=ark,s,cs:gunzip -c $wdir/TASK_ID.gselect.gz|"
      if [ $spkdim -gt 0 -a $spkvecs_gen -eq 1 ]; then
	spkvecs_opt="--spk-vecs=ark:$wdir/TASK_ID.vecs"
      else
	spkvecs_opt=''
      fi
      silphonelist=`cat $lang/silphones.csl`
#	numsubstates=`cat $wdir/numleaves`  # Initial #-substates.

      if [ $[$x%$realign_interval] -eq 0 -a $x -gt 0 ]; then
        echo "$L; iter $x: Aligning data"
	submit_jobs.sh "$qcmd" $sjopts --log=$wdir/log/align.$x.TASK_ID.log \
	  --njobs=$nj sgmm-align-compiled $spkvecs_opt $scale_opts \
	    "$gselect_opt" --utt2spk=ark:$data/split$nj/TASK_ID/utt2spk \
	    --beam=8 --retry-beam=40 $wdir/$x.mdl \
	    "ark:gunzip -c $wdir/TASK_ID.fsts.gz|" "$feats" \
	    "ark:|gzip -c >$wdir/TASK_ID.ali.gz" || \
	    { touch $dir/err; error_exit "$L, it $x: Error realigning data"; }
      fi

      if [ $spkdim -gt 0 -a $x -gt $spkvec_start \
	  -a $[$x%$spkvec_interval] -eq 0 ]; then
        echo "$L; iter $x: Computing speaker vectors"
	submit_jobs.sh "$qcmd" --njobs=$nj --log=$wdir/log/spkvecs.$x.TASK_ID.log \
	  $sjopts ali-to-post "ark:gunzip -c $wdir/TASK_ID.ali.gz|" ark:- \| \
          weight-silence-post 0.01 $silphonelist $wdir/$x.mdl ark:- ark:- \| \
          sgmm-est-spkvecs --spk2utt=ark:$data/split$nj/TASK_ID/spk2utt \
          $spkvecs_opt "$gselect_opt" --rand-prune=$randprune $wdir/$x.mdl \
          "$feats" ark,s,cs:- ark:$wdir/tmpTASK_ID.vecs || \
	  { touch $dir/err; error_exit "$L, it $x: Error computing spkvecs"; }
	for n in `seq 1 $nj`; do
          mv $wdir/tmp${n}.vecs $wdir/${n}.vecs;
	done
	spkvecs_gen=1
      fi

      submit_jobs.sh "$qcmd" --njobs=$nj --log=$wdir/log/acc.$x.TASK_ID.log \
	$sjopts sgmm-acc-stats --utt2spk=ark:$data/split$nj/TASK_ID/utt2spk \
	  --update-flags=$flags --rand-prune=$randprune $spkvecs_opt \
	  "$gselect_opt" $wdir/$x.mdl "$feats" \
	  "ark,s,cs:ali-to-post 'ark:gunzip -c $wdir/TASK_ID.ali.gz|' ark:-|" \
          $wdir/$x.TASK_ID.acc || \
	  { touch $dir/err; error_exit "$L, it $x: Error accumulating stats"; }

      # Summing accs is quite fast; run locally
      sgmm-sum-accs $wdir/sum.acc $wdir/$x.*.acc || \
	  { touch $dir/err; error_exit "$L, it $x: Error summing stats"; }
    ) &  # Accumulate in parallel for different languages
      wdir=$dir/$L
      multi_est_opts="$multi_est_opts $wdir/$x.mdl $wdir/sum.acc $wdir/$[$x+1].mdl $wdir/$[$x+1].occs"
    done
    wait
    [ -f $dir/err ] && \
      { rm $dir/err; error_exit "Iter $x: Error in accumulation"; }

    add_dim_opts=''
    if [ $x -eq $spkvec_start ]; then
      add_dim_opts="--increase-spk-dim=$spkdim --increase-phn-dim=$phndim"
    elif [ $x -eq $[$spkvec_start*2] ]; then
      add_dim_opts="--increase-spk-dim=$spkdim --increase-phn-dim=$phndim"
    fi
    split_opts=''
    if [ $[$x%$incsub_interval] -eq 1 -a $x -gt 1 ]; then
      index=$[($x/$incsub_interval)-1]
      numsubstates=${total_substates[$index]}
      split_opts="--split-substates=$numsubstates"
    fi

    submit_jobs.sh "$qcmd" --log=$dir/log/update.$x.log $sjopts \
      sgmm-est-multi --update-flags=$flags $split_opts $add_dim_opts \
	$multi_est_opts || error_exit "Error in pass $x estimation."

    # If using speaker vectors, estimate alignment model without spkvecs
    if [ $[$x%$incsub_interval] -eq 0 -a $x -gt 0 ]; then
      chmod -w $dir/??/$x.mdl $dir/??/$x.occs  # Preserve for scoring
      [ $spkdim -gt 0 ] && est_alimodel $x;
    else
      rm -f $dir/??/$x.mdl $dir/??/$x.occs
    fi
    rm -f $dir/??/$x.*.acc $dir/??/sum.acc
  fi  # End of current stage
  x=$[$x+1];
done

for L in $LANGUAGES; do
  ( 
    wdir=$dir/$L
    rm -f $wdir/final.mdl $wdir/final.occs;
    chmod -w $wdir/$x.mdl $wdir/$x.occs  # Preserve for scoring
    ln -s $wdir/$x.mdl $wdir/final.mdl; 
    ln -s $wdir/$x.occs $wdir/final.occs;
    # If using speaker vectors, estimate alignment model without spkvecs
    [ $spkdim -gt 0 ] && est_alimodel $wdir/$x.mdl;
    rm -f $wdir/final.alimdl;
    ln -sf $wdir/$x.alimdl $wdir/final.alimdl;

    # Print out summary of the warning messages.
    for x in $wdir/log/*.log; do 
      n=`grep WARNING $x | wc -l`; 
      if [ $n -ne 0 ]; then echo "$n warnings in $x"; fi;
    done
  )
done

echo Done
