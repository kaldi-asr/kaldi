#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This script does MPE or fMMI state-level minimum bayes risk (sMBR) training.
# Note: the temporary data is put in <exp-dir>/degs/, so if you want
# to use a different disk for that, just make that a soft link to some other
# volume.

# Begin configuration section.
cmd=run.pl
num_epochs=4       # Number of epochs of training
learning_rate=0.00002
acoustic_scale=0.1  # acoustic scale for MMI/MPFE/SMBR training.
criterion=smbr
boost=0.0       # option relevant for MMI
drop_frames=false #  option relevant for MMI
num_jobs_nnet=4    # Number of neural net jobs to run in parallel.  Note: this
                   # will interact with the learning rates (if you decrease
                   # this, you'll have to decrease the learning rate, and vice
                   # versa).
samples_per_iter=400000 # measured in frames, not in "examples"

spk_vecs_dir=
modify_learning_rates=false
last_layer_factor=1.0  # relates to modify-learning-rates
first_layer_factor=1.0 # relates to modify-learning-rates
shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.


stage=-8

io_opts="-tc 5" # for jobs with a lot of I/O, limits the number running at one time.   These don't

num_threads=16  # this is the default but you may want to change it, e.g. to 1 if
                # using GPUs.
parallel_opts="-pe smp 16 -l ram_free=1G,mem_free=1G" # by default we use 4 threads; this lets the queue know.
  # note: parallel_opts doesn't automatically get adjusted if you adjust num-threads.
transform_dir= # If this is a SAT system, directory for transforms
cleanup=true
transform_dir=
degs_dir=
retroactive=false
# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 6 ]; then
  echo "Usage: $0 [opts] <data> <lang> <ali-dir> <denlat-dir> <src-model-file> <exp-dir>"
  echo " e.g.: $0 data/train data/lang exp/tri3_ali exp/tri4_nnet_denlats exp/tri4/final.mdl exp/tri4_mpe"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-epochs <#epochs|4>                        # Number of epochs of training"
  echo "  --initial-learning-rate <initial-learning-rate|0.0002> # Learning rate at start of training"
  echo "  --final-learning-rate  <final-learning-rate|0.0004>   # Learning rate at end of training"
  echo "  --num-jobs-nnet <num-jobs|8>                     # Number of parallel jobs to use for main neural net"
  echo "                                                   # training (will affect results as well as speed; try 8, 16)"
  echo "                                                   # Note: if you increase this, you may want to also increase"
  echo "                                                   # the learning rate."
  echo "  --num-threads <num-threads|16>                   # Number of parallel threads per job (will affect results"
  echo "                                                   # as well as speed; may interact with batch size; if you increase"
  echo "                                                   # this, you may want to decrease the batch size."
  echo "  --parallel-opts <opts|\"-pe smp 16 -l ram_free=1G,mem_free=1G\">      # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads... note, you might have to reduce mem_free,ram_free"
  echo "                                                   # versus your defaults, because it gets multiplied by the -pe smp argument."
  echo "  --io-opts <opts|\"-tc 10\">                      # Options given to e.g. queue.pl for jobs that do a lot of I/O."
  echo "  --samples-per-iter <#samples|200000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --stage <stage|-8>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  echo "  --criterion <criterion|smbr>                     # Training criterion: may be smbr, mmi or mpfe"
  echo "  --boost <boost|0.0>                              # Boosting factor for MMI (e.g., 0.1)"
  echo "  --modify-learning-rates <true,false|false>       # If true, modify learning rates to try to equalize relative"
  echo "                                                   # changes across layers."
  echo "  --degs-dir <dir|"">                              # Directory for discriminative examples, e.g. exp/foo/degs"
  exit 1;
fi

data=$1
lang=$2
alidir=$3
denlatdir=$4
src_model=$5
dir=$6

# Check some files.
for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/num_jobs $alidir/tree \
         $denlatdir/lat.1.gz $denlatdir/num_jobs $src_model; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

nj=$(cat $alidir/num_jobs) || exit 1; # caution: $nj is the number of
                                      # splits of the denlats and alignments, but
                                      # num_jobs_nnet is the number of nnet training
                                      # jobs we run in parallel.
if ! [ $nj == $(cat $denlatdir/num_jobs) ]; then
  echo "Number of jobs mismatch: $nj versus $(cat $denlatdir/num_jobs)"
  exit 1;
fi

mkdir -p $dir/log || exit 1;
[ -z "$degs_dir" ] && mkdir -p $dir/degs

sdata=$data/split$nj
utils/split_data.sh $data $nj

splice_opts=`cat $alidir/splice_opts 2>/dev/null`
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
cp $alidir/splice_opts $dir 2>/dev/null
cp $alidir/cmvn_opts $dir 2>/dev/null
cp $alidir/tree $dir

## Set up features.
## Don't support deltas, only LDA or raw (mainly because deltas are less frequently used).
if [ -z $feat_type ]; then
  if [ -f $alidir/final.mat ] && [ ! -f $transform_dir/raw_trans.1 ]; then feat_type=lda; else feat_type=raw; fi
fi
echo "$0: feature type is $feat_type"

case $feat_type in
  raw) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
   ;;
  lda) 
    splice_opts=`cat $alidir/splice_opts 2>/dev/null`
    cp $alidir/final.mat $dir    
    feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

[ -z "$transform_dir" ] && transform_dir=$alidir

if [ -f $transform_dir/trans.1 ] && [ $feat_type != "raw" ]; then
  echo "$0: using transforms from $transform_dir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/trans.JOB ark:- ark:- |"
fi
if [ -f $transform_dir/raw_trans.1 ] && [ $feat_type == "raw" ]; then
  echo "$0: using raw-fMLLR transforms from $transform_dir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/raw_trans.JOB ark:- ark:- |"
fi


if [ -z "$degs_dir" ]; then
  if [ $stage -le -8 ]; then
    echo "$0: working out number of frames of training data"
    num_frames=`feat-to-len scp:$data/feats.scp ark,t:- | awk '{x += $2;} END{print x;}'` || exit 1;
    echo $num_frames > $dir/num_frames
    # Working out number of iterations per epoch.
    iters_per_epoch=`perl -e "print int($num_frames/($samples_per_iter * $num_jobs_nnet) + 0.5);"` || exit 1;
    [ $iters_per_epoch -eq 0 ] && iters_per_epoch=1
    echo $iters_per_epoch > $dir/degs/iters_per_epoch  || exit 1;
  else
    num_frames=$(cat $dir/num_frames) || exit 1;
    iters_per_epoch=$(cat $dir/degs/iters_per_epoch) || exit 1;
  fi

  samples_per_iter_real=$[$num_frames/($num_jobs_nnet*$iters_per_epoch)]
  echo "$0: Every epoch, splitting the data up into $iters_per_epoch iterations,"
  echo "$0: giving samples-per-iteration of $samples_per_iter_real (you requested $samples_per_iter)."
else
  iters_per_epoch=$(cat $degs_dir/iters_per_epoch) || exit 1;
  [ -z "$iters_per_epoch" ] && exit 1;
  echo "$0: Every epoch, splitting the data up into $iters_per_epoch iterations"
fi

if [ $stage -le -7 ]; then
  echo "$0: Copying initial model and removing any preconditioning"
  nnet-am-copy --learning-rate=$learning_rate --remove-preconditioning=true \
    "$src_model" $dir/0.mdl || exit 1;
fi

if [ $stage -le -6 ] && [ -z "$degs_dir" ]; then
  echo "$0: getting initial training examples by splitting lattices"
  if [ ! -z $spk_vecs_dir ]; then
    [ ! -f $spk_vecs_dir/vecs.1 ] && echo "No such file $spk_vecs_dir/vecs.1" && exit 1;
    spk_vecs_opt=("--spk-vecs=ark:cat $spk_vecs_dir/vecs.*|" "--utt2spk=ark:$data/utt2spk")
  else
    spk_vecs_opt=()
  fi

  egs_list=
  for n in `seq 1 $num_jobs_nnet`; do
    egs_list="$egs_list ark:$dir/degs/degs_orig.$n.JOB.ark"
  done

  $cmd $io_opts JOB=1:$nj $dir/log/get_egs.JOB.log \
    nnet-get-egs-discriminative --criterion=$criterion --drop-frames=$drop_frames \
    "${spk_vecs_opt[@]}" $dir/0.mdl "$feats" \
    "ark,s,cs:gunzip -c $alidir/ali.JOB.gz |" \
    "ark,s,cs:gunzip -c $denlatdir/lat.JOB.gz|" ark:- \| \
    nnet-copy-egs-discriminative ark:- $egs_list || exit 1;
fi

if [ $stage -le -5 ] && [ -z "$degs_dir" ]; then
  echo "$0: rearranging examples into parts for different parallel jobs"

  # combine all the "egs_orig.JOB.*.scp" (over the $nj splits of the data) and
  # then split into multiple parts egs.JOB.*.scp for different parts of the
  # data, 0 .. $iters_per_epoch-1.

  if [ $iters_per_epoch -eq 1 ]; then
    echo "Since iters-per-epoch == 1, just concatenating the data."
    for n in `seq 1 $num_jobs_nnet`; do
      cat $dir/degs/degs_orig.$n.*.ark > $dir/degs/degs_tmp.$n.0.ark || exit 1;
      rm $dir/degs/degs_orig.$n.*.ark  # don't "|| exit 1", due to NFS bugs...
    done
  else # We'll have to split it up using nnet-copy-egs.
    egs_list=
    for n in `seq 0 $[$iters_per_epoch-1]`; do
      egs_list="$egs_list ark:$dir/degs/degs_tmp.JOB.$n.ark"
    done
    # note, the "|| true" below is a workaround for NFS bugs
    # we encountered running this script with Debian-7, NFS-v4.
    $cmd $io_opts JOB=1:$num_jobs_nnet $dir/log/split_egs.JOB.log \
      nnet-copy-egs-discriminative --srand=JOB \
        "ark:cat $dir/degs/degs_orig.JOB.*.ark|" $egs_list '&&' \
        '(' rm $dir/degs/degs_orig.JOB.*.ark '||' true ')' || exit 1;
  fi
fi


if [ $stage -le -4 ] && [ -z "$degs_dir" ]; then
  # Next, shuffle the order of the examples in each of those files.
  # Each one should not be too large, so we can do this in memory.
  # Then combine the examples together to form suitable-size minibatches
  # (for discriminative examples, it's one example per minibatch, so we
  # have to combine the lattices).
  echo "Shuffling the order of training examples"
  echo "(in order to avoid stressing the disk, these won't all run at once)."

  # note, the "|| true" below is a workaround for NFS bugs
  # we encountered running this script with Debian-7, NFS-v4.
  for n in `seq 0 $[$iters_per_epoch-1]`; do
    $cmd $io_opts JOB=1:$num_jobs_nnet $dir/log/shuffle.$n.JOB.log \
      nnet-shuffle-egs-discriminative "--srand=\$[JOB+($num_jobs_nnet*$n)]" \
      ark:$dir/degs/degs_tmp.JOB.$n.ark ark:- \| \
      nnet-combine-egs-discriminative ark:- ark:$dir/degs/degs.JOB.$n.ark '&&' \
      '(' rm $dir/degs/degs_tmp.JOB.$n.ark '||' true ')' || exit 1;
  done
fi

if [ -z "$degs_dir" ]; then
  degs_dir=$dir/degs
fi

num_iters=$[$num_epochs * $iters_per_epoch];

echo "$0: Will train for $num_epochs epochs = $num_iters iterations"

if [ $num_threads -eq 1 ]; then
 train_suffix="-simple" # this enables us to use GPU code if
                        # we have just one thread.
else
  train_suffix="-parallel --num-threads=$num_threads"
fi


x=0   
while [ $x -lt $num_iters ]; do
  if [ $x -ge 0 ] && [ $stage -le $x ]; then
    
    echo "Training neural net (pass $x)"

    $cmd $parallel_opts JOB=1:$num_jobs_nnet $dir/log/train.$x.JOB.log \
      nnet-train-discriminative$train_suffix --silence-phones=$silphonelist \
       --criterion=$criterion --drop-frames=$drop_frames \
       --boost=$boost --acoustic-scale=$acoustic_scale \
       $dir/$x.mdl ark:$degs_dir/degs.JOB.$[$x%$iters_per_epoch].ark $dir/$[$x+1].JOB.mdl \
      || exit 1;

    nnets_list=
    for n in `seq 1 $num_jobs_nnet`; do
      nnets_list="$nnets_list $dir/$[$x+1].$n.mdl"
    done

    $cmd $dir/log/average.$x.log \
      nnet-am-average $nnets_list $dir/$[$x+1].mdl || exit 1;

    if $modify_learning_rates; then
      $cmd $dir/log/modify_learning_rates.$x.log \
        nnet-modify-learning-rates --retroactive=$retroactive \
        --last-layer-factor=$last_layer_factor \
        --first-layer-factor=$first_layer_factor \
        $dir/$x.mdl $dir/$[$x+1].mdl $dir/$[$x+1].mdl || exit 1;
    fi
    rm $nnets_list
  fi

  x=$[$x+1]
done

rm $dir/final.mdl 2>/dev/null
ln -s $x.mdl $dir/final.mdl


echo Done

if $cleanup; then
  echo Cleaning up data

  echo Removing training examples
  if [ -d $dir/degs ] && [ ! -L $dir/degs ]; then # only remove if directory is not a soft link.
    rm $dir/degs/degs*
  fi

  echo Removing most of the models
  for x in `seq 0 $num_iters`; do
    if [ $[$x%$iters_per_epoch] -ne 0 ]; then
      # delete all but the epoch-final models.
      rm $dir/$x.mdl 2>/dev/null
    fi
  done
fi

for n in $(seq 0 $num_epochs); do
  x=$[$n*$iters_per_epoch]
  rm $dir/epoch$n.mdl 2>/dev/null
  ln -s $x.mdl $dir/epoch$n.mdl
done
