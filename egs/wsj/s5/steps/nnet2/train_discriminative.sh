#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This script does MPE or fMMI state-level minimum bayes risk (sMBR) training.
# Note: the temporary data is put in <exp-dir>/egs/, so if you want
# to use a different disk for that, just make that a soft link to some other
# volume.

# Begin configuration section.
cmd=run.pl
num_epochs=4       # Number of epochs of training
initial_learning_rate=0.00002
final_learning_rate=0.00001
final_learning_rate_factor=0.5 # Train last two layers half as fast as the other layers.



num_jobs_nnet=4    # Number of neural net jobs to run in parallel.  Note: this
                   # will interact with the learning rates (if you decrease
                   # this, you'll have to decrease the learning rate, and vice
                   # versa).
samples_per_iter=200000 # measured in frames, not in "examples"

spk_vecs_dir=

shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.


stage=-8

io_opts="-tc 5" # for jobs with a lot of I/O, limits the number running at one time.   These don't

num_threads=16 # this is the default but you may want to change it, e.g. to 1 if
               # using GPUs.
parallel_opts="-pe smp 16 -l ram_free=1G,mem_free=1G" # by default we use 16 threads; this lets the queue know.
  # note: parallel_opts doesn't automatically get adjusted if you adjust num-threads.
transform_dir= # If this is a SAT system, directory for transforms
cleanup=true
transform_dir=
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

mkdir -p $dir || exit 1;


sdata=$data/split$nj
utils/split_data.sh $data $nj

mkdir -p $dir/log $dir/egs
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
cp $alidir/splice_opts $dir 2>/dev/null
cp $alidir/tree $dir

## Set up features.
## Don't support deltas, only LDA or raw (mainly because deltas are less frequently used).
if [ -z $feat_type ]; then
  if [ -f $alidir/final.mat ] && [ ! -f $transform_dir/raw_trans.1 ]; then feat_type=lda; else feat_type=raw; fi
fi
echo "$0: feature type is $feat_type"

case $feat_type in
  raw) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
   ;;
  lda) 
    splice_opts=`cat $alidir/splice_opts 2>/dev/null`
    cp $alidir/final.mat $dir    
    feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
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

if [ $stage -le -8 ]; then
  echo "$0: working out number of frames of training data"
  num_frames=`feat-to-len scp:$data/feats.scp ark,t:- | awk '{x += $2;} END{print x;}'` || exit 1;
  echo $num_frames > $dir/num_frames
else
  num_frames=`cat $dir/num_frames` || exit 1;
fi

# Working out number of iterations per epoch.
iters_per_epoch=`perl -e "print int($num_frames/($samples_per_iter * $num_jobs_nnet) + 0.5);"` || exit 1;
[ $iters_per_epoch -eq 0 ] && iters_per_epoch=1
echo $iters_per_epoch > $dir/egs/iters_per_epoch  || exit 1;

samples_per_iter_real=$[$num_frames/($num_jobs_nnet*$iters_per_epoch)]
echo "$0: Every epoch, splitting the data up into $iters_per_epoch iterations,"
echo "$0: giving samples-per-iteration of $samples_per_iter_real (you requested $samples_per_iter)."



if [ $stage -le -7 ]; then
  echo "$0: Copying initial model and removing any preconditioning"
  nnet-am-copy --remove-preconditioning=true $src_model $dir/0.mdl || exit 1;
fi

if [ $stage -le -6 ]; then
  echo "$0: getting initial training examples by splitting lattices"
  if [ ! -z $spk_vecs_dir ]; then
    [ ! -f $spk_vecs_dir/vecs.1 ] && echo "No such file $spk_vecs_dir/vecs.1" && exit 1;
    spk_vecs_opt=("--spk-vecs=ark:cat $spk_vecs_dir/vecs.*|" "--utt2spk=ark:$data/utt2spk")
  else
    spk_vecs_opt=()
  fi

  egs_list=
  for n in `seq 1 $num_jobs_nnet`; do
    egs_list="$egs_list ark:$dir/egs/egs_orig.$n.JOB.ark"
  done

  $cmd $io_opts JOB=1:$nj $dir/log/get_egs.JOB.log \
    nnet-get-egs-discriminative "${spk_vecs_opt[@]}" $dir/0.mdl "$feats" \
    "ark,s,cs:gunzip -c $alidir/ali.JOB.gz |" \
    "ark,s,cs:gunzip -c $denlatdir/lat.JOB.gz|" ark:- \| \
    nnet-copy-egs-discriminative ark:- $egs_list || exit 1;
fi

if [ $stage -le -5 ]; then
  echo "$0: rearranging examples into parts for different parallel jobs"

  # combine all the "egs_orig.JOB.*.scp" (over the $nj splits of the data) and
  # then split into multiple parts egs.JOB.*.scp for different parts of the
  # data, 0 .. $iters_per_epoch-1.

  if [ $iters_per_epoch -eq 1 ]; then
    echo "Since iters-per-epoch == 1, just concatenating the data."
    for n in `seq 1 $num_jobs_nnet`; do
      cat $dir/egs/egs_orig.$n.*.ark > $dir/egs/egs_tmp.$n.0.ark || exit 1;
      rm $dir/egs/egs_orig.$n.*.ark  # don't "|| exit 1", due to NFS bugs...
    done
  else # We'll have to split it up using nnet-copy-egs.
    egs_list=
    for n in `seq 0 $[$iters_per_epoch-1]`; do
      egs_list="$egs_list ark:$dir/egs/egs_tmp.JOB.$n.ark"
    done
    # note, the "|| true" below is a workaround for NFS bugs
    # we encountered running this script with Debian-7, NFS-v4.
    $cmd $io_opts JOB=1:$num_jobs_nnet $dir/log/split_egs.JOB.log \
      nnet-copy-egs-discriminative --random=$random_copy --srand=JOB \
        "ark:cat $dir/egs/egs_orig.JOB.*.ark|" $egs_list '&&' \
        '(' rm $dir/egs/egs_orig.JOB.*.ark '||' true ')' || exit 1;
  fi
fi


if [ $stage -le -4 ]; then
  # Next, shuffle the order of the examples in each of those files.
  # Each one should not be too large, so we can do this in memory.
  echo "Shuffling the order of training examples"
  echo "(in order to avoid stressing the disk, these won't all run at once)."

  # note, the "|| true" below is a workaround for NFS bugs
  # we encountered running this script with Debian-7, NFS-v4.
  for n in `seq 0 $[$iters_per_epoch-1]`; do
    $cmd $io_opts JOB=1:$num_jobs_nnet $dir/log/shuffle.$n.JOB.log \
      nnet-shuffle-egs-discriminative "--srand=\$[JOB+($num_jobs_nnet*$n)]" \
      ark:$dir/egs/egs_tmp.JOB.$n.ark ark:$dir/egs/egs.JOB.$n.ark '&&' \
      '(' rm $dir/egs/egs_tmp.JOB.$n.ark '||' true ')' || exit 1;
  done
fi

num_iters=$[$num_epochs * $iters_per_epoch];

echo "$0: Will train for $num_epochs epochs = $num_iters iterations"

exit 0;


# This is when we decide to mix up from: halfway between when we've finished
# adding the hidden layers and the end of training.
finish_add_layers_iter=$[($num_hidden_layers-$initial_num_hidden_layers+1)*$add_layers_period]
mix_up_iter=$[($num_iters + $finish_add_layers_iter)/2]

if [ $num_threads -eq 1 ]; then
  train_suffix="-simple" # this enables us to use GPU code if
                         # we have just one thread.
else
  train_suffix="-parallel --num-threads=$num_threads"
fi



x=0
while [ $x -lt $num_iters ]; do
  if [ $x -ge 0 ] && [ $stage -le $x ]; then
    # Set off jobs doing some diagnostics, in the background.
    $cmd $dir/log/compute_prob_valid.$x.log \
      nnet-compute-prob $dir/$x.mdl ark:$egs_dir/valid_diagnostic.egs &
    $cmd $dir/log/compute_prob_train.$x.log \
      nnet-compute-prob $dir/$x.mdl ark:$egs_dir/train_diagnostic.egs &
    if [ $x -gt 0 ] && [ ! -f $dir/log/mix_up.$[$x-1].log ]; then
      $cmd $dir/log/progress.$x.log \
        nnet-show-progress --use-gpu=no $dir/$[$x-1].mdl $dir/$x.mdl ark:$egs_dir/train_diagnostic.egs &
    fi
    
    echo "Training neural net (pass $x)"
    if [ $x -gt 0 ] && \
      [ $x -le $[($num_hidden_layers-1)*$add_layers_period] ] && \
      [ $[($x-1) % $add_layers_period] -eq 0 ]; then
      mdl="nnet-init --srand=$x $dir/hidden.config - | nnet-insert $dir/$x.mdl - - |"
    else
      mdl=$dir/$x.mdl
    fi


    $cmd $parallel_opts JOB=1:$num_jobs_nnet $dir/log/train.$x.JOB.log \
      nnet-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x \
      ark:$egs_dir/egs.JOB.$[$x%$iters_per_epoch].ark ark:- \| \
      nnet-train$train_suffix \
         --minibatch-size=$minibatch_size --srand=$x "$mdl" \
        ark:- $dir/$[$x+1].JOB.mdl \
      || exit 1;

    nnets_list=
    for n in `seq 1 $num_jobs_nnet`; do
      nnets_list="$nnets_list $dir/$[$x+1].$n.mdl"
    done

    learning_rate=`perl -e '($x,$n,$i,$f)=@ARGV; print ($x >= $n ? $f : $i*exp($x*log($f/$i)/$n));' $[$x+1] $num_iters_reduce $initial_learning_rate $final_learning_rate`;
    softmax_learning_rate=`perl -e "print $learning_rate * $softmax_learning_rate_factor;"`;
    nnet-am-info $dir/$[$x+1].1.mdl > $dir/foo  2>/dev/null || exit 1
    nu=`cat $dir/foo | grep num-updatable-components | awk '{print $2}'`
    na=`cat $dir/foo | grep -v Fixed | grep AffineComponent | wc -l` 
    # na is number of last updatable AffineComponent layer [one-based, counting only
    # updatable components.]
    lr_string="$learning_rate"
    for n in `seq 2 $nu`; do 
      if [ $n -eq $na ] || [ $n -eq $[$na-1] ]; then lr=$softmax_learning_rate;
      else lr=$learning_rate; fi
      lr_string="$lr_string:$lr"
    done
    
    $cmd $dir/log/average.$x.log \
      nnet-am-average $nnets_list - \| \
      nnet-am-copy --learning-rates=$lr_string - $dir/$[$x+1].mdl || exit 1;

    if $shrink && [ $[$x % $shrink_interval] -eq 0 ]; then
      mb=$[($num_frames_shrink+$num_threads-1)/$num_threads]
      $cmd $parallel_opts $dir/log/shrink.$x.log \
        nnet-subset-egs --n=$num_frames_shrink --randomize-order=true --srand=$x \
          ark:$egs_dir/train_diagnostic.egs ark:-  \| \
        nnet-combine-fast --num-threads=$num_threads --verbose=3 --minibatch-size=$mb \
          $dir/$[$x+1].mdl ark:- $dir/$[$x+1].mdl || exit 1;
    else
      # On other iters, do nnet-am-fix which is much faster and has roughly
      # the same effect.
      nnet-am-fix $dir/$[$x+1].mdl $dir/$[$x+1].mdl 2>$dir/log/fix.$x.log 
    fi

    if [ "$mix_up" -gt 0 ] && [ $x -eq $mix_up_iter ]; then
      # mix up.
      echo Mixing up from $num_leaves to $mix_up components
      $cmd $dir/log/mix_up.$x.log \
        nnet-am-mixup --min-count=10 --num-mixtures=$mix_up \
        $dir/$[$x+1].mdl $dir/$[$x+1].mdl || exit 1;
    fi
    rm $nnets_list
  fi
  x=$[$x+1]
done

# Now do combination.
# At the end, final.mdl will be a combination of the last e.g. 10 models.
nnets_list=()
if [ $num_iters_final -gt $num_iters_extra ]; then
  echo "Setting num_iters_final=$num_iters_extra"
fi
start=$[$num_iters-$num_iters_final+1]
for x in `seq $start $num_iters`; do
  idx=$[$x-$start]
  if [ $x -gt $mix_up_iter ]; then
    nnets_list[$idx]=$dir/$x.mdl # "nnet-am-copy --remove-dropout=true $dir/$x.mdl - |"
  fi
done

if [ $stage -le $num_iters ]; then
  # Below, use --use-gpu=no to disable nnet-combine-fast from using a GPU, as
  # if there are many models it can give out-of-memory error; set num-threads to 8
  # to speed it up (this isn't ideal...)
  this_num_threads=$num_threads
  [ $this_num_threads -lt 8 ] && this_num_threads=8
  num_egs=`nnet-copy-egs ark:$egs_dir/combine.egs ark:/dev/null 2>&1 | tail -n 1 | awk '{print $NF}'`
  mb=$[($num_egs+$this_num_threads-1)/$this_num_threads]
  [ $mb -gt 512 ] && mb=512
  $cmd $parallel_opts $dir/log/combine.log \
    nnet-combine-fast --use-gpu=no --num-threads=$this_num_threads \
      --verbose=3 --minibatch-size=$mb "${nnets_list[@]}" ark:$egs_dir/combine.egs \
      $dir/final.mdl || exit 1;
fi

# Compute the probability of the final, combined model with
# the same subset we used for the previous compute_probs, as the
# different subsets will lead to different probs.
$cmd $dir/log/compute_prob_valid.final.log \
  nnet-compute-prob $dir/final.mdl ark:$egs_dir/valid_diagnostic.egs &
$cmd $dir/log/compute_prob_train.final.log \
  nnet-compute-prob $dir/final.mdl ark:$egs_dir/train_diagnostic.egs &

sleep 2

echo Done

if $cleanup; then
  echo Cleaning up data
  if [ $egs_dir == "$dir/egs" ]; then
    echo Removing training examples
    rm $dir/egs/egs*
  fi
  echo Removing most of the models
  for x in `seq 0 $num_iters`; do
    if [ $[$x%10] -ne 0 ] && [ $x -lt $[$num_iters-$num_iters_final+1] ]; then 
       # delete all but every 10th model; don't delete the ones which combine to form the final model.
      rm $dir/$x.mdl
    fi
  done
fi
