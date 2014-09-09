#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey). 
#           2013  Guoguo Chen
#           2014  Xiaohui Zhang
# Apache 2.0.


# This script trains an ensemble of neural networks with pnorm nonlinearities. 
# An ensemble of nets are first differently initialized, and then trained using the 
# same data during each iteration. In each training iteration, one term is added to 
# the objf, which is beta times the cross-entropy between the current net's posterior 
# output and the geometrically averaged posterior outputs of the ensemble of nets. 
# The beta values obey an exponentially increasing schedule (determined by initial_beta
# and final_beta). 

# Begin configuration section.
cmd=run.pl
num_epochs=15      # Number of epochs during which we reduce
                   # the learning rate; number of iteration is worked out from this.
num_epochs_extra=5 # Number of epochs after we stop reducing
                   # the learning rate.
num_iters_final=20 # Maximum number of final iterations to give to the
                   # optimization over the validation set.
initial_learning_rate=0.04
final_learning_rate=0.004
bias_stddev=0.5
softmax_learning_rate_factor=1.0 # In the default setting keep the same learning rate.

combine_regularizer=1.0e-14 # Small regularizer so that parameters won't go crazy.
pnorm_input_dim=3000 
pnorm_output_dim=300
p=2
minibatch_size=128 # by default use a smallish minibatch size for neural net
                   # training; this controls instability which would otherwise
                   # be a problem with multi-threaded update.  Note: it also
                   # interacts with the "preconditioned" update which generally
                   # works better with larger minibatch size, so it's not
                   # completely cost free.

samples_per_iter=200000 # each iteration of training, see this many samples
                        # per job.  This option is passed to get_egs.sh
num_jobs_nnet=16   # Number of neural net jobs to run in parallel.  This option
                   # is passed to get_egs.sh.
get_egs_stage=0
spk_vecs_dir=

shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.

add_layers_period=2 # by default, add new layers every 2 iterations.
num_hidden_layers=3
stage=-5

io_opts="-tc 5" # for jobs with a lot of I/O, limits the number running at one time.   These don't
splice_width=4 # meaning +- 4 frames on each side for second LDA
randprune=4.0 # speeds up LDA.
alpha=4.0
max_change=10.0
mix_up=0 # Number of components to mix up to (should be > #tree leaves, if
        # specified.)
num_threads=16
parallel_opts="-pe smp 16 -l ram_free=1G,mem_free=1G" # by default we use 16 threads; this lets the queue know.
  # note: parallel_opts doesn't automatically get adjusted if you adjust num-threads.
cleanup=true
egs_dir=
lda_opts=
egs_opts=
initial_beta=0.1
final_beta=6
ensemble_size=2
# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: $0 [opts] <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: $0 data/train data/lang exp/tri3_ali exp/tri4_nnet"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-epochs <#epochs|15>                        # Number of epochs of main training"
  echo "                                                   # while reducing learning rate (determines #iterations, together"
  echo "                                                   # with --samples-per-iter and --num-jobs-nnet)"
  echo "  --num-epochs-extra <#epochs-extra|5>             # Number of extra epochs of training"
  echo "                                                   # after learning rate fully reduced"
  echo "  --initial-learning-rate <initial-learning-rate|0.02> # Learning rate at start of training, e.g. 0.02 for small"
  echo "                                                       # data, 0.01 for large data"
  echo "  --final-learning-rate  <final-learning-rate|0.004>   # Learning rate at end of training, e.g. 0.004 for small"
  echo "                                                   # data, 0.001 for large data"
  echo "  --num-hidden-layers <#hidden-layers|2>           # Number of hidden layers, e.g. 2 for 3 hours of data, 4 for 100hrs"
  echo "  --initial-num-hidden-layers <#hidden-layers|1>   # Number of hidden layers to start with."
  echo "  --add-layers-period <#iters|2>                   # Number of iterations between adding hidden layers"
  echo "  --mix-up <#pseudo-gaussians|0>                   # Can be used to have multiple targets in final output layer,"
  echo "                                                   # per context-dependent state.  Try a number several times #states."
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
  echo "  --minibatch-size <minibatch-size|128>            # Size of minibatch to process (note: product with --num-threads"
  echo "                                                   # should not get too large, e.g. >2k)."
  echo "  --samples-per-iter <#samples|400000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --splice-width <width|4>                         # Number of frames on each side to append for feature input"
  echo "                                                   # (note: we splice processed, typically 40-dimensional frames"
  echo "  --lda-dim <dim|250>                              # Dimension to reduce spliced features to with LDA"
  echo "  --num-iters-final <#iters|10>                    # Number of final iterations to give to nnet-combine-fast to "
  echo "                                                   # interpolate parameters (the weights are learned with a validation set)"
  echo "  --num-utts-subset <#utts|300>                    # Number of utterances in subsets used for validation and diagnostics"
  echo "                                                   # (the validation subset is held out from training)"
  echo "  --num-frames-diagnostic <#frames|4000>           # Number of frames used in computing (train,valid) diagnostics"
  echo "  --num-valid-frames-combine <#frames|10000>       # Number of frames used in getting combination weights at the"
  echo "                                                   # very end."
  echo "  --stage <stage|-9>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  
  exit 1;
fi

data=$1
lang=$2
alidir=$3
dir=$4

# Check some files.
for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl $alidir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


# Set some variables.
num_leaves=`tree-info $alidir/tree 2>/dev/null | awk '{print $2}'` || exit 1
[ -z $num_leaves ] && echo "\$num_leaves is unset" && exit 1
[ "$num_leaves" -eq "0" ] && echo "\$num_leaves is 0" && exit 1

nj=`cat $alidir/num_jobs` || exit 1;  # number of jobs in alignment dir...
# in this dir we'll have just one job.
sdata=$data/split$nj
utils/split_data.sh $data $nj

mkdir -p $dir/log
echo $nj > $dir/num_jobs
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
cp $alidir/splice_opts $dir 2>/dev/null
cp $alidir/tree $dir

if [ $stage -le -4 ]; then
  echo "$0: calling get_lda.sh"
  steps/nnet2/get_lda.sh $lda_opts --splice-width $splice_width --cmd "$cmd" $data $lang $alidir $dir || exit 1;
fi

# these files will have been written by get_lda.sh
feat_dim=`cat $dir/feat_dim` || exit 1;
lda_dim=`cat $dir/lda_dim` || exit 1;

if [ $stage -le -3 ] && [ -z "$egs_dir" ]; then
  echo "$0: calling get_egs.sh"
  [ ! -z $spk_vecs_dir ] && spk_vecs_opt="--spk-vecs-dir $spk_vecs_dir";
  steps/nnet2/get_egs.sh $spk_vecs_opt --samples-per-iter $samples_per_iter --num-jobs-nnet $num_jobs_nnet \
      --splice-width $splice_width --stage $get_egs_stage --cmd "$cmd" $egs_opts --io-opts "$io_opts" \
      $data $lang $alidir $dir || exit 1;
fi

if [ -z $egs_dir ]; then
  egs_dir=$dir/egs
fi

iters_per_epoch=`cat $egs_dir/iters_per_epoch`  || exit 1;
! [ $num_jobs_nnet -eq `cat $egs_dir/num_jobs_nnet` ] && \
  echo "$0: Warning: using --num-jobs-nnet=`cat $egs_dir/num_jobs_nnet` from $egs_dir"
num_jobs_nnet=`cat $egs_dir/num_jobs_nnet` || exit 1;


if ! [ $num_hidden_layers -ge 1 ]; then
  echo "Invalid num-hidden-layers $num_hidden_layers"
  exit 1
fi

if [ $stage -le -2 ]; then
  echo "$0: initializing neural net";

  # Get spk-vec dim (in case we're using them).
  if [ ! -z "$spk_vecs_dir" ]; then
    spk_vec_dim=$[$(copy-vector --print-args=false "ark:cat $spk_vecs_dir/vecs.1|" ark,t:- | head -n 1 | wc -w) - 3];
    ! [ $spk_vec_dim -gt 0 ] && echo "Error getting spk-vec dim" && exit 1;
    ext_lda_dim=$[$lda_dim + $spk_vec_dim]
    extend-transform-dim --new-dimension=$ext_lda_dim $dir/lda.mat $dir/lda_ext.mat || exit 1;
    lda_mat=$dir/lda_ext.mat
    ext_feat_dim=$[$feat_dim + $spk_vec_dim]
  else
    spk_vec_dim=0
    lda_mat=$dir/lda.mat
    ext_lda_dim=$lda_dim
    ext_feat_dim=$feat_dim
  fi

  stddev=`perl -e "print 1.0/sqrt($pnorm_input_dim);"`
  cat >$dir/nnet.config <<EOF
SpliceComponent input-dim=$ext_feat_dim left-context=$splice_width right-context=$splice_width const-component-dim=$spk_vec_dim
FixedAffineComponent matrix=$lda_mat
AffineComponentPreconditioned input-dim=$ext_lda_dim output-dim=$pnorm_input_dim alpha=$alpha max-change=$max_change learning-rate=$initial_learning_rate param-stddev=$stddev bias-stddev=$bias_stddev
PnormComponent input-dim=$pnorm_input_dim output-dim=$pnorm_output_dim p=$p
NormalizeComponent dim=$pnorm_output_dim
AffineComponentPreconditioned input-dim=$pnorm_output_dim output-dim=$num_leaves alpha=$alpha max-change=$max_change learning-rate=$initial_learning_rate param-stddev=0 bias-stddev=0
SoftmaxComponent dim=$num_leaves
EOF

  # to hidden.config it will write the part of the config corresponding to a
  # single hidden layer; we need this to add new layers. 
  cat >$dir/hidden.config <<EOF
AffineComponentPreconditioned input-dim=$pnorm_output_dim output-dim=$pnorm_input_dim alpha=$alpha max-change=$max_change learning-rate=$initial_learning_rate param-stddev=$stddev bias-stddev=$bias_stddev
PnormComponent input-dim=$pnorm_input_dim output-dim=$pnorm_output_dim p=$p
NormalizeComponent dim=$pnorm_output_dim
EOF
  for i in `seq 1 $ensemble_size`; do
    $cmd $parallel_opts JOB=1:$ensemble_size $dir/log/nnet_init.JOB.log \
      nnet-am-init $alidir/tree $lang/topo "nnet-init --srand=JOB $dir/nnet.config -|" \
      $dir/0.JOB.mdl || exit 1;
  done
fi

if [ $stage -le -1 ]; then
  echo "Training transition probabilities and setting priors"
  $cmd $parallel_opts JOB=1:$ensemble_size $dir/log/train_trans.JOB.log \
      nnet-train-transitions $dir/0.JOB.mdl "ark:gunzip -c $alidir/ali.*.gz|" $dir/0.JOB.mdl \
      || exit 1;
fi

num_iters_reduce=$[$num_epochs * $iters_per_epoch];
num_iters_extra=$[$num_epochs_extra * $iters_per_epoch];
num_iters=$[$num_iters_reduce+$num_iters_extra]

echo "$0: Will train for $num_epochs + $num_epochs_extra epochs, equalling "
echo "$0: $num_iters_reduce + $num_iters_extra = $num_iters iterations, "
echo "$0: (while reducing learning rate) + (with constant learning rate)."

# This is when we decide to mix up from: halfway between when we've finished
# adding the hidden layers and the end of training.
finish_add_layers_iter=$[$num_hidden_layers*$add_layers_period]
mix_up_iter=$[($num_iters + $finish_add_layers_iter)/2]

x=0

while [ $x -lt $num_iters ]; do
  if [ $x -ge 0 ] && [ $stage -le $x ]; then
    # Set off jobs doing some diagnostics, in the background.
    $cmd $dir/log/compute_prob_valid.$x.log \
      nnet-compute-prob $dir/$x.1.mdl ark:$egs_dir/valid_diagnostic.egs &
    $cmd $dir/log/compute_prob_train.$x.log \
      nnet-compute-prob $dir/$x.1.mdl ark:$egs_dir/train_diagnostic.egs &
    if [ $x -gt 0 ] && [ ! -f $dir/log/mix_up.$[$x-1].log ]; then
      $cmd $dir/log/progress.$x.log \
        nnet-show-progress --use-gpu=no $dir/$[$x-1].1.mdl $dir/$x.1.mdl ark:$egs_dir/train_diagnostic.egs &
    fi
    
    declare -A mdl
    echo "Training neural net (pass $x)"
    if [ $x -gt 0 ] && \
      [ $x -le $[($num_hidden_layers-1)*$add_layers_period] ] && \
      [ $[($x-1) % $add_layers_period] -eq 0 ]; then
      for i in `seq 1 $ensemble_size`; do
        mdl[$i]="nnet-init --srand=$[$x+$i] $dir/hidden.config - | nnet-insert $dir/$x.$i.mdl - - |"
      done
    else
      for i in `seq 1 $ensemble_size`; do
        mdl[$i]=$dir/$x.$i.mdl
      done
    fi

    nnets_ensemble_in=
    nnets_ensemble_out=
    for i in `seq 1 $ensemble_size`; do
      nnets_ensemble_in="$nnets_ensemble_in '${mdl[$i]}'"
      nnets_ensemble_out="${nnets_ensemble_out} $dir/$[$x+1].JOB.$i.mdl "
    done

    beta=`perl -e '($x,$n,$i,$f)=@ARGV; print ($i+$x*($f-$i)/$n);' $[$x+1] $num_iters $initial_beta $final_beta`; 

    $cmd $parallel_opts JOB=1:$num_jobs_nnet $dir/log/train.$x.JOB.log \
      nnet-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x \
      ark:$egs_dir/egs.JOB.$[$x%$iters_per_epoch].ark ark:- \| \
      nnet-train-ensemble \
         --minibatch-size=$minibatch_size --srand=$x --beta=$beta $nnets_ensemble_in \
        ark:- $nnets_ensemble_out \
      || exit 1;

    learning_rate=`perl -e '($x,$n,$i,$f)=@ARGV; print ($x >= $n ? $f : $i*exp($x*log($f/$i)/$n));' $[$x+1] $num_iters_reduce $initial_learning_rate $final_learning_rate`;
    softmax_learning_rate=`perl -e "print $learning_rate * $softmax_learning_rate_factor;"`;
    nnet-am-info $dir/$[$x+1].1.1.mdl > $dir/foo  2>/dev/null || exit 1
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
    
    for i in `seq 1 $ensemble_size`; do 
      nnets_list=
      for n in `seq 1 $num_jobs_nnet`; do
        nnets_list="$nnets_list $dir/$[$x+1].$n.$i.mdl"
      done
      $cmd $dir/log/average.$x.$i.log \
        nnet-am-average $nnets_list - \| \
        nnet-am-copy --learning-rates=$lr_string - $dir/$[$x+1].$i.mdl || exit 1;
      rm $nnets_list
      if [ "$mix_up" -gt 0 ] && [ $x -eq $mix_up_iter ]; then
        # mix up.
        echo Mixing up from $num_leaves to $mix_up components
        $cmd $dir/log/mix_up.$x.$i.log \
          nnet-am-mixup --min-count=10 --num-mixtures=$mix_up \
          $dir/$[$x+1].$i.mdl $dir/$[$x+1].$i.mdl || exit 1;
      fi
    done
  fi
  x=$[$x+1]
done

# Now do combination.
# At the end, final.mdl will be a combination of the last e.g. 10 models.

for i in `seq 1 $ensemble_size`; do 
  nnets_list=()
  if [ $num_iters_final -gt $num_iters_extra ]; then
    echo "Setting num_iters_final=$num_iters_extra"
  fi
  start=$[$num_iters-$num_iters_final+1]
  for x in `seq $start $num_iters`; do
    idx=$[$x-$start]
    if [ $x -gt $mix_up_iter ]; then
      nnets_list[$idx]=$dir/$x.$i.mdl # "nnet-am-copy --remove-dropout=true $dir/$x.mdl - |"
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
    # Setting --initial-model to a large value makes it initialize the combination
    # with the average of all the models.  It's important not to start with a
    # single model, or, due to the invariance to scaling that these nonlinearities
    # give us, we get zero diagonal entries in the fisher matrix that
    # nnet-combine-fast uses for scaling, which after flooring and inversion, has
    # the effect that the initial model chosen gets much higher learning rates
    # than the others.  This prevents the optimization from working well.
    $cmd $parallel_opts $dir/log/combine.$i.log \
      nnet-combine-fast --initial-model=100000 --num-lbfgs-iters=40 --use-gpu=no \
        --num-threads=$this_num_threads --regularizer=$combine_regularizer \
        --initial-model=100000 --num-lbfgs-iters=40 \
        --verbose=3 --minibatch-size=$mb "${nnets_list[@]}" ark:$egs_dir/combine.egs \
        $dir/final.$i.mdl || exit 1;
  
    # Normalize stddev for affine or block affine layers that are followed by a
    # pnorm layer and then a normalize layer.
    $cmd $parallel_opts $dir/log/normalize.$i.log \
      nnet-normalize-stddev $dir/final.$i.mdl $dir/final.$i.mdl || exit 1;
  fi
  # Compute the probability of the final, combined model with
  # the same subset we used for the previous compute_probs, as the
  # different subsets will lead to different probs.
  $cmd $dir/log/compute_prob_valid.final.$i.log \
    nnet-compute-prob $dir/final.$i.mdl ark:$egs_dir/valid_diagnostic.egs &
  $cmd $dir/log/compute_prob_train.final.$i.log \
    nnet-compute-prob $dir/final.$i.mdl ark:$egs_dir/train_diagnostic.egs &
done
cp $dir/final.1.mdl $dir/final.mdl

sleep 2

echo Done

if $cleanup; then
  echo Cleaning up data
  if [ $egs_dir == "$dir/egs" ]; then
    steps/nnet2/remove_egs.sh $dir/egs
  fi
  echo Removing most of the models
  for x in `seq 0 $num_iters`; do
    if [ $[$x%10] -ne 0 ] && [ $x -lt $[$num_iters-$num_iters_final+1] ]; then 
       # delete all but every 10th model; don't delete the ones which combine to form the final model.
      for i in `seq 1 $ensemble_size`; do 
        rm $dir/$x.$i.mdl
      done
    fi
  done
fi
