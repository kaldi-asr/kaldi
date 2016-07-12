#!/bin/bash

# note, TDNN is the same as what we used to call multisplice.

# Copyright 2012-2015  Johns Hopkins University (Author: Daniel Povey).
#           2013  Xiaohui Zhang
#           2013  Guoguo Chen
#           2014  Vimal Manohar
#           2014  Vijayaditya Peddinti
#           2016  Pegah Ghahremani
# Apache 2.0.


# Begin configuration section.
cmd=run.pl
num_epochs=15      # Number of epochs of training;
                   # the number of iterations is worked out from this.
print_interval=100 # print interval used to output objectives during training.
use_ivector=false
initial_effective_lrate=0.01
final_effective_lrate=0.001
pnorm_input_dim=3000
pnorm_output_dim=300
relu_dim=  # you can use this to make it use ReLU's instead of p-norms.
rand_prune=4.0 # Relates to a speedup we do for LDA.
minibatch_size=512  # This default is suitable for GPU-based training.
                    # Set it to 128 for multi-threaded CPU-based training.
max_param_change=2.0  # max param change per minibatch
samples_per_iter=400000 # each iteration of training, see this many samples
                        # per job.  This option is passed to get_egs.sh
num_jobs_initial=1  # Number of neural net jobs to run in parallel at the start of training
num_jobs_final=8   # Number of neural net jobs to run in parallel at the end of training
prior_subset_size=20000 # 20k samples per job, for computing priors.
num_jobs_compute_prior=10 # these are single-threaded, run on CPU.
get_egs_stage=0    # can be used for rerunning after partial
online_ivector_dir=
presoftmax_prior_scale_power=-0.25
use_presoftmax_prior_scale=true
remove_egs=true  # set to false to disable removing egs after training is done.

max_models_combine=20 # The "max_models_combine" is the maximum number of models we give
  # to the final 'combine' stage, but these models will themselves be averages of
  # iteration-number ranges.

shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.
                # (the point of this is to get data in different minibatches on different iterations,
                # since in the preconditioning method, 2 samples in the same minibatch can
                # affect each others' gradients.

add_layers_period=2 # by default, add new layers every 2 iterations.
stage=-6
exit_stage=-100 # you can set this to terminate the training early.  Exits before running this stage

# count space-separated fields in splice_indexes to get num-hidden-layers.
splice_indexes="-4,-3,-2,-1,0,1,2,3,4  0  -2,2  0  -4,4 0"
# Format : layer<hidden_layer>/<frame_indices>....layer<hidden_layer>/<frame_indices> "
# note: hidden layers which are composed of one or more components,
# so hidden layer indexing is different from component count

randprune=4.0 # speeds up LDA.
use_gpu=true    # if true, we run on GPU.
cleanup=false
egs_dir=
max_lda_jobs=10  # use no more than 10 jobs for the LDA accumulation.
lda_opts=
egs_opts=
transform_dir=     # If supplied, this dir used instead of alidir to find transforms.
cmvn_opts=  # will be passed to get_lda.sh and get_egs.sh, if supplied.
            # only relevant for "raw" features, not lda.
feat_type=raw  # or set to 'lda' to use LDA features.
align_cmd=              # The cmd that is passed to steps/nnet2/align.sh
align_use_gpu=          # Passed to use_gpu in steps/nnet2/align.sh [yes/no]
realign_times=          # List of times on which we realign.  Each time is
                        # floating point number strictly between 0 and 1, which
                        # will be multiplied by the num-iters to get an iteration
                        # number.
num_jobs_align=30       # Number of jobs for realignment
# End configuration section.
frames_per_eg=8 # to be passed on to get_egs.sh

trap 'for pid in $(jobs -pr); do kill -KILL $pid; done' INT QUIT TERM

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ]; then
  echo "Usage: $0 [opts] <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: $0 data/train data/lang exp/tri3_ali exp/tri4_nnet"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-epochs <#epochs|15>                        # Number of epochs of training"
  echo "  --initial-effective-lrate <lrate|0.02> # effective learning rate at start of training."
  echo "  --final-effective-lrate <lrate|0.004>   # effective learning rate at end of training."
  echo "                                                   # data, 0.00025 for large data"
  echo "  --num-hidden-layers <#hidden-layers|2>           # Number of hidden layers, e.g. 2 for 3 hours of data, 4 for 100hrs"
  echo "  --add-layers-period <#iters|2>                   # Number of iterations between adding hidden layers"
  echo "  --presoftmax-prior-scale-power <power|-0.25>     # use the specified power value on the priors (inverse priors) to scale"
  echo "                                                   # the pre-softmax outputs (set to 0.0 to disable the presoftmax element scale)"
  echo "  --num-jobs-initial <num-jobs|1>                  # Number of parallel jobs to use for neural net training, at the start."
  echo "  --num-jobs-final <num-jobs|8>                    # Number of parallel jobs to use for neural net training, at the end"
  echo "  --num-threads <num-threads|16>                   # Number of parallel threads per job, for CPU-based training (will affect"
  echo "                                                   # results as well as speed; may interact with batch size; if you increase"
  echo "                                                   # this, you may want to decrease the batch size."
  echo "  --parallel-opts <opts|\"-pe smp 16 -l ram_free=1G,mem_free=1G\">      # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads... note, you might have to reduce mem_free,ram_free"
  echo "                                                   # versus your defaults, because it gets multiplied by the -pe smp argument."
  echo "  --minibatch-size <minibatch-size|128>            # Size of minibatch to process (note: product with --num-threads"
  echo "                                                   # should not get too large, e.g. >2k)."
  echo "  --samples-per-iter <#samples|400000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --splice-indexes <string|layer0/-4:-3:-2:-1:0:1:2:3:4> "
  echo "                                                   # Frame indices used for each splice layer."
  echo "                                                   # Format : layer<hidden_layer_index>/<frame_indices>....layer<hidden_layer>/<frame_indices> "
  echo "                                                   # (note: we splice processed, typically 40-dimensional frames"
  echo "  --lda-dim <dim|''>                               # Dimension to reduce spliced features to with LDA"
  echo "  --realign-times <list-of-times|\"\">             # A list of space-separated floating point numbers between 0.0 and"
  echo "                                                   # 1.0 to specify how far through training realignment is to be done"
  echo "  --align-cmd (utils/run.pl|utils/queue.pl <queue opts>) # passed to align.sh"
  echo "  --align-use-gpu (yes/no)                         # specify is gpu is to be used for realignment"
  echo "  --num-jobs-align <#njobs|30>                     # Number of jobs to perform realignment"
  echo "  --stage <stage|-4>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."


  exit 1;
fi
#data=$1
#lang=$2
#alidir=$3
args=("$@")
dir=${args[-1]}
unset args[${#args[@]}-1]
num_lang=$[${#args[@]}/3]

for i in `seq 0 $[$num_lang-1]`; do
  langs[$i]=${args[$i]} 
  alidirs[$i]=${args[$i+$num_lang]}
  common_egs_dir[$i]=${args[$i+2*$num_lang]}
  langdirs[$i]=$(dirname ${common_egs_dir[$i]})
done

unset args
echo num_lang = $num_lang

echo common_egs_dir = ${common_egs_dir[0]} and  ${common_egs_dir[1]} 
echo alidirs = ${alidirs[0]} and  ${alidirs[1]}


# sourcing the "vars" below sets
# left_context=(something)
# right_context=(something)
# num_hidden_layers=(something)
. $dir/configs/vars || exit 1;


context_opts="--left-context=$left_context --right-context=$right_context"

! [ "$num_hidden_layers" -gt 0 ] && echo \
 "$0: Expected num_hidden_layers to be defined" && exit 1;

# copy any of the following that exist, to $dir.
cp ${common_egs_dir[0]}/{cmvn_opts,splice_opts} $dir 2>/dev/null

# confirm that the egs_dir has the necessary context (especially important if
# the --egs-dir option was used on the command line).
egs_left_context=$(cat ${common_egs_dir[0]}/info/left_context) || exit -1
egs_right_context=$(cat ${common_egs_dir[0]}/info/right_context) || exit -1
 ( [ $egs_left_context -lt $left_context ] || \
   [ $egs_right_context -lt $right_context ] ) && \
   echo "$0: egs in $egs_dir have too little context" && exit -1;

min_num_archives_expanded=1000
max_num_archives_expanded=-1000
min_frames=$(cat ${common_egs_dir[0]}/info/num_frames || { echo "error: no such file $egs_dir/info/num_frames"; exit 1; })
for lang in `seq 0 $[$num_lang-1]`; do 
  egs_dir=${common_egs_dir[$lang]}
  echo egs_dir = $egs_dir out of $num_lang languages
  frames_per_eg[$lang]=$(cat $egs_dir/info/frames_per_eg) || { echo "error: no such file $egs_dir/info/frames_per_eg"; exit 1; }
  num_archives[$lang]=$(cat $egs_dir/info/num_archives) || { echo "error: no such file $egs_dir/info/frames_per_eg"; exit 1; }
  num_frames[$lang]=$(cat $egs_dir/info/num_frames || { echo "error: no such file $egs_dir/info/num_frames"; exit 1; })
  if [ ${num_frames[$lang]} -lt $min_frames ]; then
    min_frames=${num_frames[$lang]}
  fi
  tot_frames=$[$tot_frames+$(cat $egs_dir/info/num_frames || { echo "error: no such file $egs_dir/info/num_frames"; exit 1; })]

# num_archives_expanded considers each separate label-position from
# 0..frames_per_eg-1 to be a separate archive.
  num_archives_expanded[$lang]=$[${num_archives[$lang]}*$frames_per_eg]
  
  if [ ${num_archives_expanded[$lang]} -lt $min_num_archives_expanded ]; then
    min_num_archives_expanded=${num_archives_expanded[$lang]}
  fi
  
  if [ ${num_archives_expanded[$lang]} -gt $max_num_archives_expanded ]; then
    max_num_archives_expanded=${num_archives_expanded[$lang]}
  fi
  
done
echo tot_frames = $tot_frames
# Probability distribution used to read egs of different languages randomly.
# w.r.t their number of frame per language and egs from larger languages read
# more frequently.
read_input_prob=`perl -e "printf('%.3f', $(cat ${common_egs_dir[0]}/info/num_frames)/$tot_frames)"`
 
for lang in `seq 1 $[$num_lang-1]`; do
    egs_dir=${common_egs_dir[$lang]}
    new_prob=`perl -e "printf('%.3f', $(cat ${common_egs_dir[$lang]}/info/num_frames)/$tot_frames)"`
    read_input_prob="$read_input_prob,$new_prob"
done 

echo read_input_prob = $read_input_prob
# set num_iters so that as close as possible, we process the data $num_epochs
# times, i.e. $num_iters*$avg_num_jobs) == $num_epochs*$num_archives_expanded,
# where avg_num_jobs=(num_jobs_initial+num_jobs_final)/2.

num_archives_to_process=$[$num_epochs*($max_num_archives_expanded+$min_num_archives_expanded)/2]
echo num_epochs, min_archive and max_archive and num_archive_to_process = $num_epochs ,  $min_num_archives_expanded and $max_num_archives_expanded and $num_archives_to_process
num_archives_processed=0
num_iters=$[($num_archives_to_process*2)/($num_jobs_initial+$num_jobs_final)]

[ $num_jobs_initial -gt $num_jobs_final ] && \
  echo "$0: --initial-num-jobs cannot exceed --final-num-jobs" && exit 1;

[ $num_jobs_final -gt $max_num_archives_expanded ] && \
  echo "$0: --final-num-jobs cannot exceed min #archives $min_num_archives_expanded." && exit 1;

if [ $stage -le -1 ]; then
  # Add the first layer; this will add in the lda.mat and
  # presoftmax_prior_scale.vec.
  $cmd $dir/log/add_first_layer.log \
       nnet3-init --srand=-3 $dir/init.raw $dir/configs/layer1.config $dir/0.raw || exit 1;

fi


! [ $num_iters -gt $[$finish_add_layers_iter+2] ] \
  && echo "$0: Insufficient epochs" && exit 1

finish_add_layers_iter=$[$num_hidden_layers * $add_layers_period]

echo "$0: Will train for $num_epochs epochs = $num_iters iterations"

if $use_gpu; then
  parallel_suffix=""
  train_queue_opt="--gpu 1"
  combine_queue_opt="--gpu 1"
  prior_gpu_opt="--use-gpu=yes"
  prior_queue_opt="--gpu 1"
  parallel_train_opts=
  if ! cuda-compiled; then
    echo "$0: WARNING: you are running with one thread but you have not compiled"
    echo "   for CUDA.  You may be running a setup optimized for GPUs.  If you have"
    echo "   GPUs and have nvcc installed, go to src/ and do ./configure; make"
    exit 1
  fi
else
  echo "$0: without using a GPU this will be very slow.  nnet3 does not yet support multiple threads."
  parallel_train_opts="--use-gpu=no"
  combine_queue_opt=""  # the combine stage will be quite slow if not using
                        # GPU, as we didn't enable that program to use
                        # multiple threads.
  prior_gpu_opt="--use-gpu=no"
  prior_queue_opt=""
fi


approx_iters_per_epoch_final=$[($max_num_archives_expanded+$min_num_archives_expanded)/(2*$num_jobs_final)]
# First work out how many iterations we want to combine over in the final
# nnet3-combine-fast invocation.  (We may end up subsampling from these if the
# number exceeds max_model_combine).  The number we use is:
# min(max(max_models_combine, approx_iters_per_epoch_final),
#     1/2 * iters_after_last_layer_added)
num_iters_combine=$max_models_combine
if [ $num_iters_combine -lt $approx_iters_per_epoch_final ]; then
   num_iters_combine=$approx_iters_per_epoch_final
fi
half_iters_after_add_layers=$[($num_iters-$finish_add_layers_iter)/2]
if [ $num_iters_combine -gt $half_iters_after_add_layers ]; then
  num_iters_combine=$half_iters_after_add_layers
fi
first_model_combine=$[$num_iters-$num_iters_combine+1]

x=0

for realign_time in $realign_times; do
  # Work out the iterations on which we will re-align, if the --realign-times
  # option was used.  This is slightly approximate.
  ! perl -e "exit($realign_time > 0.0 && $realign_time < 1.0 ? 0:1);" && \
    echo "Invalid --realign-times option $realign_times: elements must be strictly between 0 and 1.";
  # the next formula is based on the one for mix_up_iter above.
  realign_iter=$(perl -e '($j,$k,$n,$p)=@ARGV; print int(0.5 + ($j==$k ? $n*$p : $n*(sqrt((1-$p)*$j*$j+$p*$k*$k)-$j)/($k-$j))); ' $num_jobs_initial $num_jobs_final $num_iters $realign_time) || exit 1;
  realign_this_iter[$realign_iter]=$realign_time
done

cur_egs_dir=$egs_dir

while [ $x -lt $num_iters ]; do
  [ $x -eq $exit_stage ] && echo "$0: Exiting early due to --exit-stage $exit_stage" && exit 0;

  this_num_jobs=$(perl -e "print int(0.5+$num_jobs_initial+($num_jobs_final-$num_jobs_initial)*$x/$num_iters);")

  ilr=$initial_effective_lrate; flr=$final_effective_lrate; np=$num_archives_processed; nt=$num_archives_to_process;
  echo num_iters = $num_iters and flr = $flr , ilr $ilr and nt = $nt and this_num_jobs = $this_num_jobs and np = $np
  this_learning_rate=$(perl -e "print (($x + 1 >= $num_iters ? $flr : $ilr*exp($np*log($flr/$ilr)/$nt))*$this_num_jobs);");

  echo "On iteration $x, learning rate is $this_learning_rate."

  if [ ! -z "${realign_this_iter[$x]}" ]; then
    prev_egs_dir=$cur_egs_dir
    cur_egs_dir=$dir/egs_${realign_this_iter[$x]}
  fi

  if [ $x -ge 0 ] && [ $stage -le $x ]; then
    # Set off jobs doing some diagnostics, in the background.
    # Use the egs dir from the previous iteration for the diagnostics
    for i in `seq 0 $[$num_lang-1]`;do
      rename_io_names="output-$i/output"

      $cmd $dir/${langs[$i]}/log/compute_prob_valid.$x.log \
        nnet3-compute-prob "nnet3-copy --rename-node-names='$rename_io_names' $dir/$x.raw - |" \
              "ark:nnet3-merge-egs ark:${common_egs_dir[$i]}/valid_diagnostic.egs ark:- |" &
      $cmd $dir/${langs[$i]}/log/compute_prob_train.$x.log \
        nnet3-compute-prob "nnet3-copy --rename-node-names='$rename_io_names' $dir/$x.raw - |" \
             "ark:nnet3-merge-egs ark:${common_egs_dir[$i]}/train_diagnostic.egs ark:- |" &
    done
    if [ $x -gt 0 ]; then
      rename_io_names="output-0/output"

      $cmd $dir/log/progress.$x.log \
        nnet3-show-progress --use-gpu=no "nnet3-copy --rename-node-names='$rename_io_names' $dir/$[$x-1].raw - |"  \
        "nnet3-copy --rename-node-names='$rename_io_names' $dir/$x.raw - |" \
        "ark:nnet3-merge-egs ark:${common_egs_dir[0]}/train_diagnostic.egs ark:-|" '&&' \
        nnet3-info $dir/$x.raw &
    fi

    echo "Training neural net (pass $x)"

    if [ $x -gt 0 ] && \
      [ $x -le $[($num_hidden_layers-1)*$add_layers_period] ] && \
      [ $[$x%$add_layers_period] -eq 0 ]; then
      do_average=false # if we've just mixed up, don't do averaging but take the
                       # best.
      cur_num_hidden_layers=$[1+$x/$add_layers_period]
      config=$dir/configs/layer$cur_num_hidden_layers.config
      raw="nnet3-copy --learning-rate=$this_learning_rate $dir/$x.raw - | nnet3-init --srand=$x - $config - |"
    else
      do_average=true
      if [ $x -eq 0 ]; then do_average=false; fi # on iteration 0, pick the best, don't average.
      raw="nnet3-copy --learning-rate=$this_learning_rate $dir/$x.raw -|"
    fi
    if $do_average; then
      this_minibatch_size=$minibatch_size
    else
      # on iteration zero or when we just added a layer, use a smaller minibatch
      # size (and we will later choose the output of just one of the jobs): the
      # model-averaging isn't always helpful when the model is changing too fast
      # (i.e. it can worsen the objective function), and the smaller minibatch
      # size will help to keep the update stable.
      this_minibatch_size=$[$minibatch_size/2];
    fi

    rm $dir/.error 2>/dev/null


    ( # this sub-shell is so that when we "wait" below,
      # we only wait for the training jobs that we just spawned,
      # not the diagnostic jobs that we spawned above.

      # We can't easily use a single parallel SGE job to do the main training,
      # because the computation of which archive and which --frame option
      # to use for each job is a little complex, so we spawn each one separately.
      for n in $(seq $this_num_jobs); do
        egs_list=""
        k=$[$num_archives_processed + $n - 1]; # k is a zero-based index that we'll derive
                                               # the other indexes from.
        for lang in `seq 0 $[$num_lang-1]`; do
          num_archive=${num_archives[$lang]}
          archive=$[($k%$num_archive)+1]; # work out the 1-based archive index.
          frame=$[(($k/$num_archive)%$frames_per_eg)]; # work out the 0-based frame
          cur_egs_dir=${common_egs_dir[$lang]}
          rename_io_names="output/output-$lang"

          egs_list="$egs_list 'ark:nnet3-copy-multiple-egs --rename-io-names='$rename_io_names' --frame=$frame $context_opts ark:$cur_egs_dir/egs.$archive.ark ark:- | nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x ark:- ark:-| nnet3-merge-egs --minibatch-size=$this_minibatch_size --discard-partial-minibatches=true ark:- ark:- |'" 
        done
        # index; this increases more slowly than the archive index because the
        # same archive with different frame indexes will give similar gradients,
        # so we want to separate them in time.
        cache_read_opt=""
        if [ $x -gt $[($num_hidden_layers-1)*$add_layers_period] ]; then
          cache_read_opt="--read-cache=$dir/cache.$[$x-1]"
        fi
        
        cache_write_opt=""
        if [ $n == 1 ]; then
          cache_write_opt="--write-cache=$dir/cache.$x"
        fi
      
        $cmd $train_queue_opt $dir/log/train.$x.$n.log \
          nnet3-train $parallel_train_opts $cache_read_opt $cache_write_opt \
          --print-interval=$print_interval \
          --max-param-change=$max_param_change "$raw" \
          "ark:nnet3-copy-multiple-egs --read-input-prob=$read_input_prob --num-input-egs=$num_lang $egs_list ark:- |" \
          $dir/$[$x+1].$n.raw || touch $dir/.error &
      done
      wait
    )
    # the error message below is not that informative, but $cmd will
    # have printed a more specific one.
    [ -f $dir/.error ] && echo "$0: error on iteration $x of training" && exit 1;

    nnets_list=
    for n in `seq 1 $this_num_jobs`; do
      nnets_list="$nnets_list $dir/$[$x+1].$n.raw"
    done

    if $do_average; then
      # average the output of the different jobs.
      $cmd $dir/log/average.$x.log \
        nnet3-average $nnets_list - \| \
        nnet3-copy --set-nnet=- $dir/$x.raw $dir/$[$x+1].raw || exit 1;
    else
      # choose the best from the different jobs.
      n=$(perl -e '($nj,$pat)=@ARGV; $best_n=1; $best_logprob=-1.0e+10; for ($n=1;$n<=$nj;$n++) {
          $fn = sprintf($pat,$n); open(F, "<$fn") || die "Error opening log file $fn";
          undef $logprob; while (<F>) { if (m/log-prob-per-frame=(\S+)/) { $logprob=$1; } }
          close(F); if (defined $logprob && $logprob > $best_logprob) { $best_logprob=$logprob;
          $best_n=$n; } } print "$best_n\n"; ' $num_jobs_nnet $dir/log/train.$x.%d.log) || exit 1;
      [ -z "$n" ] && echo "Error getting best model" && exit 1;
      $cmd $dir/log/select.$x.log \
        nnet3-copy --set-nnet=$dir/$[$x+1].$n.raw  $dir/$x.raw $dir/$[$x+1].raw || exit 1;
    fi

    rm $nnets_list
    [ ! -f $dir/$[$x+1].raw ] && exit 1;
    if [ -f $dir/$[$x-1].raw ] && $cleanup && \
       [ $[($x-1)%100] -ne 0  ] && [ $[$x-1] -lt $first_model_combine ]; then
      rm $dir/$[$x-1].raw
    fi
  fi
  x=$[$x+1]
  num_archives_processed=$[$num_archives_processed+$this_num_jobs]
done


if [ $stage -le $num_iters ]; then
  echo "Doing final combination to produce final.raw"

  # Now do combination.  In the nnet3 setup, the logic
  # for doing averaging of subsets of the models in the case where
  # there are too many models to reliably esetimate interpolation
  # factors (max_models_combine) is moved into the nnet3-combine
  nnets_list=()
  for n in $(seq 0 $[num_iters_combine-1]); do
    iter=$[$first_model_combine+$n]
    mdl=$dir/$iter.raw
    [ ! -f $mdl ] && echo "Expected $mdl to exist" && exit 1;
    nnets_list[$n]="nnet3-copy $mdl -|";
  done
  
  # combined.egs for all egs.
  combined_egs_list=
  for lang in `seq 0 $[$num_lang-1]`; do
    egs_dir=${common_egs_dir[$lang]}
    rename_io_names="output/output-$lang"

    combined_egs_list="$combined_egs_list 'ark:nnet3-copy-multiple-egs --rename-io-names='$rename_io_names' ark:$egs_dir/combine.egs ark:- | nnet3-merge-egs --minibatch-size=1024 ark:- ark:- |'"
  done
  # Below, we use --use-gpu=no to disable nnet3-combine-fast from using a GPU,
  # as if there are many models it can give out-of-memory error; and we set
  # num-threads to 8 to speed it up (this isn't ideal...)
  
    
  $cmd $combine_queue_opt $dir/log/combine.log \
    nnet3-combine --num-iters=40 \
       --enforce-sum-to-one=true --enforce-positive-weights=true \
       --verbose=3 "${nnets_list[@]}" "ark:nnet3-copy-multiple-egs --num-input-egs=$num_lang --read-input-prob=$read_input_prob $combined_egs_list ark:- |" \
    "|nnet3-copy --set-nnet=- $dir/$num_iters.raw $dir/combined.raw" || exit 1;
  
  # Compute the probability of the final, combined model with
  # the same subset we used for the previous compute_probs, as the
  # different subsets will lead to different probs.
  for lang in `seq 0 $[$num_lang-1]`;do
    rename_io_names="output-${lang}/output"
    $cmd $dir/${langs[$lang]}/log/compute_prob_valid.final.log \
      nnet3-compute-prob "nnet3-copy --rename-node-names='$rename_io_names' $dir/combined.raw - |" \
        "ark:nnet3-merge-egs ark:${common_egs_dir[$lang]}/valid_diagnostic.egs ark:- |" &

    $cmd $dir/${langs[$lang]}/log/compute_prob_train.final.log \
      nnet3-compute-prob "nnet3-copy --rename-node-names='$rename_io_names' $dir/combined.raw - |" \
        "ark:nnet3-merge-egs ark:${common_egs_dir[$lang]}/train_diagnostic.egs ark:- |" &
  done
fi

nnet3-copy $dir/combined.raw $dir/final.raw

if [ ! -f $dir/final.raw ]; then
  echo "$0: $dir/final.raw does not exist."
  # we don't want to clean up if the training didn't succeed.
  exit 1;
fi

sleep 2

echo Done

for lang in `seq 0 $[$num_lang-1]`; do
  # Convert to .mdl, train the transitions, set the priors.
  rename_io_names="output-$lang/output"

  $cmd $dir/${langs[$lang]}/log/init_mdl.log \
    nnet3-am-init ${alidirs[$lang]}/final.mdl \
      "nnet3-copy --rename-node-names='$rename_io_names' $dir/final.raw - |"  - \| \
      nnet3-am-train-transitions - "ark:gunzip -c ${alidirs[$lang]}/ali.*.gz|" $dir/${langs[$lang]}/final.mdl || exit 1;

  echo "Getting average posterior for purposes of adjusting the priors."
  # Note: this just uses CPUs, using a smallish subset of data.
  # always use the first egs archive, which makes the script simpler;
  # we're using different random subsets of it.
  rm $dir/post.final.*.vec 2>/dev/null
  $cmd JOB=1:$num_jobs_compute_prior $prior_queue_opt $dir/${langs[$lang]}/log/get_post.final.JOB.log \
    nnet3-copy-egs --srand=JOB --frame=random $context_opts ark:${common_egs_dir[$lang]}/egs.1.ark ark:- \| \
    nnet3-subset-egs --srand=JOB --n=$prior_subset_size ark:- ark:- \| \
    nnet3-merge-egs ark:- ark:- \| \
    nnet3-compute-from-egs --apply-exp=true "nnet3-copy --rename-node-names='$rename_io_names' $dir/final.raw -|" ark:- ark:- \| \
    matrix-sum-rows ark:- ark:- \| vector-sum ark:- $dir/${langs[$lang]}/post.final.JOB.vec || exit 1;

  sleep 3;  # make sure there is time for $dir/post.$x.*.vec to appear.

  $cmd $dir/${langs[$lang]}/log/vector_sum.final.log \
    vector-sum $dir/${langs[$lang]}/post.final.*.vec $dir/${langs[$lang]}/post.final.vec || exit 1;
  rm $dir/${langs[$lang]}/post.final.*.vec;

  echo "Re-adjusting priors based on computed posteriors for ${langs[$lang]}"
  $cmd $dir/${langs[$lang]}/log/adjust_priors.final.log \
    nnet3-am-adjust-priors $dir/${langs[$lang]}/final.mdl $dir/${langs[$lang]}/post.final.vec $dir/${langs[$lang]}/final.mdl || exit 1;

  sleep 2

done

if $cleanup; then
  echo Cleaning up data
  if $remove_egs && [[ $cur_egs_dir =~ $dir/egs* ]]; then
    steps/nnet2/remove_egs.sh $cur_egs_dir
  fi

  echo Removing most of the models
  for x in `seq 0 $num_iters`; do
    if [ $[$x%100] -ne 0 ] && [ $x -ne $num_iters ] && [ -f $dir/$x.raw ]; then
       # delete all but every 100th model; don't delete the ones which combine to form the final model.
      rm $dir/$x.raw
    fi
  done
fi
