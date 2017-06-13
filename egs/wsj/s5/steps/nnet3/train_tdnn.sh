#!/bin/bash

# THIS SCRIPT IS DEPRECATED, see ./train_dnn.py

# note, TDNN is the same as what we used to call multisplice.

# Copyright 2012-2015  Johns Hopkins University (Author: Daniel Povey).
#           2013  Xiaohui Zhang
#           2013  Guoguo Chen
#           2014  Vimal Manohar
#           2014  Vijayaditya Peddinti
# Apache 2.0.


# Begin configuration section.
cmd=run.pl
num_epochs=15      # Number of epochs of training;
                   # the number of iterations is worked out from this.
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
cleanup=true
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

echo "$0: THIS SCRIPT IS DEPRECATED"
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
  echo "  --parallel-opts <opts|\"--num-threads 16 --mem 1G\">      # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads... note, you might have to reduce --mem"
  echo "                                                   # versus your defaults, because it gets multiplied by the --num-threads argument."
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

data=$1
lang=$2
alidir=$3
dir=$4

if [ ! -z "$realign_times" ]; then
  [ -z "$align_cmd" ] && echo "$0: realign_times specified but align_cmd not specified" && exit 1
  [ -z "$align_use_gpu" ] && echo "$0: realign_times specified but align_use_gpu not specified" && exit 1
fi

# Check some files.
for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl $alidir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


# Set some variables.
num_leaves=`tree-info $alidir/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1
[ -z $num_leaves ] && echo "\$num_leaves is unset" && exit 1
[ "$num_leaves" -eq "0" ] && echo "\$num_leaves is 0" && exit 1

nj=`cat $alidir/num_jobs` || exit 1;  # number of jobs in alignment dir...
# in this dir we'll have just one job.
sdata=$data/split$nj
utils/split_data.sh $data $nj

mkdir -p $dir/log
echo $nj > $dir/num_jobs
cp $alidir/tree $dir

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

# First work out the feature and iVector dimension, needed for tdnn config creation.
case $feat_type in
  raw) feat_dim=$(feat-to-dim --print-args=false scp:$data/feats.scp -) || \
      { echo "$0: Error getting feature dim"; exit 1; }
    ;;
  lda)  [ ! -f $alidir/final.mat ] && echo "$0: With --feat-type lda option, expect $alidir/final.mat to exist."
   # get num-rows in lda matrix, which is the lda feature dim.
   feat_dim=$(matrix-dim --print-args=false $alidir/final.mat | cut -f 1)
    ;;
  *)
   echo "$0: Bad --feat-type '$feat_type';"; exit 1;
esac
if [ -z "$online_ivector_dir" ]; then
  ivector_dim=0
else
  ivector_dim=$(feat-to-dim scp:$online_ivector_dir/ivector_online.scp -) || exit 1;
fi


if [ $stage -le -5 ]; then
  echo "$0: creating neural net configs";

  if [ ! -z "$relu_dim" ]; then
    dim_opts="--relu-dim $relu_dim"
  else
    dim_opts="--pnorm-input-dim $pnorm_input_dim --pnorm-output-dim  $pnorm_output_dim"
  fi

  # create the config files for nnet initialization
  python steps/nnet3/make_tdnn_configs.py  \
    --splice-indexes "$splice_indexes"  \
    --feat-dim $feat_dim \
    --ivector-dim $ivector_dim  \
     $dim_opts \
    --use-presoftmax-prior-scale $use_presoftmax_prior_scale \
    --num-targets  $num_leaves  \
   $dir/configs || exit 1;

  # Initialize as "raw" nnet, prior to training the LDA-like preconditioning
  # matrix.  This first config just does any initial splicing that we do;
  # we do this as it's a convenient way to get the stats for the 'lda-like'
  # transform.
  $cmd $dir/log/nnet_init.log \
    nnet3-init --srand=-2 $dir/configs/init.config $dir/init.raw || exit 1;
fi

# sourcing the "vars" below sets
# left_context=(something)
# right_context=(something)
# num_hidden_layers=(something)
. $dir/configs/vars || exit 1;

context_opts="--left-context=$left_context --right-context=$right_context"

! [ "$num_hidden_layers" -gt 0 ] && echo \
 "$0: Expected num_hidden_layers to be defined" && exit 1;

[ -z "$transform_dir" ] && transform_dir=$alidir


if [ $stage -le -4 ] && [ -z "$egs_dir" ]; then
  extra_opts=()
  [ ! -z "$cmvn_opts" ] && extra_opts+=(--cmvn-opts "$cmvn_opts")
  [ ! -z "$feat_type" ] && extra_opts+=(--feat-type $feat_type)
  [ ! -z "$online_ivector_dir" ] && extra_opts+=(--online-ivector-dir $online_ivector_dir)
  extra_opts+=(--transform-dir $transform_dir)
  extra_opts+=(--left-context $left_context)
  extra_opts+=(--right-context $right_context)
  echo "$0: calling get_egs.sh"
  steps/nnet3/get_egs.sh $egs_opts "${extra_opts[@]}" \
      --samples-per-iter $samples_per_iter --stage $get_egs_stage \
      --cmd "$cmd" $egs_opts \
      --frames-per-eg $frames_per_eg \
      $data $alidir $dir/egs || exit 1;
fi

[ -z $egs_dir ] && egs_dir=$dir/egs

if [ "$feat_dim" != "$(cat $egs_dir/info/feat_dim)" ]; then
  echo "$0: feature dimension mismatch with egs, $feat_dim vs $(cat $egs_dir/info/feat_dim)";
  exit 1;
fi
if [ "$ivector_dim" != "$(cat $egs_dir/info/ivector_dim)" ]; then
  echo "$0: ivector dimension mismatch with egs, $ivector_dim vs $(cat $egs_dir/info/ivector_dim)";
  exit 1;
fi

# copy any of the following that exist, to $dir.
cp $egs_dir/{cmvn_opts,splice_opts,final.mat} $dir 2>/dev/null

# confirm that the egs_dir has the necessary context (especially important if
# the --egs-dir option was used on the command line).
egs_left_context=$(cat $egs_dir/info/left_context) || exit -1
egs_right_context=$(cat $egs_dir/info/right_context) || exit -1
 ( [ $egs_left_context -lt $left_context ] || \
   [ $egs_right_context -lt $right_context ] ) && \
   echo "$0: egs in $egs_dir have too little context" && exit -1;

frames_per_eg=$(cat $egs_dir/info/frames_per_eg) || { echo "error: no such file $egs_dir/info/frames_per_eg"; exit 1; }
num_archives=$(cat $egs_dir/info/num_archives) || { echo "error: no such file $egs_dir/info/frames_per_eg"; exit 1; }

# num_archives_expanded considers each separate label-position from
# 0..frames_per_eg-1 to be a separate archive.
num_archives_expanded=$[$num_archives*$frames_per_eg]

[ $num_jobs_initial -gt $num_jobs_final ] && \
  echo "$0: --initial-num-jobs cannot exceed --final-num-jobs" && exit 1;

[ $num_jobs_final -gt $num_archives_expanded ] && \
  echo "$0: --final-num-jobs cannot exceed #archives $num_archives_expanded." && exit 1;


if [ $stage -le -3 ]; then
  echo "$0: getting preconditioning matrix for input features."
  num_lda_jobs=$num_archives
  [ $num_lda_jobs -gt $max_lda_jobs ] && num_lda_jobs=$max_lda_jobs

  # Write stats with the same format as stats for LDA.
  $cmd JOB=1:$num_lda_jobs $dir/log/get_lda_stats.JOB.log \
      nnet3-acc-lda-stats --rand-prune=$rand_prune \
        $dir/init.raw "ark:$egs_dir/egs.JOB.ark" $dir/JOB.lda_stats || exit 1;

  all_lda_accs=$(for n in $(seq $num_lda_jobs); do echo $dir/$n.lda_stats; done)
  $cmd $dir/log/sum_transform_stats.log \
    sum-lda-accs $dir/lda_stats $all_lda_accs || exit 1;

  rm $all_lda_accs || exit 1;

  # this computes a fixed affine transform computed in the way we described in
  # Appendix C.6 of http://arxiv.org/pdf/1410.7455v6.pdf; it's a scaled variant
  # of an LDA transform but without dimensionality reduction.
  $cmd $dir/log/get_transform.log \
     nnet-get-feature-transform $lda_opts $dir/lda.mat $dir/lda_stats || exit 1;

  ln -sf ../lda.mat $dir/configs/lda.mat
fi


if [ $stage -le -2 ]; then
  echo "$0: preparing initial vector for FixedScaleComponent before softmax"
  echo "  ... using priors^$presoftmax_prior_scale_power and rescaling to average 1"

  # obtains raw pdf count
  $cmd JOB=1:$nj $dir/log/acc_pdf.JOB.log \
     ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
     post-to-tacc --per-pdf=true  $alidir/final.mdl ark:- $dir/pdf_counts.JOB || exit 1;
  $cmd $dir/log/sum_pdf_counts.log \
       vector-sum --binary=false $dir/pdf_counts.* $dir/pdf_counts || exit 1;
  rm $dir/pdf_counts.*

  awk -v power=$presoftmax_prior_scale_power -v smooth=0.01 \
     '{ for(i=2; i<=NF-1; i++) { count[i-2] = $i;  total += $i; }
        num_pdfs=NF-2;  average_count = total/num_pdfs;
        for (i=0; i<num_pdfs; i++) stot += (scale[i] = (count[i] + smooth * average_count)^power)
        printf " [ "; for (i=0; i<num_pdfs; i++) printf("%f ", scale[i]*num_pdfs/stot); print "]" }' \
     $dir/pdf_counts > $dir/presoftmax_prior_scale.vec
  ln -sf ../presoftmax_prior_scale.vec $dir/configs/presoftmax_prior_scale.vec
fi

if [ $stage -le -1 ]; then
  # Add the first layer; this will add in the lda.mat and
  # presoftmax_prior_scale.vec.
  $cmd $dir/log/add_first_layer.log \
       nnet3-init --srand=-3 $dir/init.raw $dir/configs/layer1.config $dir/0.raw || exit 1;

  # Convert to .mdl, train the transitions, set the priors.
  $cmd $dir/log/init_mdl.log \
    nnet3-am-init $alidir/final.mdl $dir/0.raw - \| \
    nnet3-am-train-transitions - "ark:gunzip -c $alidir/ali.*.gz|" $dir/0.mdl || exit 1;
fi


# set num_iters so that as close as possible, we process the data $num_epochs
# times, i.e. $num_iters*$avg_num_jobs) == $num_epochs*$num_archives_expanded,
# where avg_num_jobs=(num_jobs_initial+num_jobs_final)/2.

num_archives_to_process=$[$num_epochs*$num_archives_expanded]
num_archives_processed=0
num_iters=$[($num_archives_to_process*2)/($num_jobs_initial+$num_jobs_final)]

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


approx_iters_per_epoch_final=$[$num_archives_expanded/$num_jobs_final]
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
  this_learning_rate=$(perl -e "print (($x + 1 >= $num_iters ? $flr : $ilr*exp($np*log($flr/$ilr)/$nt))*$this_num_jobs);");

  echo "On iteration $x, learning rate is $this_learning_rate."

  if [ ! -z "${realign_this_iter[$x]}" ]; then
    prev_egs_dir=$cur_egs_dir
    cur_egs_dir=$dir/egs_${realign_this_iter[$x]}
  fi

  if [ $x -ge 0 ] && [ $stage -le $x ]; then
    if [ ! -z "${realign_this_iter[$x]}" ]; then
      time=${realign_this_iter[$x]}

      echo "Getting average posterior for purposes of adjusting the priors."
      # Note: this just uses CPUs, using a smallish subset of data.
      # always use the first egs archive, which makes the script simpler;
      # we're using different random subsets of it.
      rm $dir/post.$x.*.vec 2>/dev/null
      $cmd JOB=1:$num_jobs_compute_prior $dir/log/get_post.$x.JOB.log \
        nnet3-copy-egs --srand=JOB --frame=random $context_opts ark:$prev_egs_dir/egs.1.ark ark:- \| \
        nnet3-subset-egs --srand=JOB --n=$prior_subset_size ark:- ark:- \| \
        nnet3-merge-egs ark:- ark:- \| \
        nnet3-compute-from-egs --apply-exp=true "nnet3-am-copy --raw=true $dir/$x.mdl -|" ark:- ark:- \| \
        matrix-sum-rows ark:- ark:- \| vector-sum ark:- $dir/post.$x.JOB.vec || exit 1;

      sleep 3;  # make sure there is time for $dir/post.$x.*.vec to appear.

      $cmd $dir/log/vector_sum.$x.log \
        vector-sum $dir/post.$x.*.vec $dir/post.$x.vec || exit 1;
      rm $dir/post.$x.*.vec;

      echo "Re-adjusting priors based on computed posteriors"
      $cmd $dir/log/adjust_priors.$x.log \
        nnet3-am-adjust-priors $dir/$x.mdl $dir/post.$x.vec $dir/$x.mdl || exit 1;

      sleep 2

      steps/nnet3/align.sh --nj $num_jobs_align --cmd "$align_cmd" --use-gpu $align_use_gpu \
        --transform-dir "$transform_dir" --online-ivector-dir "$online_ivector_dir" \
        --iter $x $data $lang $dir $dir/ali_$time || exit 1

      steps/nnet3/relabel_egs.sh --cmd "$cmd" --iter $x $dir/ali_$time \
        $prev_egs_dir $cur_egs_dir || exit 1

      if $cleanup && [[ $prev_egs_dir =~ $dir/egs* ]]; then
        steps/nnet3/remove_egs.sh $prev_egs_dir
      fi
    fi

    # Set off jobs doing some diagnostics, in the background.
    # Use the egs dir from the previous iteration for the diagnostics
    $cmd $dir/log/compute_prob_valid.$x.log \
      nnet3-compute-prob "nnet3-am-copy --raw=true $dir/$x.mdl - |" \
            "ark:nnet3-merge-egs ark:$cur_egs_dir/valid_diagnostic.egs ark:- |" &
    $cmd $dir/log/compute_prob_train.$x.log \
      nnet3-compute-prob "nnet3-am-copy --raw=true $dir/$x.mdl - |" \
           "ark:nnet3-merge-egs ark:$cur_egs_dir/train_diagnostic.egs ark:- |" &

    if [ $x -gt 0 ]; then
      $cmd $dir/log/progress.$x.log \
        nnet3-show-progress --use-gpu=no "nnet3-am-copy --raw=true $dir/$[$x-1].mdl - |" "nnet3-am-copy --raw=true $dir/$x.mdl - |" \
        "ark:nnet3-merge-egs ark:$cur_egs_dir/train_diagnostic.egs ark:-|" '&&' \
        nnet3-info "nnet3-am-copy --raw=true $dir/$x.mdl - |" &
    fi

    echo "Training neural net (pass $x)"

    if [ $x -gt 0 ] && \
      [ $x -le $[($num_hidden_layers-1)*$add_layers_period] ] && \
      [ $[$x%$add_layers_period] -eq 0 ]; then
      do_average=false # if we've just mixed up, don't do averaging but take the
                       # best.
      cur_num_hidden_layers=$[1+$x/$add_layers_period]
      config=$dir/configs/layer$cur_num_hidden_layers.config
      raw="nnet3-am-copy --raw=true --learning-rate=$this_learning_rate $dir/$x.mdl - | nnet3-init --srand=$x - $config - |"
    else
      do_average=true
      if [ $x -eq 0 ]; then do_average=false; fi # on iteration 0, pick the best, don't average.
      raw="nnet3-am-copy --raw=true --learning-rate=$this_learning_rate $dir/$x.mdl -|"
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
        k=$[$num_archives_processed + $n - 1]; # k is a zero-based index that we'll derive
                                               # the other indexes from.
        archive=$[($k%$num_archives)+1]; # work out the 1-based archive index.
        frame=$[(($k/$num_archives)%$frames_per_eg)]; # work out the 0-based frame
        # index; this increases more slowly than the archive index because the
        # same archive with different frame indexes will give similar gradients,
        # so we want to separate them in time.

        $cmd $train_queue_opt $dir/log/train.$x.$n.log \
          nnet3-train $parallel_train_opts \
          --max-param-change=$max_param_change "$raw" \
          "ark:nnet3-copy-egs --frame=$frame $context_opts ark:$cur_egs_dir/egs.$archive.ark ark:- | nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x ark:- ark:-| nnet3-merge-egs --minibatch-size=$this_minibatch_size --discard-partial-minibatches=true ark:- ark:- |" \
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
        nnet3-am-copy --set-raw-nnet=- $dir/$x.mdl $dir/$[$x+1].mdl || exit 1;
    else
      # choose the best from the different jobs.
      n=$(perl -e '($nj,$pat)=@ARGV; $best_n=1; $best_logprob=-1.0e+10; for ($n=1;$n<=$nj;$n++) {
          $fn = sprintf($pat,$n); open(F, "<$fn") || die "Error opening log file $fn";
          undef $logprob; while (<F>) { if (m/log-prob-per-frame=(\S+)/) { $logprob=$1; } }
          close(F); if (defined $logprob && $logprob > $best_logprob) { $best_logprob=$logprob;
          $best_n=$n; } } print "$best_n\n"; ' $num_jobs_nnet $dir/log/train.$x.%d.log) || exit 1;
      [ -z "$n" ] && echo "Error getting best model" && exit 1;
      $cmd $dir/log/select.$x.log \
        nnet3-am-copy --set-raw-nnet=$dir/$[$x+1].$n.raw  $dir/$x.mdl $dir/$[$x+1].mdl || exit 1;
    fi

    rm $nnets_list
    [ ! -f $dir/$[$x+1].mdl ] && exit 1;
    if [ -f $dir/$[$x-1].mdl ] && $cleanup && \
       [ $[($x-1)%100] -ne 0  ] && [ $[$x-1] -lt $first_model_combine ]; then
      rm $dir/$[$x-1].mdl
    fi
  fi
  x=$[$x+1]
  num_archives_processed=$[$num_archives_processed+$this_num_jobs]
done


if [ $stage -le $num_iters ]; then
  echo "Doing final combination to produce final.mdl"

  # Now do combination.  In the nnet3 setup, the logic
  # for doing averaging of subsets of the models in the case where
  # there are too many models to reliably esetimate interpolation
  # factors (max_models_combine) is moved into the nnet3-combine
  nnets_list=()
  for n in $(seq 0 $[num_iters_combine-1]); do
    iter=$[$first_model_combine+$n]
    mdl=$dir/$iter.mdl
    [ ! -f $mdl ] && echo "Expected $mdl to exist" && exit 1;
    nnets_list[$n]="nnet3-am-copy --raw=true $mdl -|";
  done

  # Below, we use --use-gpu=no to disable nnet3-combine-fast from using a GPU,
  # as if there are many models it can give out-of-memory error; and we set
  # num-threads to 8 to speed it up (this isn't ideal...)

  $cmd $combine_queue_opt $dir/log/combine.log \
    nnet3-combine --num-iters=40 \
       --enforce-sum-to-one=true --enforce-positive-weights=true \
       --verbose=3 "${nnets_list[@]}" "ark:nnet3-merge-egs --minibatch-size=1024 ark:$cur_egs_dir/combine.egs ark:-|" \
    "|nnet3-am-copy --set-raw-nnet=- $dir/$num_iters.mdl $dir/combined.mdl" || exit 1;

  # Compute the probability of the final, combined model with
  # the same subset we used for the previous compute_probs, as the
  # different subsets will lead to different probs.
  $cmd $dir/log/compute_prob_valid.final.log \
    nnet3-compute-prob "nnet3-am-copy --raw=true $dir/combined.mdl -|" \
    "ark:nnet3-merge-egs ark:$cur_egs_dir/valid_diagnostic.egs ark:- |" &
  $cmd $dir/log/compute_prob_train.final.log \
    nnet3-compute-prob  "nnet3-am-copy --raw=true $dir/combined.mdl -|" \
    "ark:nnet3-merge-egs ark:$cur_egs_dir/train_diagnostic.egs ark:- |" &
fi

if [ $stage -le $[$num_iters+1] ]; then
  echo "Getting average posterior for purposes of adjusting the priors."
  # Note: this just uses CPUs, using a smallish subset of data.
  if [ $num_jobs_compute_prior -gt $num_archives ]; then egs_part=1;
  else egs_part=JOB; fi
  rm $dir/post.$x.*.vec 2>/dev/null
  $cmd JOB=1:$num_jobs_compute_prior $prior_queue_opt $dir/log/get_post.$x.JOB.log \
    nnet3-copy-egs --frame=random $context_opts --srand=JOB ark:$cur_egs_dir/egs.$egs_part.ark ark:- \| \
    nnet3-subset-egs --srand=JOB --n=$prior_subset_size ark:- ark:- \| \
    nnet3-merge-egs ark:- ark:- \| \
    nnet3-compute-from-egs $prior_gpu_opt --apply-exp=true \
      "nnet3-am-copy --raw=true $dir/combined.mdl -|" ark:- ark:- \| \
    matrix-sum-rows ark:- ark:- \| vector-sum ark:- $dir/post.$x.JOB.vec || exit 1;

  sleep 3;  # make sure there is time for $dir/post.$x.*.vec to appear.

  $cmd $dir/log/vector_sum.$x.log \
   vector-sum $dir/post.$x.*.vec $dir/post.$x.vec || exit 1;

  rm $dir/post.$x.*.vec;

  echo "Re-adjusting priors based on computed posteriors"
  $cmd $dir/log/adjust_priors.final.log \
    nnet3-am-adjust-priors $dir/combined.mdl $dir/post.$x.vec $dir/final.mdl || exit 1;
fi


if [ ! -f $dir/final.mdl ]; then
  echo "$0: $dir/final.mdl does not exist."
  # we don't want to clean up if the training didn't succeed.
  exit 1;
fi

sleep 2

echo Done

if $cleanup; then
  echo Cleaning up data
  if $remove_egs && [[ $cur_egs_dir =~ $dir/egs* ]]; then
    steps/nnet2/remove_egs.sh $cur_egs_dir
  fi

  echo Removing most of the models
  for x in `seq 0 $num_iters`; do
    if [ $[$x%100] -ne 0 ] && [ $x -ne $num_iters ] && [ -f $dir/$x.mdl ]; then
       # delete all but every 100th model; don't delete the ones which combine to form the final model.
      rm $dir/$x.mdl
    fi
  done
fi

steps/info/nnet3_dir_info.pl $dir

exit 0
