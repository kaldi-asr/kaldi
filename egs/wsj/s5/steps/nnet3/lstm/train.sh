#!/bin/bash

# Copyright 2012-2015  Johns Hopkins University (Author: Daniel Povey).
#           2013  Xiaohui Zhang
#           2013  Guoguo Chen
#           2014  Vimal Manohar
#           2014-2015  Vijayaditya Peddinti
# Apache 2.0.

# Terminology:
# sample - one input-output tuple, which is an input sequence and output sequence for LSTM
# frame  - one output label and the input context used to compute it

# Begin configuration section.
cmd=run.pl
num_epochs=10      # Number of epochs of training;
                   # the number of iterations is worked out from this.
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=1 # Number of neural net jobs to run in parallel at the start of training
num_jobs_final=8   # Number of neural net jobs to run in parallel at the end of training
prior_subset_size=20000 # 20k samples per job, for computing priors.
num_jobs_compute_prior=10 # these are single-threaded, run on CPU.
get_egs_stage=0    # can be used for rerunning after partial
online_ivector_dir=
presoftmax_prior_scale_power=-0.25  # we haven't yet used pre-softmax prior scaling in the LSTM model
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
splice_indexes="-2,-1,0,1,2 0 0"
# Format : layer<hidden_layer>/<frame_indices>....layer<hidden_layer>/<frame_indices> "
# note: hidden layers which are composed of one or more components,
# so hidden layer indexing is different from component count

# LSTM parameters
num_lstm_layers=3
cell_dim=1024  # dimension of the LSTM cell
hidden_dim=1024  # the dimension of the fully connected hidden layer outputs
recurrent_projection_dim=256
non_recurrent_projection_dim=256
norm_based_clipping=true  # if true norm_based_clipping is used.
                          # In norm-based clipping the activation Jacobian matrix
                          # for the recurrent connections in the network is clipped
                          # to ensure that the individual row-norm (l2) does not increase
                          # beyond the clipping_threshold.
                          # If false, element-wise clipping is used.
clipping_threshold=30     # if norm_based_clipping is true this would be the maximum value of the row l2-norm,
                          # else this is the max-absolute value of each element in Jacobian.
chunk_width=20  # number of output labels in the sequence used to train an LSTM
                # Caution: if you double this you should halve --samples-per-iter.
chunk_left_context=40  # number of steps used in the estimation of LSTM state before prediction of the first label
chunk_right_context=0  # number of steps used in the estimation of LSTM state before prediction of the first label (usually used in bi-directional LSTM case)
label_delay=5  # the lstm output is used to predict the label with the specified delay
lstm_delay=" -1 -2 -3 "  # the delay to be used in the recurrence of lstms
                         # "-1 -2 -3" means the a three layer stacked LSTM would use recurrence connections with
                         # delays -1, -2 and -3 at layer1 lstm, layer2 lstm and layer3 lstm respectively
			 # "[-1,1] [-2,2] [-3,3]" means a three layer stacked bi-directional LSTM would use recurrence
			 # connections with delay -1 for the forward, 1 for the backward at layer1,
			 # -2 for the forward, 2 for the backward at layer2, and so on at layer3
num_bptt_steps=    # this variable counts the number of time steps to back-propagate from the last label in the chunk
                   # it is usually same as chunk_width


# nnet3-train options
shrink=0.99  # this parameter would be used to scale the parameter matrices
shrink_threshold=0.15  # a value less than 0.25 that we compare the mean of
                       # 'deriv-avg' for sigmoid components with, and if it's
                       # less, we shrink.
max_param_change=2.0  # max param change per minibatch
num_chunk_per_minibatch=100  # number of sequences to be processed in parallel every mini-batch

samples_per_iter=20000 # this is really the number of egs in each archive.  Each eg has
                       # 'chunk_width' frames in it-- for chunk_width=20, this value (20k)
                       # is equivalent to the 400k number that we use as a default in
                       # regular DNN training.
momentum=0.5    # e.g. 0.5.  Note: we implemented it in such a way that
                # it doesn't increase the effective learning rate.
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

rand_prune=4.0 # speeds up LDA.

# End configuration section.

trap 'for pid in $(jobs -pr); do kill -KILL $pid; done' INT QUIT TERM

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
  echo "  --num-epochs <#epochs|10>                        # Number of epochs of training"
  echo "  --initial-effective-lrate <lrate|0.0003>         # effective learning rate at start of training."
  echo "  --final-effective-lrate <lrate|0.00003>          # effective learning rate at end of training."
  echo "                                                   # data, 0.00025 for large data"
  echo "  --momentum <momentum|0.5>                        # Momentum constant: note, this is "
  echo "                                                   # implemented in such a way that it doesn't"
  echo "                                                   # increase the effective learning rate."
  echo "  --num-jobs-initial <num-jobs|1>                  # Number of parallel jobs to use for neural net training, at the start."
  echo "  --num-jobs-final <num-jobs|8>                    # Number of parallel jobs to use for neural net training, at the end"
  echo "  --num-threads <num-threads|16>                   # Number of parallel threads per job, for CPU-based training (will affect"
  echo "                                                   # results as well as speed; may interact with batch size; if you increase"
  echo "                                                   # this, you may want to decrease the batch size."
  echo "  --parallel-opts <opts|\"-pe smp 16 -l ram_free=1G,mem_free=1G\">      # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads... note, you might have to reduce mem_free,ram_free"
  echo "                                                   # versus your defaults, because it gets multiplied by the -pe smp argument."
  echo "  --splice-indexes <string|\"-2,-1,0,1,2 0 0\"> "
  echo "                                                   # Frame indices used for each splice layer."
  echo "                                                   # Format : <frame_indices> .... <frame_indices> "
  echo "                                                   # the number of fields determines the number of LSTM and non-recurrent layers"
  echo "                                                   # also see the --num-lstm-layers option"
  echo "                                                   # (note: we splice processed, typically 40-dimensional frames"
  echo "  --lda-dim <dim|''>                               # Dimension to reduce spliced features to with LDA"
  echo "  --realign-epochs <list-of-epochs|''>             # A list of space-separated epoch indices the beginning of which"
  echo "                                                   # realignment is to be done"
  echo "  --align-cmd (utils/run.pl|utils/queue.pl <queue opts>) # passed to align.sh"
  echo "  --align-use-gpu (yes/no)                         # specify is gpu is to be used for realignment"
  echo "  --num-jobs-align <#njobs|30>                     # Number of jobs to perform realignment"
  echo "  --stage <stage|-4>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."

  echo " ################### LSTM options ###################### "
  echo "  --num-lstm-layers <int|3>                        # number of LSTM layers"
  echo "  --cell-dim   <int|1024>                          # dimension of the LSTM cell"
  echo "  --hidden-dim      <int|1024>                     # the dimension of the fully connected hidden layer outputs"
  echo "  --recurrent-projection-dim  <int|256>            # the output dimension of the recurrent-projection-matrix"
  echo "  --non-recurrent-projection-dim  <int|256>        # the output dimension of the non-recurrent-projection-matrix"
  echo "  --chunk-left-context <int|40>                    # number of time-steps used in the estimation of the first LSTM state"
  echo "  --chunk-width <int|20>                           # number of output labels in the sequence used to train an LSTM"
  echo "                                                   # Caution: if you double this you should halve --samples-per-iter."
  echo "  --norm-based-clipping <bool|true>                # if true norm_based_clipping is used."
  echo "                                                   # In norm-based clipping the activation Jacobian matrix"
  echo "                                                   # for the recurrent connections in the network is clipped"
  echo "                                                   # to ensure that the individual row-norm (l2) does not increase"
  echo "                                                   # beyond the clipping_threshold."
  echo "                                                   # If false, element-wise clipping is used."
  echo "  --num-bptt-steps <int|>                          # this variable counts the number of time steps to back-propagate from the last label in the chunk"
  echo "                                                   # it defaults to chunk_width"
  echo "  --label-delay <int|5>                            # the lstm output is used to predict the label with the specified delay"

  echo "  --lstm-delay <str|\" -1 -2 -3 \">                # the delay to be used in the recurrence of lstms"
  echo "                                                   # \"-1 -2 -3\" means the a three layer stacked LSTM would use recurrence connections with "
  echo "                                                   # delays -1, -2 and -3 at layer1 lstm, layer2 lstm and layer3 lstm respectively"
  echo "  --clipping-threshold <int|30>                    # if norm_based_clipping is true this would be the maximum value of the row l2-norm,"
  echo "                                                   # else this is the max-absolute value of each element in Jacobian."

  echo " ################### LSTM specific training options ###################### "
  echo "  --num-chunks-per-minibatch <minibatch-size|100>  # Number of sequences to be processed in parallel in a minibatch"
  echo "  --samples-per-iter <#samples|20000>              # Number of egs in each archive of data.  This times --chunk-width is"
  echo "                                                   # the number of frames processed per iteration"
  echo "  --shrink <shrink|0.99>                           # if non-zero this parameter will be used to scale the parameter matrices"
  echo "  --shrink-threshold <threshold|0.15>              # a threshold (should be between 0.0 and 0.25) that controls when to"
  echo "                                                   # do parameter shrinking."
  echo " for more options see the script"
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

  # create the config files for nnet initialization
  # note an additional space is added to splice_indexes to
  # avoid issues with the python ArgParser which can have
  # issues with negative arguments (due to minus sign)
  config_extra_opts=()
  [ ! -z "$lstm_delay" ] && config_extra_opts+=(--lstm-delay "$lstm_delay")

  steps/nnet3/lstm/make_configs.py  "${config_extra_opts[@]}" \
    --splice-indexes "$splice_indexes " \
    --num-lstm-layers $num_lstm_layers \
    --feat-dim $feat_dim \
    --ivector-dim $ivector_dim \
    --cell-dim $cell_dim \
    --hidden-dim $hidden_dim \
    --recurrent-projection-dim $recurrent_projection_dim \
    --non-recurrent-projection-dim $non_recurrent_projection_dim \
    --norm-based-clipping $norm_based_clipping \
    --clipping-threshold $clipping_threshold \
    --num-targets $num_leaves \
    --label-delay $label_delay \
   $dir/configs || exit 1;
  # Initialize as "raw" nnet, prior to training the LDA-like preconditioning
  # matrix.  This first config just does any initial splicing that we do;
  # we do this as it's a convenient way to get the stats for the 'lda-like'
  # transform.
  $cmd $dir/log/nnet_init.log \
    nnet3-init --srand=-2 $dir/configs/init.config $dir/init.raw || exit 1;
fi
# sourcing the "vars" below sets
# model_left_context=(something)
# model_right_context=(something)
# num_hidden_layers=(something)
. $dir/configs/vars || exit 1;
left_context=$((chunk_left_context + model_left_context))
right_context=$((chunk_right_context + model_right_context))
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
  extra_opts+=(--valid-left-context $((chunk_width + left_context)))
  extra_opts+=(--valid-right-context $((chunk_width + right_context)))

  # Note: in RNNs we process sequences of labels rather than single label per sample
  echo "$0: calling get_egs.sh"
  steps/nnet3/get_egs.sh $egs_opts "${extra_opts[@]}" \
      --cmd "$cmd" $egs_opts \
      --stage $get_egs_stage \
      --samples-per-iter $samples_per_iter \
      --frames-per-eg $chunk_width \
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

chunk_width=$(cat $egs_dir/info/frames_per_eg) || { echo "error: no such file $egs_dir/info/frames_per_eg"; exit 1; }
num_archives=$(cat $egs_dir/info/num_archives) || { echo "error: no such file $egs_dir/info/num_archives"; exit 1; }

[ $num_jobs_initial -gt $num_jobs_final ] && \
  echo "$0: --initial-num-jobs cannot exceed --final-num-jobs" && exit 1;

[ $num_jobs_final -gt $num_archives ] && \
  echo "$0: --final-num-jobs cannot exceed #archives $num_archives." && exit 1;


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
# times, i.e. $num_iters*$avg_num_jobs) == $num_epochs*$num_archives,
# where avg_num_jobs=(num_jobs_initial+num_jobs_final)/2.

num_archives_to_process=$[$num_epochs*$num_archives]
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

approx_iters_per_epoch_final=$[$num_archives/$num_jobs_final]
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
[ -z $num_bptt_steps ] && num_bptt_steps=$chunk_width;
min_deriv_time=$((chunk_width - num_bptt_steps))
while [ $x -lt $num_iters ]; do
  [ $x -eq $exit_stage ] && echo "$0: Exiting early due to --exit-stage $exit_stage" && exit 0;

  this_num_jobs=$(perl -e "print int(0.5+$num_jobs_initial+($num_jobs_final-$num_jobs_initial)*$x/$num_iters);")

  ilr=$initial_effective_lrate; flr=$final_effective_lrate; np=$num_archives_processed; nt=$num_archives_to_process;
  this_effective_learning_rate=$(perl -e "print ($x + 1 >= $num_iters ? $flr : $ilr*exp($np*log($flr/$ilr)/$nt));");
  this_learning_rate=$(perl -e "print ($this_effective_learning_rate*$this_num_jobs);");

  if [ ! -z "${realign_this_iter[$x]}" ]; then
    prev_egs_dir=$cur_egs_dir
    cur_egs_dir=$dir/egs_${realign_this_iter[$x]}
  fi

  if [ $x -ge 0 ] && [ $stage -le $x ]; then
    # Set this_shrink value.
    if [ $x -eq 0 ] || nnet3-am-info --print-args=false $dir/$x.mdl | \
      perl -e "while(<>){ if (m/type=Sigmoid.+deriv-avg=.+mean=(\S+)/) { \$n++; \$tot+=\$1; } } exit(\$tot/\$n > $shrink_threshold);"; then
      this_shrink=$shrink; # e.g. avg-deriv of sigmoids was <= 0.125, so shrink.
    else
      this_shrink=1.0  # don't shrink: sigmoids are not over-saturated.
    fi
    echo "On iteration $x, learning rate is $this_learning_rate and shrink value is $this_shrink."

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
            "ark,bg:nnet3-merge-egs ark:$cur_egs_dir/valid_diagnostic.egs ark:- |" &
    $cmd $dir/log/compute_prob_train.$x.log \
      nnet3-compute-prob "nnet3-am-copy --raw=true $dir/$x.mdl - |" \
           "ark,bg:nnet3-merge-egs ark:$cur_egs_dir/train_diagnostic.egs ark:- |" &

    if [ $x -gt 0 ]; then
      $cmd $dir/log/progress.$x.log \
        nnet3-info "nnet3-am-copy --raw=true $dir/$x.mdl - |" '&&' \
        nnet3-show-progress --use-gpu=no "nnet3-am-copy --raw=true $dir/$[$x-1].mdl - |" "nnet3-am-copy --raw=true $dir/$x.mdl - |" \
        "ark,bg:nnet3-merge-egs --minibatch-size=256 ark:$cur_egs_dir/train_diagnostic.egs ark:-|" &
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
      cache_read_opt="" # an option for writing cache (storing pairs of nnet-computations
                        # and computation-requests) during training.
    else
      do_average=true
      if [ $x -eq 0 ]; then do_average=false; fi # on iteration 0, pick the best, don't average.
      raw="nnet3-am-copy --raw=true --learning-rate=$this_learning_rate $dir/$x.mdl -|"
      cache_read_opt="--read-cache=$dir/cache.$x"
    fi
    if $do_average; then
      this_num_chunk_per_minibatch=$num_chunk_per_minibatch
    else
      # on iteration zero or when we just added a layer, use a smaller minibatch
      # size (and we will later choose the output of just one of the jobs): the
      # model-averaging isn't always helpful when the model is changing too fast
      # (i.e. it can worsen the objective function), and the smaller minibatch
      # size will help to keep the update stable.
      this_num_chunk_per_minibatch=$[$num_chunk_per_minibatch/2];
    fi

    rm $dir/.error 2>/dev/null


    ( # this sub-shell is so that when we "wait" below,
      # we only wait for the training jobs that we just spawned,
      # not the diagnostic jobs that we spawned above.

      # We cannot easily use a single parallel SGE job to do the main training,
      # because the computation of which archive and which --frame option
      # to use for each job is a little complex, so we spawn each one separately.
      # this is no longer true for RNNs as we use do not use the --frame option
      # but we use the same script for consistency with FF-DNN code

      for n in $(seq $this_num_jobs); do
        k=$[$num_archives_processed + $n - 1]; # k is a zero-based index that we will derive
                                               # the other indexes from.
        archive=$[($k%$num_archives)+1]; # work out the 1-based archive index.
        if [ $n -eq 1 ]; then
          # an option for writing cache (storing pairs of nnet-computations and
          # computation-requests) during training.
          cache_write_opt=" --write-cache=$dir/cache.$[$x+1]"
        else
          cache_write_opt=""
        fi
        $cmd $train_queue_opt $dir/log/train.$x.$n.log \
          nnet3-train $parallel_train_opts $cache_read_opt $cache_write_opt --print-interval=10 --momentum=$momentum \
          --max-param-change=$max_param_change \
          --optimization.min-deriv-time=$min_deriv_time "$raw" \
          "ark,bg:nnet3-copy-egs $context_opts ark:$cur_egs_dir/egs.$archive.ark ark:- | nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x ark:- ark:-| nnet3-merge-egs --minibatch-size=$this_num_chunk_per_minibatch --measure-output-frames=false --discard-partial-minibatches=true ark:- ark:- |" \
          $dir/$[$x+1].$n.raw || touch $dir/.error &
      done
      wait
    )
    # the error message below is not that informative, but $cmd will
    # have printed a more specific one.
    [ -f $dir/.error ] && echo "$0: error on iteration $x of training" && exit 1;

    models_to_average=$(steps/nnet3/get_successful_models.py $this_num_jobs $dir/log/train.$x.%.log)
    nnets_list=
    for n in $models_to_average; do
      nnets_list="$nnets_list $dir/$[$x+1].$n.raw"
    done

    if $do_average; then
      # average the output of the different jobs.
      $cmd $dir/log/average.$x.log \
        nnet3-average $nnets_list - \| \
        nnet3-am-copy --scale=$this_shrink --set-raw-nnet=- $dir/$x.mdl $dir/$[$x+1].mdl || exit 1;
    else
      # choose the best from the different jobs.
      n=$(perl -e '($nj,$pat)=@ARGV; $best_n=1; $best_logprob=-1.0e+10; for ($n=1;$n<=$nj;$n++) {
          $fn = sprintf($pat,$n); open(F, "<$fn") || die "Error opening log file $fn";
          undef $logprob; while (<F>) { if (m/log-prob-per-frame=(\S+)/) { $logprob=$1; } }
          close(F); if (defined $logprob && $logprob > $best_logprob) { $best_logprob=$logprob;
          $best_n=$n; } } print "$best_n\n"; ' $this_num_jobs $dir/log/train.$x.%d.log) || exit 1;
      [ -z "$n" ] && echo "Error getting best model" && exit 1;
      $cmd $dir/log/select.$x.log \
        nnet3-am-copy --scale=$this_shrink --set-raw-nnet=$dir/$[$x+1].$n.raw  $dir/$x.mdl $dir/$[$x+1].mdl || exit 1;
    fi

    nnets_list=
    for n in `seq 1 $this_num_jobs`; do
      nnets_list="$nnets_list $dir/$[$x+1].$n.raw"
    done

    rm $nnets_list
    [ ! -f $dir/$[$x+1].mdl ] && exit 1;
    if [ -f $dir/$[$x-1].mdl ] && $cleanup && \
       [ $[($x-1)%100] -ne 0  ] && [ $[$x-1] -lt $first_model_combine ]; then
      rm $dir/$[$x-1].mdl
    fi
  fi
  rm $dir/cache.$x 2>/dev/null
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

  combine_num_chunk_per_minibatch=$(python -c "print int(1024.0/($chunk_width))")
  $cmd $combine_queue_opt $dir/log/combine.log \
    nnet3-combine --num-iters=40 \
       --enforce-sum-to-one=true --enforce-positive-weights=true \
       --verbose=3 "${nnets_list[@]}" "ark,bg:nnet3-merge-egs --measure-output-frames=false --minibatch-size=$combine_num_chunk_per_minibatch ark:$cur_egs_dir/combine.egs ark:-|" \
    "|nnet3-am-copy --set-raw-nnet=- $dir/$num_iters.mdl $dir/combined.mdl" || exit 1;

  # Compute the probability of the final, combined model with
  # the same subset we used for the previous compute_probs, as the
  # different subsets will lead to different probs.
  $cmd $dir/log/compute_prob_valid.final.log \
    nnet3-compute-prob "nnet3-am-copy --raw=true $dir/combined.mdl -|" \
    "ark,bg:nnet3-merge-egs --minibatch-size=256 ark:$cur_egs_dir/valid_diagnostic.egs ark:- |" &
  $cmd $dir/log/compute_prob_train.final.log \
    nnet3-compute-prob  "nnet3-am-copy --raw=true $dir/combined.mdl -|" \
    "ark,bg:nnet3-merge-egs --minibatch-size=256 ark:$cur_egs_dir/train_diagnostic.egs ark:- |" &
fi

if [ $stage -le $[$num_iters+1] ]; then
  echo "Getting average posterior for purposes of adjusting the priors."
  # Note: this just uses CPUs, using a smallish subset of data.
  rm $dir/post.$x.*.vec 2>/dev/null
  if [ $num_jobs_compute_prior -gt $num_archives ]; then egs_part=1;
  else egs_part=JOB; fi
  $cmd JOB=1:$num_jobs_compute_prior $prior_queue_opt $dir/log/get_post.$x.JOB.log \
    nnet3-subset-egs --srand=JOB --n=$prior_subset_size ark:$cur_egs_dir/egs.$egs_part.ark ark:- \| \
    nnet3-merge-egs --measure-output-frames=true --minibatch-size=128 ark:- ark:- \| \
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
