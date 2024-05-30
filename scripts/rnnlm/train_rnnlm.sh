#!/usr/bin/env bash

# This script does the RNNLM training.  It assumes you have already run
# 'prepare_rnnlm_dir.sh' to prepare the directory.


#num-jobs-initial, num-jobs-final, max-change, embedding-max-change [initial,final?],
#num-samples, minibatch-size, chunk-length, [and the same for dev data]...
#initial-effective-learning-rate, final-effective-learning-rate, ...
#embedding-learning-rate-factor, num-epochs


stage=0
num_jobs_initial=1
num_jobs_final=1
rnnlm_max_change=0.5
embedding_max_change=0.5
chunk_length=32
num_epochs=100  # maximum number of epochs to train.  later we
                # may find a stopping criterion.
initial_effective_lrate=0.001
final_effective_lrate=0.0001
embedding_l2=0.005
embedding_lrate_factor=0.1  # the embedding learning rate is the
                            # nnet learning rate times this factor.
backstitch_training_scale=0.0    # backstitch training scale
backstitch_training_interval=1   # backstitch training interval
cmd=run.pl  # you might want to set this to queue.pl

# some options passed into rnnlm-get-egs, relating to sampling.
num_samples=512
sample_group_size=2  # see rnnlm-get-egs
num_egs_threads=10  # number of threads used for sampling, if we're using
                    # sampling.  the actual number of threads that runs at one
                    # time, will be however many is needed to balance the
                    # sampling and the actual training, this is just the maximum
                    # possible number that are allowed to run
use_gpu=yes  # use GPU for training
use_gpu_for_diagnostics=false  # set true to use GPU for compute_prob_*.log

# optional cleanup options
cleanup=false  # add option --cleanup true to enable automatic cleanup of old models
cleanup_strategy="keep_latest"  # determines cleanup strategy, use either "keep_latest" or "keep_best"
cleanup_keep_iters=3  # number of iterations that will have their models retained

trap 'for pid in $(jobs -pr); do kill -KILL $pid; done' INT QUIT TERM
. utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: $0 [options] <rnnlm-dir>"
  echo "Trains an RNNLM, assuming the things needed for training have already been"
  echo "set up by prepare_rnnlm_dir.sh."
  exit 1
fi


dir=$1


set -e
. ./path.sh


for f in $dir/config/{words,data_weights,oov}.txt \
              $dir/text/1.txt $dir/text/dev.txt $dir/0.raw \
              $dir/text/info/num_splits $dir/text/info/num_repeats \
              $dir/special_symbol_opts.txt; do
  [ ! -f $f ] && echo "$0: expected $f to exist" && exit 1
done

# set some variables and check more files.
num_splits=$(cat $dir/text/info/num_splits)
num_repeats=$(cat $dir/text/info/num_repeats)
text_files=$(for n in $(seq $num_splits); do echo $dir/text/$n.txt; done)
vocab_size=$(tail -n 1 $dir/config/words.txt | awk '{print $NF + 1}')
embedding_type=

if [ -f $dir/feat_embedding.0.mat ]; then
  sparse_features=true
  embedding_type=feat
  if [ -f $dir/word_embedding.0.mat ]; then
    echo "$0: error: $dir/feat_embedding.0.mat and $dir/word_embedding.0.mat both exist."
    exit 1;
  fi
  ! [ -f $dir/word_feats.txt ] && echo "$0: expected $0/word_feats.txt to exist" && exit 1;
else
  sparse_features=false
  embedding_type=word
  ! [ -f $dir/word_embedding.0.mat ] && \
    echo "$0: expected $dir/word_embedding.0.mat to exist" && exit 1
fi

if [ $num_jobs_initial -gt $num_splits ] || [ $num_jobs_final -gt $num_splits ]; then
  echo -n "$0: number of initial or final jobs $num_jobs_initial/$num_jobs_final"
  echo "exceeds num-splits=$num_splits; reduce number of jobs"
  exit 1
fi

num_splits_to_process=$[($num_epochs*$num_splits)/$num_repeats]
num_split_processed=0
num_iters=$[($num_splits_to_process*2)/($num_jobs_initial+$num_jobs_final)]


# this string will combine options and arguments.
train_egs_args="--vocab-size=$vocab_size $(cat $dir/special_symbol_opts.txt)"
if [ -f $dir/sampling.lm ]; then
  # we are doing sampling.
  train_egs_args="$train_egs_args --num-samples=$num_samples --sample-group-size=$sample_group_size --num-threads=$num_egs_threads $dir/sampling.lm"
fi

echo "$0: will train for $num_iters iterations"

# recording some configuration information
cat >$dir/info.txt <<EOF
num_iters=$num_iters
num_epochs=$num_epochs
num_jobs_initial=$num_jobs_initial
num_jobs_final=$num_jobs_final
rnnlm_max_change=$rnnlm_max_change
embedding_max_change=$embedding_max_change
chunk_length=$chunk_length
initial_effective_lrate=$initial_effective_lrate
final_effective_lrate=$final_effective_lrate
embedding_lrate_factor=$embedding_lrate_factor
sample_group_size=$sample_group_size
num_samples=$num_samples
backstitch_training_scale=$backstitch_training_scale
backstitch_training_interval=$backstitch_training_interval
EOF


x=0
num_splits_processed=0
while [ $x -lt $num_iters ]; do

  this_num_jobs=$(perl -e "print int(0.5+$num_jobs_initial+($num_jobs_final-$num_jobs_initial)*$x/$num_iters);")
  ilr=$initial_effective_lrate; flr=$final_effective_lrate; np=$num_splits_processed; nt=$num_splits_to_process;
  this_learning_rate=$(perl -e "print (($x + 1 >= $num_iters ? $flr : $ilr*exp($np*log($flr/$ilr)/$nt))*$this_num_jobs);");
  embedding_lrate=$(perl -e "print ($this_learning_rate*$embedding_lrate_factor);")

  if [ $stage -le $x ]; then

    # Set off the diagnostic job in the background.
    if $sparse_features; then
      word_embedding="rnnlm-get-word-embedding $dir/word_feats.txt $dir/feat_embedding.$x.mat -|"
    else
      word_embedding="$dir/word_embedding.$x.mat"
    fi
    if $use_gpu_for_diagnostics; then queue_gpu_opt="--gpu 1"; gpu_opt="--use-gpu=yes";
    else gpu_opt=''; queue_gpu_opt=''; fi
    backstitch_opt="--rnnlm.backstitch-training-scale=$backstitch_training_scale \
      --rnnlm.backstitch-training-interval=$backstitch_training_interval \
      --embedding.backstitch-training-scale=$backstitch_training_scale \
      --embedding.backstitch-training-interval=$backstitch_training_interval"
    [ -f $dir/.error ] && rm $dir/.error
    $cmd $queue_gpu_opt $dir/log/compute_prob.$x.log \
       rnnlm-get-egs $(cat $dir/special_symbol_opts.txt) \
                     --vocab-size=$vocab_size $dir/text/dev.txt ark:- \| \
       rnnlm-compute-prob $gpu_opt $dir/$x.raw "$word_embedding" ark:- || touch $dir/.error &

    if [ $x -gt 0 ]; then
      $cmd $dir/log/progress.$x.log \
        nnet3-show-progress --use-gpu=no $dir/$[$x-1].raw $dir/$x.raw '&&' \
          nnet3-info $dir/$x.raw &
    fi

    echo "Training neural net (pass $x)"


    ( # this sub-shell is so that when we "wait" below,
      # we only wait for the training jobs that we just spawned,
      # not the diagnostic jobs that we spawned above.

      # We can't easily use a single parallel SGE job to do the main training,
      # because the computation of which archive and which --frame option
      # to use for each job is a little complex, so we spawn each one separately.
      [ -f $dir/.train_error ] && rm $dir/.train_error
      for n in $(seq $this_num_jobs); do
        k=$[$num_splits_processed + $n - 1]; # k is a zero-based index that we'll derive
                                               # the other indexes from.
        split=$[($k%$num_splits)+1]; # work out the 1-based split index.

        src_rnnlm="nnet3-copy --learning-rate=$this_learning_rate $dir/$x.raw -|"
        if $sparse_features; then
          sparse_opt="--read-sparse-word-features=$dir/word_feats.txt";
          embedding_type=feat
        else
          sparse_opt=''; embedding_type=word
        fi
        gpu_opt="--use-gpu=$use_gpu"
        if [ $use_gpu == "yes" ] || [ $use_gpu == "true" ] || [ $use_gpu == "wait" ]; then
          queue_gpu_opt="--gpu 1";
        else
          queue_gpu_opt="";
        fi
        if [ $this_num_jobs -gt 1 ]; then dest_number=$[x+1].$n
        else dest_number=$[x+1]; fi
        # in the normal case $repeated data will be just one copy.
        repeated_data=$(for n in $(seq $num_repeats); do echo -n $dir/text/$split.txt ''; done)

        rnnlm_l2_factor=$(perl -e "print (1.0/$this_num_jobs);")
        embedding_l2_regularize=$(perl -e "print ($embedding_l2/$this_num_jobs);")

        # allocate queue-slots for threads doing sampling,
        num_threads_=$[$num_egs_threads*2/3]
        [ -f $dir/sampling.lm ] && queue_thread_opt="--num-threads $num_threads_" || queue_thread_opt=

        # Run the training job or jobs.
        $cmd $queue_gpu_opt $queue_thread_opt $dir/log/train.$x.$n.log \
           rnnlm-train \
             --rnnlm.max-param-change=$rnnlm_max_change \
             --rnnlm.l2_regularize_factor=$rnnlm_l2_factor \
             --embedding.max-param-change=$embedding_max_change \
             --embedding.learning-rate=$embedding_lrate \
             --embedding.l2_regularize=$embedding_l2_regularize \
             $sparse_opt $gpu_opt $backstitch_opt \
             --read-rnnlm="$src_rnnlm" --write-rnnlm=$dir/$dest_number.raw \
             --read-embedding=$dir/${embedding_type}_embedding.$x.mat \
             --write-embedding=$dir/${embedding_type}_embedding.$dest_number.mat \
             "ark,bg:cat $repeated_data | rnnlm-get-egs --chunk-length=$chunk_length --srand=$num_splits_processed $train_egs_args - ark:- |" || touch $dir/.train_error &
      done
      wait # wait for just the training jobs.
      [ -f $dir/.train_error ] && \
        echo "$0: failure on iteration $x of training, see $dir/log/train.$x.*.log for details." && exit 1
      if [ $this_num_jobs -gt 1 ]; then
        # average the models and the embedding matrces.  Use run.pl as we don\'t
        # want this to wait on the queue (if there is a queue).
        src_models=$(for n in $(seq $this_num_jobs); do echo $dir/$[x+1].$n.raw; done)
        src_matrices=$(for n in $(seq $this_num_jobs); do echo $dir/${embedding_type}_embedding.$[x+1].$n.mat; done)
        run.pl $dir/log/average.$[x+1].log \
          nnet3-average $src_models $dir/$[x+1].raw '&&' \
          matrix-sum --average=true $src_matrices $dir/${embedding_type}_embedding.$[x+1].mat
        rm $src_models
        rm $src_matrices
      fi
      # optionally, perform cleanup after training
      if [ "$cleanup" = true ] ; then
        python3 rnnlm/rnnlm_cleanup.py $dir --$cleanup_strategy --iters_to_keep $cleanup_keep_iters
      fi
    )
    # the error message below is not that informative, but $cmd will
    # have printed a more specific one.
    [ -f $dir/.error ] && echo "$0: error with diagnostics on iteration $x of training" && exit 1;
  fi

  x=$[x+1]
  num_splits_processed=$[num_splits_processed+this_num_jobs]
done

wait # wait for diagnostic jobs in the background.

if [ $stage -le $num_iters ]; then
  # link the best model we encountered during training (based on
  # dev-set probability) as the final model.
  best_iter=$(rnnlm/get_best_model.py $dir)
  echo "$0: best iteration (out of $num_iters) was $best_iter, linking it to final iteration."
  train_best_log=$dir/log/train.$best_iter.1.log
  ppl_train=`grep 'Overall objf' $train_best_log | awk '{printf("%.1f",exp(-$10))}'`
  dev_best_log=$dir/log/compute_prob.$best_iter.log
  ppl_dev=`grep 'Overall objf' $dev_best_log | awk '{printf("%.1f",exp(-$NF))}'`
  echo "$0: train/dev perplexity was $ppl_train / $ppl_dev."
  ln -sf ${embedding_type}_embedding.$best_iter.mat $dir/${embedding_type}_embedding.final.mat
  ln -sf $best_iter.raw $dir/final.raw
fi

# Now get some diagnostics about the evolution of the objective function.
if [ $stage -le $[num_iters+1] ]; then
  (
    logs=$(for iter in $(seq 1 $[$num_iters-1]); do echo -n $dir/log/train.$iter.1.log ''; done)
    # in the non-sampling case the exact objf is printed and we plot that
    # in the sampling case we print the approximated objf for training.
    grep 'Overall objf' $logs | awk 'BEGIN{printf("Train objf: ")} /exact/{printf("%.2f ", $NF);next} {printf("%.2f ", $10)} END{print "";}'
    logs=$(for iter in $(seq 1 $[$num_iters-1]); do echo -n $dir/log/compute_prob.$iter.log ''; done)
    grep 'Overall objf' $logs | awk 'BEGIN{printf("Dev objf:   ")} {printf("%.2f ", $NF)} END{print "";}'
  ) > $dir/report.txt
  cat $dir/report.txt
fi
