#!/usr/bin/env bash

# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey). 
#           2013  Xiaohui Zhang
#           2013  Guoguo Chen
#           2014  Vimal Manohar
#           2014  Vijayaditya Peddinti
# Apache 2.0.


# train_multilang2.sh is for multi-language training of neural nets.  It
# takes multiple egs directories which must be created by get_egs2.sh, and the
# corresponding alignment directories (only needed for training the transition
# models).

# for the n languages, we share all the hidden layers but there are separate
# final layers.  On each iteration of training we average the hidden layers
# across all jobs of all languages, but average the parameters of the final,
# output layer only within each language.  The script starts from a partially
# trained model from the first language (language 0 in the directory-numbering
# scheme).  See egs/rm/s5/local/online/run_nnet2_wsj_joint.sh for example.
#
# This script requires you to supply a neural net partially trained for the 1st
# language, by one of the regular training scripts, to be used as the initial
# neural net (for use by other languages, we'll discard the last layer); it
# should not have been subject to "mix-up" (since this script does mix-up), or
# combination (since it would increase the parameter range to a too-large value
# which isn't compatible with our normal learning rate schedules).


# Begin configuration section.
cmd=run.pl
num_epochs=10      # Number of epochs of training (for first language);
                   # the number of iterations is worked out from this.
initial_learning_rate=0.04
final_learning_rate=0.004

minibatch_size=128 # by default use a smallish minibatch size for neural net
                   # training; this controls instability which would otherwise
                   # be a problem with multi-threaded update. 

num_jobs_nnet="2 2"    # Number of neural net jobs to run in parallel.  This option
                       # is passed to get_egs.sh.  Array must be same length
                       # as number of separate languages.
num_jobs_compute_prior=10 # these are single-threaded, run on CPU.

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

prior_subset_size=10000 # 10k samples per job, for computing priors.  Should be
                        # more than enough.

stage=-4


mix_up="0 0" # Number of components to mix up to (should be > #tree leaves, if
             # specified.)  An array, one per language.

num_threads=16  # default suitable for CPU-based training
parallel_opts="--num-threads 16 --mem 1G"  # default suitable for CPU-based training.
  # by default we use 16 threads; this lets the queue know.
  # note: parallel_opts doesn't automatically get adjusted if you adjust num-threads.
combine_num_threads=8
combine_parallel_opts="--num-threads 8"  # queue options for the "combine" stage.
cleanup=false # while testing, leaving cleanup=false.
# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 6 -o $[$#%2] -ne 0 ]; then
  # num-args must be at least 6 and must be even.
  echo "Usage: $0 [opts] <ali0> <egs0> <ali1> <egs1> ... <aliN-1> <egsN-1> <input-model> <exp-dir>"
  echo " e.g.: $0 data/train exp/tri6_ali exp/tri6_egs exp_lang2/tri6_ali exp_lang2/tri6_egs exp/dnn6a/10.mdl exp/tri6_multilang"
  echo ""
  echo "Note: <input-model> must correspond to the model/tree for <ali0> and <egs0>, and the"
  echo "num-epochs is computed for the zeroth language."
  echo ""
  echo "The --num-jobs-nnet should be an array saying how many jobs to allocate to each language,"
  echo "e.g. --num-jobs-nnet '2 4'"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-epochs <#epochs|15>                        # Number of epochs of training (figured from 1st corpus)"
  echo "  --initial-learning-rate <initial-learning-rate|0.02> # Learning rate at start of training, e.g. 0.02 for small"
  echo "                                                       # data, 0.01 for large data"
  echo "  --final-learning-rate  <final-learning-rate|0.004>   # Learning rate at end of training, e.g. 0.004 for small"
  echo "                                                   # data, 0.001 for large data"
  echo "  --num-hidden-layers <#hidden-layers|2>           # Number of hidden layers, e.g. 2 for 3 hours of data, 4 for 100hrs"
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
  echo "  --parallel-opts <opts|\"--num-threads 16 --mem 1G\">      # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads... "
  echo "  --stage <stage|-4>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  exit 1;
fi


argv=("$@") 
num_args=$#
num_lang=$[($num_args-2)/2]

dir=${argv[$num_args-1]}
input_model=${argv[$num_args-2]}

[ ! -f $input_model ] && echo "$0: Input model $input_model does not exist" && exit 1;


mkdir -p $dir/log

num_jobs_nnet_array=($num_jobs_nnet)
! [ "${#num_jobs_nnet_array[@]}" -eq "$num_lang" ] && \
  echo "$0: --num-jobs-nnet option must have size equal to the number of languages" && exit 1;
mix_up_array=($mix_up)
! [ "${#mix_up_array[@]}" -eq "$num_lang" ] && \
  echo "$0: --mix-up option must have size equal to the number of languages" && exit 1;


# Language index starts from 0.
for lang in $(seq 0 $[$num_lang-1]); do
  alidir[$lang]=${argv[$lang*2]}
  egs_dir[$lang]=${argv[$lang*2+1]}
  for f in ${egs_dir[$lang]}/info/frames_per_eg ${egs_dir[lang]}/egs.1.ark ${alidir[$lang]}/ali.1.gz ${alidir[$lang]}/tree; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done
  mkdir -p $dir/$lang/log
  cp ${alidir[$lang]}/tree $dir/$lang/ || exit 1;

  for f in ${egs_dir[$lang]}/{final.mat,cmvn_opts,splice_opts}; do
    # Copy any of these files that exist.
    cp $f $dir/$lang/ 2>/dev/null 
  done
done


input_model_pdfs=$(nnet-am-info $input_model | grep '^output-dim' | awk '{print $2}')
alidir0_pdfs=$(tree-info ${alidir[0]}/tree | grep '^num-pdfs' | awk '{print $2}')
if ! [ $input_model_pdfs -eq $alidir0_pdfs ]; then
  echo "$0: expected num-pdfs from the input model $input_model to match"
  echo " .. the one used for the first alignment directory ${alidir[0]}, $input_model_pdfs != $alidir0_pdfs"
  exit 1;
fi



for x in final.mat cmvn_opts splice_opts; do
  if [ -f $dir/0/$x ]; then
    for lang in $(seq 1 $[$num_lang-1]); do
      if ! cmp $dir/0/$x $dir/$lang/$x; then
        echo "$0: warning: files $dir/0/$x and $dir/$lang/$x are not identical."
      fi
    done
  fi
done

# the input model is supposed to correspond to the first language.
nnet-am-copy --learning-rate=$initial_learning_rate $input_model $dir/0/0.mdl

if nnet-am-info --print-args=false $dir/0/0.mdl | grep SumGroupComponent 2>/dev/null; then
  if [ "${mix_up_array[0]}" != "0" ]; then
    echo "$0: Your input model already has mixtures, but you are asking to mix it up."
    echo " ... best to use a model without mixtures as input.  (e.g., earlier iter)."
    exit 1;
  fi
fi


if [ $stage -le -4 ]; then
  echo "$0: initializing models for other languages"
  for lang in $(seq 1 $[$num_lang-1]); do
    # create the initial models for the other languages.
    $cmd $dir/$lang/log/reinitialize.log \
      nnet-am-reinitialize $input_model ${alidir[$lang]}/final.mdl $dir/$lang/0.mdl || exit 1;
  done
fi

if [ $stage -le -3 ]; then
  echo "Training transition probabilities and setting priors"
  for lang in $(seq 0 $[$num_lang-1]); do
    $cmd $dir/$lang/log/train_trans.log \
      nnet-train-transitions $dir/$lang/0.mdl "ark:gunzip -c ${alidir[$lang]}/ali.*.gz|" $dir/$lang/0.mdl \
      || exit 1;
  done
fi

# Work out the number of iterations... the number of epochs refers to the
# first language (language zero) and this, together with the num-jobs-nnet for
# that language and details of the egs, determine the number of epochs.

frames_per_eg0=$(cat ${egs_dir[0]}/info/frames_per_eg) || exit 1;
num_archives0=$(cat ${egs_dir[0]}/info/num_archives) || exit 1;
# num_archives_expanded considers each separate label-position from
# 0..frames_per_eg-1 to be a separate archive.
num_archives_expanded0=$[$num_archives0*$frames_per_eg0]

if [ ${num_jobs_nnet_array[0]} -gt $num_archives_expanded0 ]; then
  echo "$0: --num-jobs-nnet[0] cannot exceed num-archives*frames-per-eg which is $num_archives_expanded"
  exit 1;
fi

# set num_iters so that as close as possible, we process the data $num_epochs
# times, i.e. $num_iters*$num_jobs_nnet == $num_epochs*$num_archives_expanded
num_iters=$[($num_epochs*$num_archives_expanded0)/${num_jobs_nnet_array[0]}]

echo "$0: Will train for $num_epochs epochs (of language 0) = $num_iters iterations"

! [ $num_iters -gt 0 ] && exit 1;

# Work out the number of epochs we train for on the other languages... this is
# just informational.
for lang in $(seq 1 $[$num_lang-1]); do
  frames_per_eg=$(cat ${egs_dir[$lang]}/info/frames_per_eg) || exit 1;
  num_archives=$(cat ${egs_dir[$lang]}/info/num_archives) || exit 1;
  num_archives_expanded=$[$num_archives*$frames_per_eg]
  num_epochs=$[($num_iters*${num_jobs_nnet_array[$lang]})/$num_archives_expanded]
  echo "$0: $num_iters iterations is approximately $num_epochs epochs for language $lang"
done

# do any mixing-up after half the iters.
mix_up_iter=$[$num_iters/2]

if [ $num_threads -eq 1 ]; then
  parallel_suffix="-simple" # this enables us to use GPU code if
                         # we have just one thread.
  parallel_train_opts=
  if ! cuda-compiled; then
    echo "$0: WARNING: you are running with one thread but you have not compiled"
    echo "   for CUDA.  You may be running a setup optimized for GPUs.  If you have"
    echo "   GPUs and have nvcc installed, go to src/ and do ./configure; make"
  fi
else
  parallel_suffix="-parallel"
  parallel_train_opts="--num-threads=$num_threads"
fi


approx_iters_per_epoch=$[$num_iters/$num_epochs]
# First work out how many models we want to combine over in the final
# nnet-combine-fast invocation.  This equals
# min(max(max_models_combine, iters_per_epoch),
#     2/3 * iters_after_mixup).
# We use the same numbers of iterations for all languages, even though it's just
# worked out for the first language.
num_models_combine=$max_models_combine
if [ $num_models_combine -lt $approx_iters_per_epoch ]; then
  num_models_combine=$approx_iters_per_epoch
fi
iters_after_mixup_23=$[(($num_iters-$mix_up_iter-1)*2)/3]
if [ $num_models_combine -gt $iters_after_mixup_23 ]; then
  num_models_combine=$iters_after_mixup_23
fi
first_model_combine=$[$num_iters-$num_models_combine+1]

x=0


while [ $x -lt $num_iters ]; do
    
  if [ $x -ge 0 ] && [ $stage -le $x ]; then
    for lang in $(seq 0 $[$num_lang-1]); do
      # Set off jobs doing some diagnostics, in the background.
      $cmd $dir/$lang/log/compute_prob_valid.$x.log \
        nnet-compute-prob $dir/$lang/$x.mdl ark:${egs_dir[$lang]}/valid_diagnostic.egs &
      $cmd $dir/$lang/log/compute_prob_train.$x.log \
        nnet-compute-prob $dir/$lang/$x.mdl ark:${egs_dir[$lang]}/train_diagnostic.egs &
      if [ $x -gt 0 ] && [ ! -f $dir/$lang/log/mix_up.$[$x-1].log ]; then
        $cmd $dir/$lang/log/progress.$x.log \
          nnet-show-progress --use-gpu=no $dir/$lang/$[$x-1].mdl $dir/$lang/$x.mdl \
          ark:${egs_dir[$lang]}/train_diagnostic.egs '&&' \
           nnet-am-info $dir/$lang/$x.mdl &
      fi
    done

    echo "Training neural net (pass $x)"

    if [ $x -eq 0 ]; then
      # on iteration zero, use a smaller minibatch size and only one quarter of the
      # normal amount of training data: this will help, respectively, to ensure stability
      # and to stop the models from moving so far that averaging hurts.
      this_minibatch_size=$[$minibatch_size/2];
      this_keep_proportion=0.25
    else
      this_minibatch_size=$minibatch_size
      this_keep_proportion=1.0
      # use half the examples on iteration 1, out of a concern that the model-averaging
      # might not work if we move too far before getting close to convergence.
      [ $x -eq 1 ] && this_keep_proportion=0.5 
    fi

    rm $dir/.error 2>/dev/null


    ( # this sub-shell is so that when we "wait" below,
      # we only wait for the training jobs that we just spawned,
      # not the diagnostic jobs that we spawned above.
      
      # We can't easily use a single parallel SGE job to do the main training,
      # because the computation of which archive and which --frame option
      # to use for each job is a little complex, so we spawn each one separately.
      
      
      for lang in $(seq 0 $[$num_lang-1]); do
        this_num_jobs_nnet=${num_jobs_nnet_array[$lang]}
        this_frames_per_eg=$(cat ${egs_dir[$lang]}/info/frames_per_eg) || exit 1;
        this_num_archives=$(cat ${egs_dir[$lang]}/info/num_archives) || exit 1;

        ! [ $this_num_jobs_nnet -gt 0 -a $this_frames_per_eg -gt 0 -a $this_num_archives -gt 0 ] && exit 1

        for n in $(seq $this_num_jobs_nnet); do
          k=$[$x*$this_num_jobs_nnet + $n - 1]; # k is a zero-based index that we'll derive
                                                # the other indexes from.
          archive=$[($k%$this_num_archives)+1]; # work out the 1-based archive index.
          frame=$[(($k/$this_num_archives)%$this_frames_per_eg)];

          $cmd $parallel_opts $dir/$lang/log/train.$x.$n.log \
            nnet-train$parallel_suffix $parallel_train_opts \
            --minibatch-size=$this_minibatch_size --srand=$x $dir/$lang/$x.mdl \
            "ark,bg:nnet-copy-egs --keep-proportion=$this_keep_proportion --frame=$frame ark:${egs_dir[$lang]}/egs.$archive.ark ark:-|nnet-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x ark:- ark:-|" \
            $dir/$lang/$[$x+1].$n.mdl || touch $dir/.error &
        done
      done
      wait
    )
    # the error message below is not that informative, but $cmd will
    # have printed a more specific one.
    [ -f $dir/.error ] && echo "$0: error on iteration $x of training" && exit 1;


    learning_rate=`perl -e '($x,$n,$i,$f)=@ARGV; print ($x >= $n ? $f : $i*exp($x*log($f/$i)/$n));' $[$x+1] $num_iters $initial_learning_rate $final_learning_rate`;

    (
      # First average within each language.  Use a sub-shell so "wait" won't
      # wait for the diagnostic jobs.
      for lang in $(seq 0 $[$num_lang-1]); do
        this_num_jobs_nnet=${num_jobs_nnet_array[$lang]}
        nnets_list=$(for n in `seq 1 $this_num_jobs_nnet`; do echo $dir/$lang/$[$x+1].$n.mdl; done)
        # average the output of the different jobs.
        $cmd $dir/$lang/log/average.$x.log \
          nnet-am-average $nnets_list - \| \
          nnet-am-copy --learning-rate=$learning_rate - $dir/$lang/$[$x+1].tmp.mdl || touch $dir/.error &
      done
      wait
      [ -f $dir/.error ] && echo "$0: error averaging models on iteration $x of training" && exit 1;
      # Remove the models we just averaged.
      for lang in $(seq 0 $[$num_lang-1]); do
        this_num_jobs_nnet=${num_jobs_nnet_array[$lang]}
        for n in `seq 1 $this_num_jobs_nnet`; do rm $dir/$lang/$[$x+1].$n.mdl; done
      done
    )


    nnets_list=$(for lang in $(seq 0 $[$num_lang-1]); do echo $dir/$lang/$[$x+1].tmp.mdl; done)
    weights_csl=$(echo $num_jobs_nnet | sed 's/ /:/g') # get as colon separated list.

    # the next command produces the cross-language averaged model containing the
    # final layer corresponding to language zero.
    $cmd $dir/log/average.$x.log \
      nnet-am-average --weights=$weights_csl --skip-last-layer=true \
      $nnets_list $dir/0/$[$x+1].mdl || exit 1;

    for lang in $(seq 1 $[$num_lang-1]); do
      # the next command takes the averaged hidden parameters from language zero, and
      # the last layer from language $lang.  It's not really doing averaging.
      $cmd $dir/$lang/log/combine_average.$x.log \
        nnet-am-average --weights=0.0:1.0 --skip-last-layer=true \
          $dir/$lang/$[$x+1].tmp.mdl $dir/0/$[$x+1].mdl $dir/$lang/$[$x+1].mdl || exit 1;
    done

    $cleanup && rm $dir/*/$[$x+1].tmp.mdl

    if [ $x -eq $mix_up_iter ]; then
      for lang in $(seq 0 $[$num_lang-1]); do     
        this_mix_up=${mix_up_array[$lang]}
        if [ $this_mix_up -gt 0 ]; then
          echo "$0: for language $lang, mixing up to $this_mix_up components"
          $cmd $dir/$lang/log/mix_up.$x.log \
            nnet-am-mixup --min-count=10 --num-mixtures=$this_mix_up \
             $dir/$lang/$[$x+1].mdl $dir/$lang/$[$x+1].mdl || exit 1;
        fi
      done
    fi

    # Now average across languages.

    rm $nnets_list

    for lang in $(seq 0 $[$num_lang-1]); do # mix up.
      [ ! -f $dir/$lang/$[$x+1].mdl ] && echo "No such file $dir/$lang/$[$x+1].mdl" && exit 1;
      if [ -f $dir/$lang/$[$x-1].mdl ] && $cleanup && \
        [ $[($x-1)%100] -ne 0  ] && [ $[$x-1] -lt $first_model_combine ]; then
        rm $dir/$lang/$[$x-1].mdl
      fi
    done
  fi
  x=$[$x+1]
done


if [ $stage -le $num_iters ]; then
  echo "$0: Doing combination to produce final models"


  rm $dir/.error 2>/dev/null
  for lang in $(seq 0 $[$num_lang-1]); do
    nnets_list=()
    # the if..else..fi statement below sets 'nnets_list'.
    if [ $max_models_combine -lt $num_models_combine ]; then
      # The number of models to combine is too large, e.g. > 20.  In this case,
      # each argument to nnet-combine-fast will be an average of multiple models.
      cur_offset=0 # current offset from first_model_combine.
      for n in $(seq $max_models_combine); do
        next_offset=$[($n*$num_models_combine)/$max_models_combine]
        sub_list="" 
        for o in $(seq $cur_offset $[$next_offset-1]); do
          iter=$[$first_model_combine+$o]
          mdl=$dir/$lang/$iter.mdl
          [ ! -f $mdl ] && echo "$0: Expected $mdl to exist" && exit 1;
          sub_list="$sub_list $mdl"
        done
        nnets_list[$[$n-1]]="nnet-am-average $sub_list - |"
        cur_offset=$next_offset
      done
    else
      nnets_list=
      for n in $(seq 0 $[num_models_combine-1]); do
        iter=$[$first_model_combine+$n]
        mdl=$dir/$lang/$iter.mdl
        [ ! -f $mdl ] && echo "$0: Expected $mdl to exist" && exit 1;
        nnets_list[$n]=$mdl
      done
    fi

    # Below, use --use-gpu=no to disable nnet-combine-fast from using a GPU, as
    # if there are many models it can give out-of-memory error; set num-threads
    # to 8 to speed it up (this isn't ideal...)
    num_egs=`nnet-copy-egs ark:${egs_dir[$lang]}/combine.egs ark:/dev/null 2>&1 | tail -n 1 | awk '{print $NF}'`

    mb=$[($num_egs+$combine_num_threads-1)/$combine_num_threads]
    [ $mb -gt 512 ] && mb=512
    # Setting --initial-model to a large value makes it initialize the combination
    # with the average of all the models.  It's important not to start with a
    # single model, or, due to the invariance to scaling that these nonlinearities
    # give us, we get zero diagonal entries in the fisher matrix that
    # nnet-combine-fast uses for scaling, which after flooring and inversion, has
    # the effect that the initial model chosen gets much higher learning rates
    # than the others.  This prevents the optimization from working well.
    $cmd $combine_parallel_opts $dir/$lang/log/combine.log \
      nnet-combine-fast --initial-model=100000 --num-lbfgs-iters=40 --use-gpu=no \
        --num-threads=$combine_num_threads \
        --verbose=3 --minibatch-size=$mb "${nnets_list[@]}" ark:${egs_dir[$lang]}/combine.egs \
      - \| nnet-normalize-stddev - $dir/$lang/final.mdl || touch $dir/.error &
  done
  wait
  
  [ -f $dir/.error ] && echo "$0: error doing model combination" && exit 1;
fi


if [ $stage -le $[$num_iters+1] ]; then
  for lang in $(seq 0 $[$num_lang-1]); do  
    # Run the diagnostics for the final models.
    $cmd $dir/$lang/log/compute_prob_valid.final.log \
      nnet-compute-prob $dir/$lang/final.mdl ark:${egs_dir[$lang]}/valid_diagnostic.egs &
    $cmd $dir/$lang/log/compute_prob_train.final.log \
      nnet-compute-prob $dir/$lang/final.mdl ark:${egs_dir[$lang]}/train_diagnostic.egs &
  done
  wait
fi

if [ $stage -le $[$num_iters+2] ]; then
  # Note: this just uses CPUs, using a smallish subset of data.


  for lang in $(seq 0 $[$num_lang-1]); do
    echo "$0: Getting average posterior for purposes of adjusting the priors (language $lang)."
    rm $dir/$lang/.error 2>/dev/null
    rm $dir/$lang/post.$x.*.vec 2>/dev/null
    $cmd JOB=1:$num_jobs_compute_prior $dir/$lang/log/get_post.JOB.log \
      nnet-copy-egs --frame=random --srand=JOB ark:${egs_dir[$lang]}/egs.1.ark ark:- \| \
      nnet-subset-egs --srand=JOB --n=$prior_subset_size ark:- ark:- \| \
      nnet-compute-from-egs "nnet-to-raw-nnet $dir/$lang/final.mdl -|" ark:- ark:- \| \
      matrix-sum-rows ark:- ark:- \| vector-sum ark:- $dir/$lang/post.JOB.vec || touch $dir/$lang/.error &
  done
  echo "$0: ... waiting for jobs for all languages to complete."
  wait
  sleep 3;  # make sure there is time for $dir/$lang/post.$x.*.vec to appear.
  for lang in $(seq 0 $[$num_lang-1]); do
    [ -f $dir/$lang/.error ] && \
      echo "$0: error getting posteriors for adjusting the priors for language $lang" && exit 1;

    $cmd $dir/$lang/log/vector_sum.log \
      vector-sum $dir/$lang/post.*.vec $dir/$lang/post.vec || exit 1;

    rm $dir/$lang/post.*.vec;

    echo "Re-adjusting priors based on computed posteriors for language $lang"
    $cmd $dir/$lang/log/adjust_priors.final.log \
      nnet-adjust-priors $dir/$lang/final.mdl $dir/$lang/post.vec $dir/$lang/final.mdl || exit 1;
  done
fi


for lang in $(seq 0 $[$num_lang-1]); do
  if [ ! -f $dir/$lang/final.mdl ]; then
    echo "$0: $dir/final.mdl does not exist."
    # we don't want to clean up if the training didn't succeed.
    exit 1;
  fi
done

sleep 2

echo Done

if $cleanup; then
  echo Cleaning up data
  if [[ $egs_dir =~ $dir/egs* ]]; then
    steps/nnet2/remove_egs.sh $egs_dir
  fi

  echo Removing most of the models
  for x in `seq 0 $num_iters`; do
    if [ $[$x%100] -ne 0 ] && [ $x -ne $num_iters ] && [ -f $dir/$lang/$x.mdl ]; then
       # delete all but every 100th model; don't delete the ones which combine to form the final model.
      rm $dir/$lang/$x.mdl
    fi
  done
fi

exit 0
