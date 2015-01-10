#!/bin/bash

# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This script does MPE or MMI or state-level minimum bayes risk (sMBR) training,
# in the multi-language or at least multi-model setting where you have multiple "degs" directories.
# The input "degs" directories must be dumped by one of the get_egs_discriminative2.sh scripts.

# Begin configuration section.
cmd=run.pl
num_epochs=4       # Number of epochs of training
learning_rate=0.00002
acoustic_scale=0.1  # acoustic scale for MMI/MPFE/SMBR training.
boost=0.0       # option relevant for MMI

criterion=smbr
drop_frames=false #  option relevant for MMI
num_jobs_nnet="4 4"    # Number of neural net jobs to run in parallel, one per
                       # language..  Note: this will interact with the learning
                       # rates (if you decrease this, you'll have to decrease
                       # the learning rate, and vice versa).

modify_learning_rates=true
last_layer_factor=1.0  # relates to modify-learning-rates
first_layer_factor=1.0 # relates to modify-learning-rates
shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.


stage=-3


num_threads=16  # this is the default but you may want to change it, e.g. to 1 if
                # using GPUs.
cleanup=true
retroactive=false
remove_egs=false
src_models=  # can be used to override the defaults of <degs-dir1>/final.mdl <degs-dir2>/final.mdl .. etc.
             # set this to a space-separated list.
# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# -lt 3 ]; then
  echo "Usage: $0 [opts] <degs-dir1> <degs-dir2> ... <degs-dirN>  <exp-dir>"
  echo " e.g.: $0 exp/tri4_mpe_degs exp_other_lang/tri4_mpe_degs exp/tri4_mpe_multilang"
  echo ""
  echo "You have to first call get_egs_discriminative2.sh to dump the egs."
  echo "Caution: the options 'drop_frames' and 'criterion' are taken here"
  echo "even though they were required also by get_egs_discriminative2.sh,"
  echo "and they should normally match."
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-epochs <#epochs|4>                        # Number of epochs of training (measured on language 0)"
  echo "  --learning-rate <learning-rate|0.0002>           # Learning rate to use"
  echo "  --num-jobs-nnet <num-jobs|4 4>                   # Number of parallel jobs to use for main neural net:"
  echo "                                                   # space separated list of num-jobs per language. Affects"
  echo "                                                   # relative weighting."
  echo "  --num-threads <num-threads|16>                   # Number of parallel threads per job (will affect results"
  echo "                                                   # as well as speed; may interact with batch size; if you increase"
  echo "                                                   # this, you may want to decrease the batch size.  With GPU, must be 1."
  echo "  --parallel-opts <opts|\"-pe smp 16 -l ram_free=1G,mem_free=1G\">      # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads... note, you might have to reduce mem_free,ram_free"
  echo "                                                   # versus your defaults, because it gets multiplied by the -pe smp argument."
  echo "  --stage <stage|-3>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  echo "  --criterion <criterion|smbr>                     # Training criterion: may be smbr, mmi or mpfe"
  echo "  --boost <boost|0.0>                              # Boosting factor for MMI (e.g., 0.1)"
  echo "  --drop-frames <true,false|false>                 # Option that affects MMI training: if true, we exclude gradients from frames"
  echo "                                                   # where the numerator transition-id is not in the denominator lattice."
  echo "  --modify-learning-rates <true,false|false>       # If true, modify learning rates to try to equalize relative"
  echo "                                                   # changes across layers."
  exit 1;
fi

argv=("$@") 
num_args=$#
num_lang=$[$num_args-1]

dir=${argv[$num_args-1]}

num_jobs_nnet_array=($num_jobs_nnet)
! [ "${#num_jobs_nnet_array[@]}" -eq "$num_lang" ] && \
  echo "$0: --num-jobs-nnet option must have size equal to the number of languages" && exit 1;

for lang in $(seq 0 $[$num_lang-1]); do
  degs_dir[$lang]=${argv[$lang]}
done

if [ ! -z "$src_models" ]; then
  src_model_array=($src_models)
  ! [ "${#src_model_array[@]}" -eq "$num_lang" ] && \
    echo "$0: --src-models option must have size equal to the number of languages" && exit 1;
else
  for lang in $(seq 0 $[$num_lang-1]); do
    src_model_array[$lang]=${degs_dir[$lang]}/final.mdl
  done
fi

mkdir -p $dir/log || exit 1;

for lang in $(seq 0 $[$num_lang-1]); do
  this_degs_dir=${degs_dir[$lang]}
  mdl=${src_model_array[$lang]}
  this_num_jobs_nnet=${num_jobs_nnet_array[$lang]}
  # Check inputs
  for f in $this_degs_dir/degs.1.ark $this_degs_dir/info/{num_archives,silence.csl,frames_per_archive} $mdl; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done
  mkdir -p $dir/$lang/log || exit 1;

  # check for valid num-jobs-nnet.
  ! [ $this_num_jobs_nnet -gt 0 ] && echo "Bad num-jobs-nnet option '$num_jobs_nnet'" && exit 1;
  this_num_archives=$(cat $this_degs_dir/info/num_archives) || exit 1;
  num_archives_array[$lang]=$this_num_archives
  silphonelist_array[$lang]=$(cat $this_degs_dir/info/silence.csl) || exit 1;

  if [ $this_num_jobs_nnet -gt $this_num_archives ]; then
    echo "$0: num-jobs-nnet $this_num_jobs_nnet exceeds number of archives $this_num_archives"
    echo " ... for language $lang; setting it to $this_num_archives."
    num_jobs_nnet_array[$lang]=$this_num_archives
  fi

  # copy some things from the input directories.
  for f in splice_opts cmvn_opts tree final.mat; do
    if [ -f $this_degs_dir/$f ]; then
      cp $this_degs_dir/$f $dir/$lang/ || exit 1;
    fi
  done
  if [ -f $this_degs_dir/conf ]; then
    ln -sf $(readlink -f $this_degs_dir/conf) $dir/ || exit 1; 
  fi
done


# work out number of iterations.
num_archives0=$(cat ${degs_dir[0]}/info/num_archives) || exit 1;
num_jobs_nnet0=${num_jobs_nnet_array[0]}

! [ $num_epochs -gt 0 ] && echo "Error: num-epochs $num_epochs is not valid" && exit 1;


num_iters=$[($num_epochs*$num_archives0)/$num_jobs_nnet0]

echo "$0: Will train for $num_epochs epochs = $num_iters iterations (measured on language 0)"
# Work out the number of epochs we train for on the other languages... this is
# just informational.
for lang in $(seq 1 $[$num_lang-1]); do
  this_degs_dir=${degs_dir[$lang]}
  this_num_archives=${num_archives_array[$lang]}
  this_num_epochs=$[($num_iters*${num_jobs_nnet_array[$lang]})/$this_num_archives]
  echo "$0: $num_iters iterations is approximately $this_num_epochs epochs for language $lang"
done



if [ $stage -le -1 ]; then
  echo "$0: Copying initial models and modifying preconditioning setups"

  # Note, the baseline model probably had preconditioning, and we'll keep it;
  # but we want online preconditioning with a larger number of samples of
  # history, since in this setup the frames are only randomized at the segment
  # level so they are highly correlated.  It might make sense to tune this a
  # little, later on, although I doubt it matters once the --num-samples-history
  # is large enough.

  for lang in $(seq 0 $[$num_lang-1]); do
    $cmd $dir/$lang/log/convert.log \
      nnet-am-copy --learning-rate=$learning_rate ${src_model_array[$lang]} - \| \
      nnet-am-switch-preconditioning  --num-samples-history=50000 - $dir/$lang/0.mdl || exit 1;
  done
fi



if [ $num_threads -eq 1 ]; then
 train_suffix="-simple" # this enables us to use GPU code if
                        # we have just one thread.
else
  train_suffix="-parallel --num-threads=$num_threads"
fi


x=0   
while [ $x -lt $num_iters ]; do
  if [ $stage -le $x ]; then
    
    echo "Training neural net (pass $x)"


    rm $dir/.error 2>/dev/null

    for lang in $(seq 0 $[$num_lang-1]); do
      this_num_jobs_nnet=${num_jobs_nnet_array[$lang]}
      this_num_archives=${num_archives_array[$lang]}
      this_degs_dir=${degs_dir[$lang]}
      this_silphonelist=${silphonelist_array[$lang]}

      # The \$ below delays the evaluation of the expression until the script runs (and JOB
      # will be replaced by the job-id).  That expression in $[..] is responsible for
      # choosing the archive indexes to use for each job on each iteration... we cycle through
      # all archives.

      (
        $cmd JOB=1:$this_num_jobs_nnet $dir/$lang/log/train.$x.JOB.log \
          nnet-combine-egs-discriminative \
          "ark:$this_degs_dir/degs.\$[((JOB-1+($x*$this_num_jobs_nnet))%$this_num_archives)+1].ark" ark:- \| \
          nnet-train-discriminative$train_suffix --silence-phones=$this_silphonelist \
           --criterion=$criterion --drop-frames=$drop_frames \
           --boost=$boost --acoustic-scale=$acoustic_scale \
           $dir/$lang/$x.mdl ark:- $dir/$lang/$[$x+1].JOB.mdl || exit 1;

        nnets_list=$(for n in $(seq $this_num_jobs_nnet); do echo $dir/$lang/$[$x+1].$n.mdl; done)

        # produce an average just within this language.
        $cmd $dir/$lang/log/average.$x.log \
          nnet-am-average $nnets_list $dir/$lang/$[$x+1].tmp.mdl || exit 1;

        rm $nnets_list
      ) || touch $dir/.error &
    done
    wait
    [ -f $dir/.error ] && echo "$0: error on pass $x" && exit 1


    # apply the modify-learning-rates thing to the model for the zero'th language;
    # we'll use the resulting learning rates for the other languages.
    if $modify_learning_rates; then
      $cmd $dir/log/modify_learning_rates.$x.log \
        nnet-modify-learning-rates --retroactive=$retroactive \
        --last-layer-factor=$last_layer_factor \
        --first-layer-factor=$first_layer_factor \
        $dir/0/$x.mdl $dir/0/$[$x+1].tmp.mdl $dir/0/$[$x+1].tmp.mdl || exit 1;
    fi

    nnets_list=$(for lang in $(seq 0 $[$num_lang-1]); do echo $dir/$lang/$[$x+1].tmp.mdl; done)
    weights_csl=$(echo $num_jobs_nnet | sed 's/ /:/g') # get as colon separated list.

    # the next command produces the cross-language averaged model containing the
    # final layer corresponding to language zero.  Note, if we did modify-learning-rates,
    # it will also have the modified learning rates.
    $cmd $dir/log/average.$x.log \
      nnet-am-average --weights=$weights_csl --skip-last-layer=true \
      $nnets_list $dir/0/$[$x+1].mdl || exit 1;

    # we'll transfer these learning rates to the other models.
    learning_rates=$(nnet-am-info --print-learning-rates=true $dir/0/$[$x+1].mdl 2>/dev/null)        

    for lang in $(seq 1 $[$num_lang-1]); do
      # the next command takes the averaged hidden parameters from language zero, and
      # the last layer from language $lang.  It's not really doing averaging.
      # we use nnet-am-copy to transfer the learning rates from model zero.
      $cmd $dir/$lang/log/combine_average.$x.log \
        nnet-am-average --weights=0.0:1.0 --skip-last-layer=true \
          $dir/$lang/$[$x+1].tmp.mdl $dir/0/$[$x+1].mdl - \| \
        nnet-am-copy --learning-rates=$learning_rates - $dir/$lang/$[$x+1].mdl || exit 1;
    done

    $cleanup && rm $dir/*/$[$x+1].tmp.mdl

  fi

  x=$[$x+1]
done


for lang in $(seq 0 $[$num_lang-1]); do
  rm $dir/$lang/final.mdl 2>/dev/null
  ln -s $x.mdl $dir/$lang/final.mdl


  epoch_final_iters=
  for e in $(seq 0 $num_epochs); do
    x=$[($e*$num_archives0)/$num_jobs_nnet0] # gives the iteration number.
    ln -sf $x.mdl $dir/$lang/epoch$e.mdl
    epoch_final_iters="$epoch_final_iters $x"
  done

  if $cleanup; then
    echo "Removing most of the models for language $lang"
    for x in `seq 0 $num_iters`; do
      if ! echo $epoch_final_iters | grep -w $x >/dev/null; then 
        # if $x is not an epoch-final iteration..
        rm $dir/$lang/$x.mdl 2>/dev/null
      fi
    done
  fi
done


echo Done
