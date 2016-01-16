#!/bin/bash

# Copyright 2015 Hossein Hadian

num_epochs=15
minibatch_size=512
compute_prob_interval=2
nnet_config=
cmd=run.pl
use_gpu=true    # if true, we run on GPU.
stage=0

egs_opts=
nnet_opts=
max_duration=0
left_context=4
right_context=2
min_repeat_count=5

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [[ $# != 3 ]]; then
   echo "Usage: $0 [options] <phones-dir> <ali-dir> <duration-model-dir>"
   echo "e.g.: $0 data/lang/phones exp/mono_ali exp/mono/durmod"
   echo ""
   echo "Main options (for others, see top of script file):"
   echo "  --num-epochs <number>                       # max number of epochs for training"
   echo "  --minibatch-size <size>                     # minibatch size"
   echo "  --compute-prob-interval <int>               # interval for measuring accuracy (diagnostics)"
   echo "  --nnet-config <nnet3-conf-file>             # use this config for training"
   echo "Options related to initializing (stage 0):"
   echo "  --max-duration <duration-in-frames>         # max duration; if not set, it will be determined automatically"
   echo "  --left-context <size>                       # left phone context size"
   echo "  --right-context <size>                      # right phone context size"
   exit 1;
fi

phones_dir=$1
alidir=$2
dir=$3

durmodel=$dir/durmodel.mdl
mkdir -p $dir/log

if [ $stage -le 0 ]; then
  echo "$0: Initializing the duration model and nnet..."

  transmodel=$alidir/final.mdl
  for f in $transmodel $phones_dir/roots.int $phones_dir/extra_questions.int; do
    [ ! -f $f ] && echo "$0: Required file for initializing not found: $f" && exit 1;
  done

  if [ $max_duration == 0 ]; then
    echo "$0: Determining max-duration..."
    if [ ! -f $dir/all.ali ]; then
      gunzip -c $alidir/ali.*.gz >> $dir/all.ali || exit 1;
    fi

    max_duration=`ali-to-phones --write-lengths $transmodel ark:$dir/all.ali \
                  ark,t:- | awk -v min_count=$min_repeat_count -F';' \
                  '{ for(i=1; i<=NF; i++){ \
                       n=split($(i), a, " ");\
                       duration=a[n];\
                       counts[duration]++;\
                       if(duration>max_duration) max_duration=duration; \
                   }} END { for(d=3; d<=max_duration; d++) \
                              if(counts[d]<min_count){\
                                print d-1; exit 0;\
                              }\
                            print max_duration;\
                            exit 0;}'` || exit 1;

  fi
  echo "$0: Max duration: $max_duration"

  $cmd $dir/log/durmod_init.log \
       durmodel-init --max-duration=$max_duration \
                     --left-context=$left_context \
                     --right-context=$right_context \
                     $phones_dir/roots.int $phones_dir/extra_questions.int \
                     $durmodel || exit 1;

fi

if [ $stage -le 1 ]; then
  echo "$0: Preparing examples..."
  steps/dur_model/get_examples.sh --cmd $cmd $egs_opts $alidir $dir || exit 1;
fi


if [ $stage -le 2 ]; then

  [ ! -f $durmodel ] && echo "$0: Duration model file not found (have you completed stage 0?): $durmodel" && exit 1;
  [ ! -f $dir/train.egs ] && echo "$0: Train examples file not found (have you completed stage 1?): $dir/train.egs" && exit 1;
  [ ! -f $dir/val.egs ] && echo "$0: Validation examples file not found (have you completed stage 1?): $dir/val.egs" && exit 1;

  if $use_gpu; then
    train_queue_opt="--gpu 1"
    if [ $cmd != "queue.pl" ]; then
      echo "$0: WARNING: you are running on GPU but you are not using queue.pl."
      echo -n "Are you sure you are not in a cluster? (if yes press ENTER, else Ctrl-c)"
      read OK
    fi
    if ! cuda-compiled; then
      echo "$0: WARNING: you are running with one thread but you have not compiled"
      echo "   for CUDA.  You may be running a setup optimized for GPUs.  If you have"
      echo "   GPUs and have nvcc installed, go to src/ and do ./configure; make"
      exit 1
    fi
  else
    parallel_train_opts="--use-gpu=no"
  fi

  # TODO(hhadian): config is appiled here. Should it be moved to stage 0?
  # At stage 0, the user does not know the feat dim and ouput dim.
  if [[ ! -z $nnet_config ]]; then
    echo "$0: Using provided config file for nnet."
  else
    nnet_config=$dir/nnet.conf
    feat_dim=$(durmodel-info $durmodel | grep feature-dim | awk '{ print $2 }')
    output_dim=$(durmodel-info $durmodel | grep max-duration | awk '{ print $2 }')
    steps/dur_model/make_nnet_config.sh $nnet_opts $feat_dim $output_dim >$nnet_config
    echo "$0: Wrote nnet config to "$nnet_config
  fi

  $cmd $dir/log/nnet_init.log \
       nnet3-init $nnet_config $dir/nnet.raw || exit 1;
  $cmd $dir/log/durmod_set_raw_nnet.log \
       nnet3-durmodel-init $durmodel $dir/nnet.raw $dir/0.mdl || exit 1;

  for epoch in $(seq 0 $[$num_epochs-1]); do
    echo "Epoch: "$epoch
    curr_mdl=$dir/$[$epoch].mdl
    next_mdl=$dir/$[$epoch+1].mdl

    $cmd $train_queue_opt $dir/log/train_$epoch.log \
         nnet3-train $parallel_train_opts "nnet3-durmodel-copy --raw=true $curr_mdl -|" \
              "ark:nnet3-shuffle-egs --srand=$epoch \
              'ark:for n in 1 2 3 4 5; do cat $dir/train.egs; done |' ark:-| \
              nnet3-merge-egs --minibatch-size=$minibatch_size ark:- ark:- |" \
              $dir/nnet.raw || exit 1;

    # grep Overall $dir/log/train_$epoch.log
    $cmd $dir/log/durmod_set_raw_nnet.log \
         nnet3-durmodel-copy --set-raw-nnet=$dir/nnet.raw $curr_mdl $next_mdl

    if [[ $[$epoch%$compute_prob_interval] == 0 ]]; then
      $cmd $dir/log/compute_prob_$[$epoch+1].log \
         nnet3-compute-prob $dir/nnet.raw \
         "ark:nnet3-merge-egs --minibatch-size=$minibatch_size ark:$dir/val.egs ark:- |" \
         && grep -o -H "Overall.*" $dir/log/compute_prob_$[$epoch+1].log &
    fi
  done # training loop
  wait $! # wait for last background process if any
fi # stage 1
