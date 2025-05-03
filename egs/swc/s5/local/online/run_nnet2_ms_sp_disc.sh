#!/bin/bash


# This script does discriminative training on top of the online, multi-splice
# system trained in run_nnet2_ms.sh.
# note: this relies on having a cluster that has plenty of CPUs as well as GPUs,
# since the lattice generation runs in about real-time, so takes of the order of
# 1000 hours of CPU time.
# 
# Note: rather than using any features we have dumped on disk, this script
# regenerates them from the wav data three times-- when we do lattice
# generation, numerator alignment and discriminative training.  This made the
# script easier to write and more generic, because we don't have to know where
# the features and the iVectors are, but of course it's a little inefficient.
# The time taken is dominated by the lattice generation anyway, so this isn't
# a huge deal.

. cmd.sh


stage=0
train_stage=-10
use_gpu=true
criterion=smbr
drop_frames=false  # only matters for MMI anyway.
effective_lrate=0.000005
srcdir=
mic=ihm
num_jobs_nnet=6
train_stage=-10 # can be used to start training in the middle.
decode_start_epoch=0 # can be used to avoid decoding all epochs, e.g. if we decided to run more.
num_epochs=4
cleanup=false  # run with --cleanup true --stage 6 to clean up (remove large things like denlats,
               # alignments and degs).
gmm_dir=exp/$mic/tri4a

set -e
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts=" -l gpu=1,hostname='!g01*&!g02*' " #we want to submit to all.q as we use multiple GPUs for this 
  num_threads=1
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  parallel_opts="-pe smp $num_threads" 
fi

if [ -z $srcdir ]; then
  srcdir=exp/$mic/nnet2_online/nnet_ms_sp
fi

if [ ! -f ${srcdir}_online/final.mdl ]; then
  echo "$0: expected ${srcdir}_online/final.mdl to exist; first run run_nnet2_ms.sh."
  exit 1;
fi

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$gmm_dir/graph_${LM}

if [ $stage -le 1 ]; then
  nj=50  # this doesn't really affect anything strongly, except the num-jobs for one of
         # the phases of get_egs_discriminative2.sh below.
  num_threads_denlats=6
  subsplit=40 # number of jobs that run per job (but 2 run at a time, so total jobs is 80, giving
              # max total slots = 80 * 6 = 480.
  steps/nnet2/make_denlats.sh --cmd "$decode_cmd -l mem_free=1G,ram_free=1G -pe smp $num_threads_denlats" \
      --online-ivector-dir exp/$mic/nnet2_online/ivectors_train_hires_sp2 \
      --nj $nj --sub-split $subsplit --num-threads "$num_threads_denlats" --config conf/decode.conf \
     data/$mic/train_hires_sp data/lang $srcdir ${srcdir}_denlats || exit 1;

fi

if [ $stage -le 2 ]; then
  # hardcode no-GPU for alignment, although you could use GPU [you wouldn't
  # get excellent GPU utilization though.]
  nj=76 # have a high number of jobs because this could take a while, and we might
         # have some stragglers.
  use_gpu=no
  gpu_opts=

  steps/nnet2/align.sh  --cmd "$decode_cmd $gpu_opts" --use-gpu "$use_gpu" \
     --online-ivector-dir exp/$mic/nnet2_online/ivectors_train_hires_sp2 \
     --nj $nj data/$mic/train_hires_sp data/lang $srcdir ${srcdir}_ali || exit 1;

  # the command below is a more generic, but slower, way to do it.
  # steps/online/nnet2/align.sh --cmd "$decode_cmd $gpu_opts" --use-gpu "$use_gpu" \
  #    --nj $nj data/train_hires data/lang ${srcdir}_online ${srcdir}_ali || exit 1;
fi


if [ $stage -le 3 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${srcdir}_degs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{1,2,5,6}/$USER/kaldi-data/egs/ami-${mic}-$(date +'%m_%d_%H_%M')/s5/${srcdir}_degs/storage ${srcdir}_degs/storage
  fi
  # have a higher maximum num-jobs if
  if [ -d ${srcdir}_degs/storage ]; then max_jobs=10; else max_jobs=5; fi

  steps/nnet2/get_egs_discriminative2.sh \
    --stage 0 \
    --cmd "$decode_cmd -tc $max_jobs" \
    --online-ivector-dir exp/$mic/nnet2_online/ivectors_train_hires_sp2 \
    --criterion $criterion --drop-frames $drop_frames \
     data/$mic/train_hires_sp data/lang ${srcdir}{_ali,_denlats,/final.mdl,_degs} || exit 1;

  # the command below is a more generic, but slower, way to do it.
  #steps/online/nnet2/get_egs_discriminative2.sh \
  #  --cmd "$decode_cmd -tc $max_jobs" \
  #  --criterion $criterion --drop-frames $drop_frames \
  #   data/train_hires data/lang ${srcdir}{_ali,_denlats,_online,_degs} || exit 1;
fi

if [ $stage -le 4 ]; then
  steps/nnet2/train_discriminative2.sh --cmd "$decode_cmd $parallel_opts" \
    --stage $train_stage \
    --effective-lrate $effective_lrate \
    --criterion $criterion --drop-frames $drop_frames \
    --num-epochs $num_epochs \
    --num-jobs-nnet 6 --num-threads $num_threads \
      ${srcdir}_degs ${srcdir}_${criterion}_${effective_lrate} || exit 1;
fi

if [ $stage -le 5 ]; then
  dir=${srcdir}_${criterion}_${effective_lrate}
  ln -sf $(readlink -f ${srcdir}_online/conf) $dir/conf # so it acts like an online-decoding directory

  for epoch in $(seq $decode_start_epoch $num_epochs); do
    for decode_set in dev eval; do
      (
        num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
        decode_dir=$dir/decode_epoch${epoch}_${decode_set}_utt
        
        steps/online/nnet2/decode.sh --config conf/decode.conf --cmd "$decode_cmd" --nj $num_jobs \
        --per-utt true  --iter epoch$epoch $graph_dir data/$mic/${decode_set}_hires $decode_dir || exit 1
      ) &
    done
  done

  for epoch in $(seq $decode_start_epoch $num_epochs); do
    for decode_set in dev eval; do
      (
        num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
        decode_dir=$dir/decode_epoch${epoch}_${decode_set}_utt_offline
        
        steps/online/nnet2/decode.sh --config conf/decode.conf --cmd "$decode_cmd" --nj $num_jobs \
        --per-utt true --online false --iter epoch$epoch $graph_dir data/$mic/${decode_set}_hires $decode_dir || exit 1
      ) &
    done
  done
  
  wait
fi

if [ $stage -le 6 ] && $cleanup; then
  # if you run with "--cleanup true --stage 6" you can clean up.
  rm ${srcdir}_denlats/lat.*.gz || true
  rm ${srcdir}_ali/ali.*.gz || true
  steps/nnet2/remove_egs.sh ${srcdir}_degs || true
fi


exit 0;
