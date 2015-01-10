#!/bin/bash


# this script is discriminative training after multi-language training (as
# run_nnet2_gale_combined_disc1.sh), but the discriminative training is
# multi-language too. 
# some of the stages are the same as run_nnet2_gale_combined_disc1.sh,
# and we didn't repeat them (we used the --stage option, it defaults to 4).

# This script is to be run after run_nnet2_gale_combined.sh.  
# It's discriminative training, using just the BOLT data.
# note, the _filt data has some bad conversations removed, that
# weren't aligning.

train_stage=-10
stage=0
# dir is the base directory, as in ./run_nnet2_wsj_joint.sh
dir=exp/nnet2_online_wsj/nnet_ms_a
criterion=smbr
drop_frames=false # only relevant for MMI.
learning_rate=0.00005
data_wsj=../../wsj/s5/data/train_si284_max2
data_rm=data/train_max2
lang_wsj=../../wsj/s5/data/lang
lang_rm=data/lang
num_epochs=4

set -e


. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if [ $stage -le 1 ]; then
  nj=30
  sub_split=100
  num_threads=6

  steps/online/nnet2/make_denlats.sh --cmd "$decode_cmd -l mem_free=1G,ram_free=1G -pe smp $num_threads" \
      --nj $nj --sub-split $sub_split --num-threads "$num_threads" \
     $data_wsj $lang_wsj ${dir}_wsj_online ${dir}_wsj_denlats
fi


if [ $stage -le 2 ]; then
  # hardcode no-GPU for alignment, although you could use GPU [you wouldn't
  # get excellent GPU utilization though.]
  nj=200
  use_gpu=no
  gpu_opts=
  steps/online/nnet2/align.sh  --cmd "$decode_cmd $gpu_opts" --use-gpu "$use_gpu" \
      --nj $nj $data_wsj $lang_wsj ${dir}_wsj_online ${dir}_wsj_ali
fi

if [ $stage -le 3 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/egs/rm-$date/s5/${dir}_wsj_degs/storage ${dir}_wsj_degs/storage
  fi

  steps/online/nnet2/get_egs_discriminative2.sh \
    --cmd "$decode_cmd --max-jobs-run 10" \
    --criterion $criterion --drop-frames $drop_frames \
     $data_wsj $lang_wsj ${dir}_wsj_{ali,denlats,online,degs}
fi


if [ $stage -le 4 ]; then
  nj=30
  sub_split=100
  num_threads=6

  steps/online/nnet2/make_denlats.sh \
      --cmd "$decode_cmd -l mem_free=1G,ram_free=1G -pe smp $num_threads" \
      --nj $nj --sub-split $sub_split --num-threads "$num_threads" \
     $data_rm $lang_rm  ${dir}_rm_online ${dir}_rm_denlats
fi


if [ $stage -le 5 ]; then
  # hardcode no-GPU for alignment, although you could use GPU [you wouldn't
  # get excellent GPU utilization though.]
  nj=200
  use_gpu=no
  gpu_opts=
  steps/online/nnet2/align.sh  --cmd "$decode_cmd $gpu_opts" --use-gpu "$use_gpu" \
      --nj $nj $data_rm $lang_rm ${dir}_rm_online ${dir}_rm_ali
fi

if [ $stage -le 6 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/egs/rm-$date/s5/${dir}_rm_degs/storage ${dir}_rm_degs/storage
  fi

  steps/online/nnet2/get_egs_discriminative2.sh \
    --cmd "$decode_cmd --max-jobs-run 10" \
    --criterion $criterion --drop-frames $drop_frames \
    $data_rm $lang_rm ${dir}_rm_{ali,denlats,online,degs}
fi

if [ $stage -le 7 ]; then
  
  steps/nnet2/train_discriminative_multilang2.sh --cmd "$decode_cmd -l gpu=1" --stage $train_stage \
    --learning-rate $learning_rate --num-jobs-nnet "4 1" \
    --criterion $criterion --drop-frames $drop_frames \
    --num-epochs $num_epochs --num-threads 1 \
    ${dir}_wsj_degs ${dir}_rm_degs ${dir}_${criterion}_${learning_rate}
fi

if [ $stage -le 8 ]; then
  discdir=${dir}_${criterion}_${learning_rate}/1 # RM is directory number 1.
  ln -sf $(readlink -f ${dir}_rm_online/conf) $discdir/conf
  # ... so it acts like an online-decoding directory.

  for epoch in $(seq 0 $num_epochs); do
    steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
      --iter epoch$epoch exp/tri3b/graph data/test $discdir/decode_epoch$epoch &
    steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
      --iter epoch$epoch exp/tri3b/graph_ug data/test $discdir/decode_ug_epoch$epoch &
  done
  wait
fi

