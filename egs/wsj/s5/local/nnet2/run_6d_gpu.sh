#!/bin/bash


# This script demonstrates discriminative training of p-norm neural nets.  It's on top
# of run_5d_gpu.sh, which uses adapted 40-dimensional features.  This version of
# the script uses GPUs.  We distinguish it by putting "_gpu" at the end of the
# directory name.


gpu_opts="-l gpu=1"                # This is suitable for the CLSP network,
                                   # you'll likely have to change it.  we'll
                                   # use it later on, in the training (it's
                                   # not used in denlat creation)
. ./cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF

# The denominator lattice creation currently doesn't use GPUs.

set -e # exit on error.

nj=$(cat exp/tri4b_ali_si284/num_jobs)

steps/nnet2/make_denlats.sh --cmd "$decode_cmd -l mem_free=1G,ram_free=1G" \
      --nj $nj --sub-split 20 --num-threads 6 --parallel-opts "-pe smp 6" \
      --transform-dir exp/tri4b_ali_si284 \
     data/train_si284 data/lang exp/nnet5d_gpu exp/nnet5d_gpu_denlats

steps/nnet2/align.sh  --cmd "$decode_cmd $gpu_opts" \
      --use-gpu yes --transform-dir exp/tri4b_ali_si284 \
      --nj $nj data/train_si284 data/lang exp/nnet5d_gpu exp/nnet5d_gpu_ali

steps/nnet2/train_discriminative.sh --cmd "$decode_cmd" --learning-rate 0.00002 \
    --num-jobs-nnet 4  --transform-dir exp/tri4b_ali_si284 \
    --num-threads 1 --parallel-opts "$gpu_opts" data/train_si284 data/lang \
    exp/nnet5d_gpu_ali exp/nnet5d_gpu_denlats exp/nnet5d_gpu/final.mdl exp/nnet6d_mpe_gpu

for epoch in 1 2 3 4; do
  dir=exp/nnet6d_mpe_gpu
  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 10 --iter epoch$epoch \
    --transform-dir exp/tri4b/decode_bd_tgpr_dev93 \
     exp/tri4b/graph_bd_tgpr data/test_dev93 $dir/decode_bd_tgpr_dev93_epoch$epoch &

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 8  --iter epoch$epoch \
    --transform-dir exp/tri4b/decode_bd_tgpr_eval92 \
     exp/tri4b/graph_bd_tgpr data/test_eval92 $dir/decode_bd_tgpr_eval92_epoch$epoch &
done



exit 0;
