#!/bin/bash


# This script demonstrates discriminative training of p-norm neural nets.
# It's on top of run_4c_gpu.sh which uses adapted 40-dimensional features.
# This version of the script uses GPUs.  We distinguish it by putting "_gpu"
# at the end of the directory name.


gpu_opts="-l gpu=1,hostname=g*"  # This is suitable for the CLSP network,
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

# Note: we specify 1G each for the mem_free and ram_free which, is per
# thread... it will likely be less than the default.  Increase the beam relative
# to the defaults; this is just for this RM setup, where the default beams will
# likely generate very thin lattices.  Note: the transform-dir is important to
# specify, since this system is on top of fMLLR features.

nj=$(cat exp/tri3b_ali/num_jobs)
dir=nnet4d_gpu
steps/nnet2/make_denlats.sh --cmd "$decode_cmd -l mem_free=1G,ram_free=1G" \
      --nj $nj --sub-split 20 --num-threads 6 --parallel-opts "-pe smp 6" \
      --beam 20.0 --lattice-beam 10.0 \
      --transform-dir exp/tri3b_ali \
     data/train data/lang exp/$dir exp/$dir_denlats

steps/nnet2/align.sh  --cmd "$decode_cmd $gpu_opts" --use-gpu yes \
      --transform-dir exp/tri3b_ali \
      --nj $nj data/train data/lang exp/$dir exp/$dir_ali

steps/nnet2/train_discriminative.sh --cmd "$decode_cmd" \
    --num-jobs-nnet 2 --transform-dir exp/tri3b_ali \
    --num-threads 1 --parallel-opts "$gpu_opts" data/train data/lang \
    exp/$dir_ali exp/$dir_denlats exp/$dir/final.mdl exp/nnet5d_mpe_gpu

for epoch in 1 2 3 4; do
   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --iter epoch$epoch \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/nnet5d_mpe_gpu/decode_epoch$epoch  &

   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --iter epoch$epoch \
     --transform-dir exp/tri3b/decode_ug \
     exp/tri3b/graph_ug data/test exp/nnet5d_mpe_gpu/decode_ug_epoch$epoch &
done


exit 0;



# The following is some test commands that I ran in order to verify that
# the neural-net splitting and excising code was working as intended.

# (
# acoustic_scale=0.1
# for criterion in smbr mmi mpfe; do
#   for drop_frames in true false; do
#     nnet-get-egs-discriminative  --drop-frames=$drop_frames  --criterion=$criterion --excise=true exp/tri5c_mpe/0.mdl 'ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:data/train/split8/1/utt2spk scp:data/train/split8/1/cmvn.scp "scp:head -n 40 data/train/split8/1/feats.scp|" ark:- | splice-feats --left-context=3 --right-context=3 ark:- ark:- | transform-feats exp/tri5c_mpe/final.mat ark:- ark:- | transform-feats --utt2spk=ark:data/train/split8/1/utt2spk ark:exp/tri3b_ali/trans.1 ark:- ark:- |' 'ark,s,cs:gunzip -c exp/$dir_ali/ali.1.gz |' 'ark,s,cs:gunzip -c exp/$dir_denlats/lat.1.gz|' "ark:|nnet-combine-egs-discriminative ark:- ark:1.egs"

#     nnet-get-egs-discriminative --drop-frames=$drop_frames --criterion=$criterion --split=false --excise=false exp/tri5c_mpe/0.mdl 'ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:data/train/split8/1/utt2spk scp:data/train/split8/1/cmvn.scp "scp:head -n 40 data/train/split8/1/feats.scp|" ark:- | splice-feats --left-context=3 --right-context=3 ark:- ark:- | transform-feats exp/tri5c_mpe/final.mat ark:- ark:- | transform-feats --utt2spk=ark:data/train/split8/1/utt2spk ark:exp/tri3b_ali/trans.1 ark:- ark:- |' 'ark,s,cs:gunzip -c exp/$dir_ali/ali.1.gz |' 'ark,s,cs:gunzip -c exp/$dir_denlats/lat.1.gz|' ark:2.egs

#    nnet-compare-hash-discriminative --acoustic-scale=$acoustic_scale --drop-frames=$drop_frames --criterion=$criterion exp/$dir/final.mdl ark:1.egs ark:2.egs || exit 1;
#  done
# done
# )
