#!/bin/bash

. conf/common_vars.sh
. ./lang.conf
sopt="--min-lmwt 25 --max-lmwt 35"

# At this point you may want to make sure the directory exp_BNF/bnf_dnn is
# somewhere with a lot of space, preferably on the local GPU-containing machine.

echo ---------------------------------------------------------------------
echo "Starting exp_BNF/bnf_dnn on" `date`
echo ---------------------------------------------------------------------
# Note that align_fmllr.sh may have been implemented in run-limited.sh
#steps/align_fmllr.sh --boost-silence 1.5 --nj $train_nj --cmd "$train_cmd" \
#    data/train data/lang exp/tri4 exp/tri4_ali || exit 1
steps_BNF/build_nnet_pfile.sh --cmd "$train_cmd" \
    data/train data/lang exp/tri4_ali exp_BNF/bnf_dnn || exit 1

# Now you can copy train.pfile.gz and valid.pfile.gz to the GPU machine and run
# ptdnn.   
# Dan: I was doing this on a GPU machine. I copied the ptdnn directory to ~/ptdnn
#
#Work out #pdfs.
#gmm-info exp/tri4_ali/final.mdl | grep pdfs

# edit ptdnn/exp_bnf/config.py to have: self.wdir = "/home/dpovey/kaldi-trunk/egs/babel/s5-tagalog-limited/exp_BNF/bnf_dnn/"
# and n_outs = #pdfs, which was 1944 in this case.
# note: instead of just saying "python" you might in general have to use a specific python version
# that you've set up to work with Theano, and maybe do more messing with the python path.  I (Dan) installed
# theano as administrator of the machine I was running on.  This is kind of machine specific.

# the following variables were needed on the JHUhines, and possibly also a specially
# installed version of python, but not in the cloud where I installed theano globally.
#export LD_LIBRARY_PATH=/opt/nvidia_cuda/cuda-5.0/lib64
#export PATH=$PATH:/opt/nvidia_cuda/cuda-5.0/lib64/libcublas
export THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32

export PYTHONPATH=$PYTHONPATH:`pwd`/ptdnn/
python ptdnn/main.py


echo ---------------------------------------------------------------------
echo "Now run run-BNF-system.sh"
echo ---------------------------------------------------------------------

exit 0
