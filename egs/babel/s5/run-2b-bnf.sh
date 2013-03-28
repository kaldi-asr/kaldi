#!/bin/bash

# This is the "final" version of the script that runs trains the bottleneck system.
# It is to be run after run.sh (the new version, that uses the same number of phases
# for both the limitedLP and fullLP systems), and before run-BNF-system.sh.  This
# script is a bit "special" as it requires a GPU, and also the "ptdnn" tools by
# Yajie Miao to train the DNN.  Eventually the DNN training will be made a part
# of Kaldi, but this recipe uses these external tools.

. conf/common_vars.sh
. ./lang.conf

# At this point you may want to make sure the directory exp_BNF/bnf_dnn is
# somewhere with a lot of space, preferably on the local GPU-containing machine.

# This script is not fully automated.
# The "exit 1" below is to stop you from trying to just run it as a script.
# you should run it line by line (not forgetting to source the things at the
# top first).  
# At some point you have to edit ptdnn/exp_bnf/config.py
exit 1;

if [ ! -d ptdnn ]; then
  echo "Checking out PTDNN code.  You will have to edit the config file!"
  svn co svn://svn.code.sf.net/p/ptdnn/code-0/trunk/ptdnn ptdnn
fi

echo "Not finished!  Note, you need a GPU for this."
exit 0;

echo ---------------------------------------------------------------------
echo "Starting exp_BNF/bnf_dnn on" `date`
echo ---------------------------------------------------------------------
# Note that align_fmllr.sh may have been implemented in run-limited.sh
#steps/align_fmllr.sh --boost-silence 1.5 --nj $train_nj --cmd "$train_cmd" \
#    data/train data/lang exp/tri5 exp/tri5_ali || exit 1
# made exp_BNF a link to local storage before running this.  It produces a lot of temp files.
steps_BNF/build_nnet_pfile.sh --cmd "run.pl" --every-nth-frame 2 \
    data/train data/lang exp/tri5_ali exp_BNF/bnf_dnn || exit 1

# Now you can copy train.pfile.gz and valid.pfile.gz to the GPU machine and run
# ptdnn.   
# Dan: I was doing this on a GPU machine. I copied the ptdnn directory to ~/ptdnn
#
#Work out #pdfs.
#gmm-info exp/tri5_ali/final.mdl | grep pdfs

# edit ptdnn/exp_bnf/config.py to have: self.wdir = "/home/dpovey/kaldi-trunk/egs/babel/s5-tagalog-limited/exp_BNF/bnf_dnn/"
# and n_outs = #pdfs, which was 4711 in this case.
# note: instead of just saying "python" you might in general have to use a specific python version
# that you've set up to work with Theano, and maybe do more messing with the python path.  I (Dan) installed
# theano as administrator of the machine I was running on.  This is kind of machine specific.

# the following variables were needed on the JHUhines, and possibly also a specially
# installed version of python, but not in the cloud where I installed theano globally.
#export LD_LIBRARY_PATH=/opt/nvidia_cuda/cuda-5.0/lib64
#export PATH=$PATH:/opt/nvidia_cuda/cuda-5.0/lib64/libcublas
export THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32

export PYTHONPATH=$PYTHONPATH:`pwd`/ptdnn/
python ptdnn/main.py


echo ---------------------------------------------------------------------
echo "Now run run-BNF-system.sh"
echo ---------------------------------------------------------------------
