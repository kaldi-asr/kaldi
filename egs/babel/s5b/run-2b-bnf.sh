#!/bin/bash

# This is the "final" version of the script that runs trains the bottleneck system.
# It is to be run after run.sh (the new version, that uses the same number of phases
# for both the limitedLP and fullLP systems), and before run-BNF-system.sh.  This
# script is a bit "special" as it requires a GPU, and also the "ptdnn" tools by
# Yajie Miao to train the DNN.  Eventually the DNN training will be made a part
# of Kaldi, but this recipe uses these external tools.

# you will probably want to make exp_BNF/bnf_dnn_run a link to somewhere local
# while you are running this script.  It doesn't have to be accessible globally,
# the script copies the things you will need.

working_dir=exp_BNF/bnf_dnn_run
cmd=run.pl
. utils/parse_options.sh

. conf/common_vars.sh
. ./lang.conf
. ./cmd.sh

# At this point you may want to make sure the directory exp_BNF/bnf_dnn is
# somewhere with a lot of space, preferably on the local GPU-containing machine.

if [ ! -d ptdnn ]; then
  echo "Checking out PTDNN code.  You will have to edit the config file!"
  svn co svn://svn.code.sf.net/p/ptdnn/code-0/trunk/ptdnn ptdnn
fi

if ! which nvcc; then
  echo "The command nvcc could not be found on the path: please make sure it is on"
  echo "your path (and that you have NVidia tools installed in the first place)"
fi

if ! nvidia-smi; then
  echo "The command nvidia-smi was not found: this probably means you don't have a GPU.  Not continuing"
  echo "(Note: this script might still work, it would just be slower.)"
  exit 1;
fi

if ! python -c 'import theano;'; then
  echo "Theano does not seem to be installed on your machine.  Not continuing."
  echo "(Note: this script might still work, it would just be slower.)"
  exit 1;
fi

mkdir -p exp_BNF/bnf_dnn
mkdir -p $working_dir

! gmm-info exp/tri5_ali/final.mdl >&/dev/null && \
   echo "Error getting GMM info from exp/tri5_ali/final.mdl" && exit 1;

num_pdfs=`gmm-info exp/tri5_ali/final.mdl | grep pdfs | awk '{print $NF}'` || exit 1;

[ -z "$babel_type" ] && echo "Variable babel_type not set " && exit 1;

# Now we copy conf/bnf/config_limited.py or conf/bnf/config_full.py, as appropriate,
# to ptdnn/exp_bnf/config.py, replacing a couple of things as we copy it.
WORK=`readlink -f $working_dir`
config_in=conf/bnf/config_${babel_type}.py
[ ! -f $config_in ] && echo "No such config file $config_in" && exit 1;
! cat $config_in | sed "s|CWD|$PWD|" | sed "s|WORK|$WORK|" | sed "s/N_OUTS/${num_pdfs}/" > ptdnn/exp_bnf/config.py && \
  echo "Error setting ptdnn/exp_bnf/config.py" && exit 1;
  

echo ---------------------------------------------------------------------
echo "Starting exp_BNF/bnf_dnn on" `date`
echo ---------------------------------------------------------------------
# Note: align_fmllr.sh will have been run in run-1-main.sh
# make exp_BNF a link to local storage before running this.  It produces a lot of temp files.

if [ ! -s exp_BNF/bnf_dnn_run/concat.pfile ]; then
  steps_BNF/build_nnet_pfile.sh --nj 1 --every-nth-frame "$bnf_every_nth_frame" \
    data/train data/lang exp/tri5_ali $working_dir || exit 1
fi

#export LD_LIBRARY_PATH=/opt/nvidia_cuda/cuda-5.0/lib64
#export PATH=$PATH:/opt/nvidia_cuda/cuda-5.0/lib64/libcublas

$cmd $working_dir/theano.log \
  export PYTHONPATH=$PYTHONPATH:`pwd`/ptdnn/ \; \
  export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 \; \
  python ptdnn/main.py 

mkdir -p exp_BNF/bnf_dnn
cp -r $working_dir/{final.mat,final.nnet,log,LOG,theano.log} exp_BNF/bnf_dnn/
mv $working_dir exp_BNF/bnf_dnn_run


echo ---------------------------------------------------------------------
echo "Now run run-3-bnf-system.sh"
echo ---------------------------------------------------------------------
