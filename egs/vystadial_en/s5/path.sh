# Needed for "correct" sorting
export LC_ALL=C
export KALDI_ROOT=../../..

# adding Kaldi binaries to path
export PATH=$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$PWD:$PWD/utils:$PWD/steps:$PATH


srilm_bin=$KALDI_ROOT/tools/srilm/bin/
if [ ! -e "$srilm_bin" ] ; then
    echo "SRILM is not installed in $KALDI_ROOT/tools."
    echo "May not be able to create LMs!"
    echo "Please go to $KALDI_ROOT/tools and run ./install_srilm.sh"
fi
srilm_sub_bin=`find "$srilm_bin" -type d`
for d in $srilm_sub_bin ; do
    export PATH=$d:$PATH
done
