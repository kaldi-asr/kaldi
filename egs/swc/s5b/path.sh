# My binaries
export KALDI_ROOT=`pwd`/../../..

# Morrie's compilation
# export KALDI_ROOT=/share/spandh.ami1/sw/spl/kaldi/git_2015_12_07_b09e2f2/
# export LD_LIBRARY_PATH=/share/spandh.ami1/sw/std/mkl/v2016_update1/x86_x64/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64_lin:${LD_LIBRARY_PATH}

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh 
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/nnet3bin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/lmbin/:$PWD:$PATH
export LC_ALL=C

LMBIN=$KALDI_ROOT/tools/irstlm/bin
SRILM=$KALDI_ROOT/tools/srilm/bin/i686-m64
BEAMFORMIT=$KALDI_ROOT/tools/BeamformIt-3.51

export PATH=$PATH:$LMBIN:$BEAMFORMIT:$SRILM

export PERL5LIB=/share/spandh.ami1/sw/lib/std/libperl4-corelibs-perl/Perl4-CoreLibs-0.003/lib:${PERL5LIB}
export LD_LIBRARY_PATH=/usr/lib64/atlas/:${LD_LIBRARY_PATH}

# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/share/spandh.ami1/usr/yulan/sw/kaldi/dev/kaldi-trunk/egs/swc/s5/lib/

# Openfst
export PATH=$PATH:/share/spandh.ami1/sw/spl/openfst/v1.4.1/x86_64/bin

# For scoring
export PERL5LIB=/share/spandh.ami1/sw/lib/std/libperl4-corelibs-perl/Perl4-CoreLibs-0.003/lib:${PERL5LIB}

