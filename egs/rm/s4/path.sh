# Path to the root directory of Kaldi
root=../../../

# The root of the directory containing the features and metadata
RM1_ROOT=`pwd`/data

export PATH=${root}/src/bin:${root}/tools/openfst/bin:${root}/src/fstbin/:${root}/src/gmmbin/:${root}/src/featbin/:${root}/src/fgmmbin:${root}/src/sgmmbin:${root}/src/lm:${root}/src/latbin:$PATH

export LC_ALL=C
export LC_LOCALE_ALL=C

