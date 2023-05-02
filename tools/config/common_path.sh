# we assume KALDI_ROOT is already defined
[ -z "$KALDI_ROOT" ] && echo >&2 "The variable KALDI_ROOT must be already defined" && exit 1
# The formatting of the path export command is intentionally weird, because
# this allows for easy diff'ing
export PATH=\
${KALDI_ROOT}/src/bin:\
${KALDI_ROOT}/src/chainbin:\
${KALDI_ROOT}/src/featbin:\
${KALDI_ROOT}/src/fgmmbin:\
${KALDI_ROOT}/src/fstbin:\
${KALDI_ROOT}/src/gmmbin:\
${KALDI_ROOT}/src/ivectorbin:\
${KALDI_ROOT}/src/kwsbin:\
${KALDI_ROOT}/src/latbin:\
${KALDI_ROOT}/src/lmbin:\
${KALDI_ROOT}/src/nnet2bin:\
${KALDI_ROOT}/src/nnet3bin:\
${KALDI_ROOT}/src/nnetbin:\
${KALDI_ROOT}/src/online2bin:\
${KALDI_ROOT}/src/onlinebin:\
${KALDI_ROOT}/src/rnnlmbin:\
${KALDI_ROOT}/src/sgmm2bin:\
${KALDI_ROOT}/src/sgmmbin:\
${KALDI_ROOT}/src/tfrnnlmbin:\
${KALDI_ROOT}/src/cudadecoderbin:\
${KALDI_ROOT}/src/cudafeatbin:\
$PATH
