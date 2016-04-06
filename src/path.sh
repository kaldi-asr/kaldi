# we are using BASH_SOURCE[0], because its set correctly even when the file
# is sourced.
# The formatting of the path export command is intentionally weird, because
# this allows for easy diff'ing
this_script_path=$(readlink -f "${BASH_SOURCE[0]}") 
my_kaldi_src=$(dirname $this_script_path)
export PATH=\
$my_kaldi_src/bin:\
$my_kaldi_src/chainbin:\
$my_kaldi_src/featbin:\
$my_kaldi_src/fgmmbin:\
$my_kaldi_src/fstbin:\
$my_kaldi_src/gmmbin:\
$my_kaldi_src/ivectorbin:\
$my_kaldi_src/kwsbin:\
$my_kaldi_src/latbin:\
$my_kaldi_src/lmbin:\
$my_kaldi_src/nnet2bin:\
$my_kaldi_src/nnet3bin:\
$my_kaldi_src/nnetbin:\
$my_kaldi_src/online2bin:\
$my_kaldi_src/onlinebin:\
$my_kaldi_src/sgmm2bin:\
$my_kaldi_src/sgmmbin:\
$PATH
