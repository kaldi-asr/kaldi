add_kaldi_executable(NAME lattice-lmrescore-tf-rnnlm SOURCES lattice-lmrescore-tf-rnnlm.cc DEPENDS kaldi-tfrnnlm kaldi-lat)
add_kaldi_executable(NAME lattice-lmrescore-tf-rnnlm-pruned SOURCES lattice-lmrescore-tf-rnnlm-pruned.cc DEPENDS kaldi-tfrnnlm kaldi-lat)
