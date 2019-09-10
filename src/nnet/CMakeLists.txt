add_library(kaldi-nnet
    nnet-pdf-prior.cc
    nnet-nnet.cc
    nnet-randomizer.cc
    nnet-loss.cc
    nnet-component.cc
)

target_include_directories(kaldi-nnet PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-nnet PUBLIC
    kaldi-cudamatrix
    kaldi-hmm
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME nnet-randomizer-test SOURCES nnet-randomizer-test.cc DEPENDS kaldi-nnet)
    add_kaldi_test_executable(NAME nnet-component-test SOURCES nnet-component-test.cc DEPENDS kaldi-nnet)
endif()
