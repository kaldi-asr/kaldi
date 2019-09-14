set(PUBLIC_HEADERS
    nnet-activation.h
    nnet-affine-transform.h
    nnet-average-pooling-component.h
    nnet-blstm-projected.h
    nnet-component.h
    nnet-convolutional-component.h
    nnet-frame-pooling-component.h
    nnet-kl-hmm.h
    nnet-linear-transform.h
    nnet-loss.h
    nnet-lstm-projected.h
    nnet-matrix-buffer.h
    nnet-max-pooling-component.h
    nnet-multibasis-component.h
    nnet-nnet.h
    nnet-parallel-component.h
    nnet-parametric-relu.h
    nnet-pdf-prior.h
    nnet-randomizer.h
    nnet-rbm.h
    nnet-recurrent.h
    nnet-sentence-averaging-component.h
    nnet-trnopts.h
    nnet-utils.h
    nnet-various.h
)

add_library(kaldi-nnet
    nnet-component.cc
    nnet-loss.cc
    nnet-nnet.cc
    nnet-pdf-prior.cc
    nnet-randomizer.cc
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
    add_kaldi_test_executable(NAME nnet-component-test SOURCES nnet-component-test.cc DEPENDS kaldi-nnet)
    add_kaldi_test_executable(NAME nnet-randomizer-test SOURCES nnet-randomizer-test.cc DEPENDS kaldi-nnet)
endif()

install(TARGETS kaldi-nnet
    EXPORT kaldi-nnet-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/nnet)

