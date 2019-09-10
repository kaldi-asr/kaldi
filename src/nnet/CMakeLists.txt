set(PUBLIC_HEADERS
    nnet-nnet.h
    nnet-convolutional-component.h
    nnet-loss.h
    nnet-affine-transform.h
    nnet-various.h
    nnet-frame-pooling-component.h
    nnet-matrix-buffer.h
    nnet-pdf-prior.h
    nnet-blstm-projected.h
    nnet-sentence-averaging-component.h
    nnet-max-pooling-component.h
    nnet-linear-transform.h
    nnet-recurrent.h
    nnet-average-pooling-component.h
    nnet-component.h
    nnet-parallel-component.h
    nnet-activation.h
    nnet-lstm-projected.h
    nnet-utils.h
    nnet-kl-hmm.h
    nnet-multibasis-component.h
    nnet-trnopts.h
    nnet-parametric-relu.h
    nnet-rbm.h
    nnet-randomizer.h
)

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

install(TARGETS kaldi-nnet
    EXPORT kaldi-nnet-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/nnet)

