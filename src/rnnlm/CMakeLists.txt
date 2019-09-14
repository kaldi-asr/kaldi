set(PUBLIC_HEADERS
    rnnlm-compute-state.h
    rnnlm-core-compute.h
    rnnlm-core-training.h
    rnnlm-embedding-training.h
    rnnlm-example-utils.h
    rnnlm-example.h
    rnnlm-lattice-rescoring.h
    rnnlm-test-utils.h
    rnnlm-training.h
    rnnlm-utils.h
    sampler.h
    sampling-lm-estimate.h
    sampling-lm.h
)

add_library(kaldi-rnnlm
    rnnlm-compute-state.cc
    rnnlm-core-compute.cc
    rnnlm-core-training.cc
    rnnlm-embedding-training.cc
    rnnlm-example-utils.cc
    rnnlm-example.cc
    rnnlm-lattice-rescoring.cc
    rnnlm-test-utils.cc
    rnnlm-training.cc
    rnnlm-utils.cc
    sampler.cc
    sampling-lm-estimate.cc
    sampling-lm.cc
)

target_include_directories(kaldi-rnnlm PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-rnnlm PUBLIC
    kaldi-nnet3
    kaldi-cudamatrix
    kaldi-lm
    kaldi-hmm
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME rnnlm-example-test SOURCES rnnlm-example-test.cc DEPENDS kaldi-rnnlm)
    add_kaldi_test_executable(NAME sampler-test SOURCES sampler-test.cc DEPENDS kaldi-rnnlm)
    add_kaldi_test_executable(NAME sampling-lm-test SOURCES sampling-lm-test.cc DEPENDS kaldi-rnnlm)
endif()

install(TARGETS kaldi-rnnlm
    EXPORT kaldi-rnnlm-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/rnnlm)

