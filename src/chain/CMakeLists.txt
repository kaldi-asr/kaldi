set(PUBLIC_HEADERS
    chain-datastruct.h
    chain-den-graph.h
    chain-denominator.h
    chain-generic-numerator.h
    chain-kernels-ansi.h
    chain-numerator.h
    chain-supervision.h
    chain-training.h
    language-model.h
)

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
cuda_compile(CUDA_OBJS
    chain-kernels.cu
)

add_library(kaldi-chain
    chain-den-graph.cc
    chain-denominator.cc
    chain-generic-numerator.cc
    chain-numerator.cc
    chain-supervision.cc
    chain-training.cc
    language-model.cc
    ${CUDA_OBJS}
)

target_include_directories(kaldi-chain PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-chain PUBLIC
    kaldi-cudamatrix
    kaldi-lat
    kaldi-fstext
    kaldi-hmm
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME chain-supervision-test SOURCES chain-supervision-test.cc DEPENDS kaldi-chain)
    add_kaldi_test_executable(NAME language-model-test SOURCES language-model-test.cc DEPENDS kaldi-chain)
endif()

install(TARGETS kaldi-chain
    EXPORT kaldi-chain-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/chain)

