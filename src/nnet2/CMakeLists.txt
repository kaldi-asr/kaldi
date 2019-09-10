set(PUBLIC_HEADERS
    nnet-nnet.h
    nnet-precondition.h
    online-nnet2-decodable.h
    nnet-compute.h
    nnet-update.h
    nnet-compute-discriminative-parallel.h
    nnet-compute-discriminative.h
    combine-nnet.h
    combine-nnet-a.h
    rescale-nnet.h
    nnet-compute-online.h
    nnet-precondition-online.h
    train-nnet-ensemble.h
    nnet-component.h
    nnet-stats.h
    train-nnet.h
    decodable-am-nnet.h
    nnet-functions.h
    combine-nnet-fast.h
    nnet-example.h
    get-feature-transform.h
    shrink-nnet.h
    nnet-limit-rank.h
    nnet-fix.h
    mixup-nnet.h
    am-nnet.h
    nnet-example-functions.h
    nnet-update-parallel.h
    widen-nnet.h
)

add_library(kaldi-nnet2
    nnet-compute-discriminative.cc
    train-nnet.cc
    online-nnet2-decodable.cc
    am-nnet.cc
    nnet-example.cc
    nnet-update-parallel.cc
    nnet-precondition-online.cc
    nnet-nnet.cc
    mixup-nnet.cc
    combine-nnet-a.cc
    nnet-compute.cc
    rescale-nnet.cc
    train-nnet-ensemble.cc
    nnet-precondition.cc
    get-feature-transform.cc
    combine-nnet.cc
    nnet-stats.cc
    nnet-compute-discriminative-parallel.cc
    nnet-example-functions.cc
    shrink-nnet.cc
    nnet-functions.cc
    widen-nnet.cc
    nnet-compute-online.cc
    combine-nnet-fast.cc
    nnet-component.cc
    nnet-limit-rank.cc
    nnet-fix.cc
    nnet-update.cc
)

target_include_directories(kaldi-nnet2 PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-nnet2 PUBLIC
    kaldi-cudamatrix
    kaldi-lat
    kaldi-hmm
    kaldi-transform
    kaldi-gmm
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME nnet2-example-functions-test SOURCES nnet-example-functions-test.cc DEPENDS kaldi-nnet2)
    add_kaldi_test_executable(NAME nnet2-compute-test SOURCES nnet-compute-test.cc DEPENDS kaldi-nnet2)
    add_kaldi_test_executable(NAME am-nnet-test SOURCES am-nnet-test.cc DEPENDS kaldi-nnet2)
    add_kaldi_test_executable(NAME nnet2-precondition-test SOURCES nnet-precondition-test.cc DEPENDS kaldi-nnet2)
    add_kaldi_test_executable(NAME online-nnet2-decodable-test SOURCES online-nnet2-decodable-test.cc DEPENDS kaldi-nnet2)
    add_kaldi_test_executable(NAME nnet2-nnet-test SOURCES nnet-nnet-test.cc DEPENDS kaldi-nnet2)
    add_kaldi_test_executable(NAME nnet2-component-test SOURCES nnet-component-test.cc DEPENDS kaldi-nnet2)
    add_kaldi_test_executable(NAME nnet2-precondition-online-test SOURCES nnet-precondition-online-test.cc DEPENDS kaldi-nnet2)
endif()

install(TARGETS kaldi-nnet2
    EXPORT kaldi-nnet2-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/nnet2)

