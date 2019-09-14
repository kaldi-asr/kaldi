set(PUBLIC_HEADERS
    am-nnet.h
    combine-nnet-a.h
    combine-nnet-fast.h
    combine-nnet.h
    decodable-am-nnet.h
    get-feature-transform.h
    mixup-nnet.h
    nnet-component.h
    nnet-compute-discriminative-parallel.h
    nnet-compute-discriminative.h
    nnet-compute-online.h
    nnet-compute.h
    nnet-example-functions.h
    nnet-example.h
    nnet-fix.h
    nnet-functions.h
    nnet-limit-rank.h
    nnet-nnet.h
    nnet-precondition-online.h
    nnet-precondition.h
    nnet-stats.h
    nnet-update-parallel.h
    nnet-update.h
    online-nnet2-decodable.h
    rescale-nnet.h
    shrink-nnet.h
    train-nnet-ensemble.h
    train-nnet.h
    widen-nnet.h
)

add_library(kaldi-nnet2
    am-nnet.cc
    combine-nnet-a.cc
    combine-nnet-fast.cc
    combine-nnet.cc
    get-feature-transform.cc
    mixup-nnet.cc
    nnet-component.cc
    nnet-compute-discriminative-parallel.cc
    nnet-compute-discriminative.cc
    nnet-compute-online.cc
    nnet-compute.cc
    nnet-example-functions.cc
    nnet-example.cc
    nnet-fix.cc
    nnet-functions.cc
    nnet-limit-rank.cc
    nnet-nnet.cc
    nnet-precondition-online.cc
    nnet-precondition.cc
    nnet-stats.cc
    nnet-update-parallel.cc
    nnet-update.cc
    online-nnet2-decodable.cc
    rescale-nnet.cc
    shrink-nnet.cc
    train-nnet-ensemble.cc
    train-nnet.cc
    widen-nnet.cc
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
    add_kaldi_test_executable(NAME am-nnet-test SOURCES am-nnet-test.cc DEPENDS kaldi-nnet2)
    add_kaldi_test_executable(NAME nnet2-component-test SOURCES nnet-component-test.cc DEPENDS kaldi-nnet2)
    add_kaldi_test_executable(NAME nnet2-compute-test SOURCES nnet-compute-test.cc DEPENDS kaldi-nnet2)
    add_kaldi_test_executable(NAME nnet2-example-functions-test SOURCES nnet-example-functions-test.cc DEPENDS kaldi-nnet2)
    add_kaldi_test_executable(NAME nnet2-nnet-test SOURCES nnet-nnet-test.cc DEPENDS kaldi-nnet2)
    add_kaldi_test_executable(NAME nnet2-precondition-online-test SOURCES nnet-precondition-online-test.cc DEPENDS kaldi-nnet2)
    add_kaldi_test_executable(NAME nnet2-precondition-test SOURCES nnet-precondition-test.cc DEPENDS kaldi-nnet2)
    add_kaldi_test_executable(NAME online-nnet2-decodable-test SOURCES online-nnet2-decodable-test.cc DEPENDS kaldi-nnet2)
endif()

install(TARGETS kaldi-nnet2
    EXPORT kaldi-nnet2-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/nnet2)

