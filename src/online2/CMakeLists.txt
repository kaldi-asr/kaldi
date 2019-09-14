set(PUBLIC_HEADERS
    online-endpoint.h
    online-feature-pipeline.h
    online-gmm-decodable.h
    online-gmm-decoding.h
    online-ivector-feature.h
    online-nnet2-decoding-threaded.h
    online-nnet2-decoding.h
    online-nnet2-feature-pipeline.h
    online-nnet3-decoding.h
    online-speex-wrapper.h
    online-timing.h
    onlinebin-util.h
)

add_library(kaldi-online2
    online-endpoint.cc
    online-feature-pipeline.cc
    online-gmm-decodable.cc
    online-gmm-decoding.cc
    online-ivector-feature.cc
    online-nnet2-decoding-threaded.cc
    online-nnet2-decoding.cc
    online-nnet2-feature-pipeline.cc
    online-nnet3-decoding.cc
    online-speex-wrapper.cc
    online-timing.cc
    onlinebin-util.cc
)

target_include_directories(kaldi-online2 PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-online2 PUBLIC
    kaldi-ivector
    kaldi-nnet3
    kaldi-chain
    kaldi-nnet2
    kaldi-cudamatrix
    kaldi-decoder
    kaldi-lat
    kaldi-hmm
    kaldi-feat
    kaldi-transform
    kaldi-gmm
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)


install(TARGETS kaldi-online2
    EXPORT kaldi-online2-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/online2)

