add_library(kaldi-online2
    online-ivector-feature.cc
    online-endpoint.cc
    online-speex-wrapper.cc
    online-feature-pipeline.cc
    online-timing.cc
    online-nnet3-decoding.cc
    online-gmm-decodable.cc
    online-gmm-decoding.cc
    online-nnet2-decoding-threaded.cc
    online-nnet2-feature-pipeline.cc
    online-nnet2-decoding.cc
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

