add_library(kaldi-kws
    kws-scoring.cc
    kws-functions2.cc
    kws-functions.cc
)

target_include_directories(kaldi-kws PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-kws PUBLIC
    kaldi-lat
    kaldi-hmm
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)

