add_library(kaldi-tfrnnlm
    tensorflow-rnnlm.cc
)

target_include_directories(kaldi-tfrnnlm PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-tfrnnlm PUBLIC
    kaldi-lm
    kaldi-util
    kaldi-matrix
    kaldi-base
    LDLIBS
    +
    -lz
    -ldl
    -fPIC
    -lrt
    LDLIBS
    +
    tensorflow
    -ltensorflow_cc
    -ltensorflow_framework
)

