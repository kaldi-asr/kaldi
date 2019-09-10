set(PUBLIC_HEADERS
    tensorflow-rnnlm.h
)

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


install(TARGETS kaldi-tfrnnlm
    EXPORT kaldi-tfrnnlm-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/tfrnnlm)

