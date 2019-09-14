set(PUBLIC_HEADERS
    basic-filebuf.h
    common-utils.h
    const-integer-set-inl.h
    const-integer-set.h
    edit-distance-inl.h
    edit-distance.h
    hash-list-inl.h
    hash-list.h
    kaldi-cygwin-io-inl.h
    kaldi-holder-inl.h
    kaldi-holder.h
    kaldi-io-inl.h
    kaldi-io.h
    kaldi-pipebuf.h
    kaldi-semaphore.h
    kaldi-table-inl.h
    kaldi-table.h
    kaldi-thread.h
    parse-options.h
    simple-io-funcs.h
    simple-options.h
    stl-utils.h
    table-types.h
    text-utils.h
)

add_library(kaldi-util
    kaldi-holder.cc
    kaldi-io.cc
    kaldi-semaphore.cc
    kaldi-table.cc
    kaldi-thread.cc
    parse-options.cc
    simple-io-funcs.cc
    simple-options.cc
    text-utils.cc
)

target_include_directories(kaldi-util PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-util PUBLIC
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME const-integer-set-test SOURCES const-integer-set-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME edit-distance-test SOURCES edit-distance-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME hash-list-test SOURCES hash-list-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME kaldi-io-test SOURCES kaldi-io-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME kaldi-table-test SOURCES kaldi-table-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME kaldi-thread-test SOURCES kaldi-thread-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME parse-options-test SOURCES parse-options-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME simple-options-test SOURCES simple-options-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME stl-utils-test SOURCES stl-utils-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME text-utils-test SOURCES text-utils-test.cc DEPENDS kaldi-util)
endif()

install(TARGETS kaldi-util
    EXPORT kaldi-util-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/util)

