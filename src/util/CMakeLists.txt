add_library(kaldi-util
    simple-options.cc
    kaldi-semaphore.cc
    text-utils.cc
    simple-io-funcs.cc
    kaldi-thread.cc
    kaldi-io.cc
    parse-options.cc
    kaldi-holder.cc
    kaldi-table.cc
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
    add_kaldi_test_executable(NAME parse-options-test SOURCES parse-options-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME simple-options-test SOURCES simple-options-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME kaldi-io-test SOURCES kaldi-io-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME kaldi-thread-test SOURCES kaldi-thread-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME stl-utils-test SOURCES stl-utils-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME hash-list-test SOURCES hash-list-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME edit-distance-test SOURCES edit-distance-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME const-integer-set-test SOURCES const-integer-set-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME text-utils-test SOURCES text-utils-test.cc DEPENDS kaldi-util)
    add_kaldi_test_executable(NAME kaldi-table-test SOURCES kaldi-table-test.cc DEPENDS kaldi-util)
endif()
