if(NOT OPENFST_ROOT_DIR)
    message(FATAL_ERROR)
endif()

set(fst_source_dir ${OPENFST_ROOT_DIR}/src/lib)
set(fst_include_dir ${OPENFST_ROOT_DIR}/src/include)

include_directories(${fst_include_dir})
file(GLOB fst_sources "${fst_source_dir}/*.cc")
file(GLOB fst_headers "${fst_include_dir}/fst/*.h")

add_library(fst ${fst_sources})
target_include_directories(fst PUBLIC
     $<BUILD_INTERFACE:${fst_include_dir}>
     $<INSTALL_INTERFACE:include>
)

install(TARGETS fst
    EXPORT kaldi-exports
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${fst_headers} DESTINATION include/fst)

unset(fst_source_dir)
unset(fst_include_dir)
unset(fst_sources)
unset(fst_headers)
