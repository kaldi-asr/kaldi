add_library(matrix compressed-matrix.cc kaldi-matrix.cc kaldi-vector.cc
  matrix-functions.cc 
  optimization.cc packed-matrix.cc qr.cc sp-matrix.cc
  sparse-matrix.cc srfft.cc tp-matrix.cc)
target_link_libraries(matrix PUBLIC base PRIVATE ${OpenBLAS_LIB})
target_include_directories(matrix PRIVATE ${OpenBLAS_INCLUDE_DIR})
target_compile_definitions(matrix PRIVATE HAVE_OPENBLAS=1)

foreach(test matrix-lib-speed-test matrix-lib-test sparse-matrix-test)
  add_executable(${test} "${test}.cc")
  target_link_libraries(${test} PRIVATE matrix)
  add_test(${test} ${test})
endforeach(test)

target_compile_definitions(matrix-lib-test PRIVATE HAVE_OPENBLAS=1)
target_include_directories(matrix-lib-test PRIVATE ${OpenBLAS_INCLUDE_DIR})
