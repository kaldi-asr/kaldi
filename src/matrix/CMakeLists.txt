add_library(matrix compressed-matrix.cc kaldi-matrix.cc kaldi-vector.cc
  matrix-functions.cc 
  optimization.cc packed-matrix.cc qr.cc sp-matrix.cc
  sparse-matrix.cc srfft.cc tp-matrix.cc)
target_link_libraries(matrix PUBLIC base PRIVATE OpenBLAS)

foreach(test matrix-lib-speed-test matrix-lib-test sparse-matrix-test)
  add_executable(${test} "${test}.cc")
  target_link_libraries(${test} PRIVATE matrix)
  add_test(${test} ${test})
endforeach(test)

target_link_libraries(matrix-lib-test PRIVATE OpenBLAS)
