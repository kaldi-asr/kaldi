add_library(util kaldi-holder.cc kaldi-table.cc kaldi-thread-test.cc
  kaldi-thread.cc kaldi-io.cc kaldi-semaphore.cc parse-options.cc
  simple-io-funcs.cc simple-options.cc text-utils.cc)
target_link_libraries(util PUBLIC base PUBLIC matrix PUBLIC Threads::Threads)
foreach(test const-integer-set-test 	edit-distance-test hash-list-test kaldi-io-test parse-options-test simple-options-test stl-utils-test text-utils-test kaldi-table-test)
  add_executable(${test} "${test}.cc")
  target_link_libraries(${test} PRIVATE util)
  add_test(${test} ${test})
endforeach(test)
