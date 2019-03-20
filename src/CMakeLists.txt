function(create_library library_name source_files test_files dependencies)
  add_library(${library_name} ${source_files})
  target_link_libraries(${library_name} PUBLIC ${dependencies})

  foreach(test ${test_files})
    add_executable(${test} "${test}.cc")
    target_link_libraries(${test} PRIVATE ${library_name})
    add_test(NAME ${library_name}-${test} COMMAND ${test})
  endforeach(test)
endfunction(create_library)

function(create_binaries library_name source_files test_files dependencies)
  add_library(${library_name} ${source_files})
endfunction(create_binaries)


function(create_cuda_library library_name source_files)
  
endfunction(create_cuda_library)


foreach(project base matrix util feat tree gmm transform fstext hmm lm
    decoder lat ) #kws)
  add_subdirectory(${project})
endforeach(project)

# How to make a shared object which includes all of the .a's contents?
# add_library(kaldi SHARED)
# target_link_libraries(kaldi base)
