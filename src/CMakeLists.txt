foreach(project base matrix util)
  message(STATUS "Looping over ${project}")
  add_subdirectory(${project})
endforeach(project)

# How to make a shared object which includes all of the .a's contents?
# add_library(kaldi SHARED)
# target_link_libraries(kaldi base)
