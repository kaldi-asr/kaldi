foreach(project base) # matrix util)
  message(STATUS "Looping over ${project}")
  add_subdirectory(${project})
endforeach(project)
