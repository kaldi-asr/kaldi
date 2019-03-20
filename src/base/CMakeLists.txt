add_library(base kaldi-math.cc kaldi-error.cc io-funcs.cc kaldi-utils.cc timer.cc)
# Used only for version.h
target_include_directories(base PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/..")
target_link_libraries(base PUBLIC OpenFST)
add_dependencies(base generate_version_h)

add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/version.h"
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/get_version.sh"
  COMMAND ${CMAKE_COMMAND} -E rename "${CMAKE_CURRENT_SOURCE_DIR}/version.h" "${CMAKE_CURRENT_BINARY_DIR}/version.h")
add_custom_target(generate_version_h ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/version.h")


foreach(test kaldi-math-test io-funcs-test kaldi-error-test timer-test)
  add_executable(${test} "${test}.cc")
  target_link_libraries(${test} PRIVATE base)
  add_test(${test} ${test})
endforeach(test)
