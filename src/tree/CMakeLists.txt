set(source_files build-tree.cc build-tree-utils.cc cluster-utils.cc
  tree-renderer.cc build-tree-questions.cc event-map.cc
  clusterable-classes.cc context-dep.cc)
set(test_files context-dep-test build-tree-utils-test build-tree-test
  event-map-test)
set(dependencies base util matrix)
create_library(tree "${source_files}" "${test_files}" "${dependencies}")
