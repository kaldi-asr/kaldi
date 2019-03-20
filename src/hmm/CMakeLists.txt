set(source_files topology.cc transitions.cc hmm-utils.cc tree-accu.cc posterior.cc
  hmm-test-utils.cc)
set(test_files hmm-utils-test transitions-test posterior-test)
set(dependencies base tree util matrix OpenFST)
create_library(hmm "${source_files}" "${test_files}" "${dependencies}")
