set(source_files diag-gmm.cc diag-gmm-normal.cc mle-diag-gmm.cc am-diag-gmm.cc
  mle-am-diag-gmm.cc full-gmm.cc full-gmm-normal.cc mle-full-gmm.cc
  model-common.cc decodable-am-diag-gmm.cc model-test-common.cc
  ebw-diag-gmm.cc indirect-diff-diag-gmm.cc)

set(test_files diag-gmm-test mle-diag-gmm-test full-gmm-test mle-full-gmm-test
  am-diag-gmm-test mle-am-diag-gmm-test ebw-diag-gmm-test)

set(dependencies base util matrix tree)

create_library(gmm "${source_files}" "${test_files}" "${dependencies}")
