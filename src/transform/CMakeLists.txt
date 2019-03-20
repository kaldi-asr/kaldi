set(source_files lda-estimate.cc
    cmvn.cc transform-common.cc fmllr-diag-gmm.cc
    lvtln.cc mllt.cc basis-fmllr-diag-gmm.cc
    compressed-transform-stats.cc)

set(test_files lda-estimate-test fmllr-diag-gmm-test)

set(dependencies base util matrix gmm tree)

create_library(transform "${source_files}" "${test_files}" "${dependencies}")
