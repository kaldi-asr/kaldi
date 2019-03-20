set(source_files feature-functions.cc feature-mfcc.cc feature-fbank.cc
  mel-computations.cc wave-reader.cc
  pitch-functions.cc resample.cc online-feature.cc signal.cc
  feature-window.cc)

set(test_files feature-mfcc-test feature-fbank-test
  feature-functions-test pitch-functions-test feature-sdc-test
  resample-test online-feature-test signal-test wave-reader-test)

set(dependencies base matrix util gmm transform tree)

create_library(feat "${source_files}" "${test_files}" "${dependencies}")
