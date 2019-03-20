set(source_files arpa-file-parser.cc arpa-lm-compiler.cc const-arpa-lm.cc
  kaldi-rnnlm.cc mikolov-rnnlm-lib.cc)
set(test_files arpa-file-parser-test arpa-lm-compiler-test)
set(dependencies base util matrix fstext)
create_library(lm "${source_files}" "${test_files}" "${dependencies}")
