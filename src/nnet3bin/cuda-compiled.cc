// nnet2bin/cuda-compiled.cc

// Copyright 2014 Johns Hopkins University (author:  Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "cudamatrix/cu-device.h"

int main(int argc, char *argv[]) {
  const char *usage = "This program returns exit status 0 (success) if the code\n"
      "was compiled with CUDA support, and 1 otherwise.  To support CUDA, you\n"
      "must run 'configure' on a machine that has the CUDA compiler 'nvcc'\n"
      "available.\n";
  if (argc > 1) {
    std::cerr << usage << "\n";
  }
#if HAVE_CUDA==1
  return 0;
#else
  return 1;
#endif
}
