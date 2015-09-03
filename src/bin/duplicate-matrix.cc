// bin/duplicate-matrix.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "transform/transform-common.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Copy tables of BaseFloat matrices, from one input to possibly multiple outputs,\n"
        "with each element of the input written too all outputs.\n"
        "\n"
        "Usage: duplicate-matrix [options] <matrix-rspecifier> <matrix-wspecifier1> [<matrix-wspecifier2> ...]\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }


    std::string matrix_rspecifier = po.GetArg(1);

    SequentialBaseFloatMatrixReader matrix_reader(matrix_rspecifier);
    
    std::vector<BaseFloatMatrixWriter> writers(po.NumArgs() - 1);
    for (size_t i = 0; i < writers.size(); i++)
      if (!writers[i].Open(po.GetArg(i + 1)))
        KALDI_ERR << "Error opening table for writing with wspecifier \""
                  <<  po.GetArg(i + 1) << '"';

    int32 num_done = 0;
    for (; !matrix_reader.Done(); matrix_reader.Next(), num_done++)
      for (size_t i = 0; i < writers.size(); i++)
        writers[i].Write(matrix_reader.Key(), matrix_reader.Value());

    KALDI_LOG << "Copied " << num_done << " matrices to "
              << writers.size() << " outputs.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


