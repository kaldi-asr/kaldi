// bin/vector-scale.cc

// Copyright 2009-2011  Microsoft Corporation
//                2014  Johns Hopkins University (author: Daniel Povey)

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


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Scale a set of vectors in a Table (useful for speaker vectors and "
        "per-frame weights)\n"
        "Usage: vector-scale [options] <in-rspecifier> <out-wspecifier>\n";

    ParseOptions po(usage);
    BaseFloat scale = 1.0;

    po.Register("scale", &scale, "Scaling factor for vectors");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    BaseFloatVectorWriter vec_writer(wspecifier);

    SequentialBaseFloatVectorReader vec_reader(rspecifier);
    for (; !vec_reader.Done(); vec_reader.Next()) {
      Vector<BaseFloat> vec(vec_reader.Value());
      vec.Scale(scale);
      vec_writer.Write(vec_reader.Key(), vec);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


