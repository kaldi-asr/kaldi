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
        "Scale vectors, or archives of vectors (useful for speaker vectors and "
        "per-frame weights)\n"
        "Usage: vector-scale [options] <vector-in-rspecifier> <vector-out-wspecifier>\n"
        "   or: vector-scale [options] <vector-in-rxfilename> <vector-out-wxfilename>\n"
        " e.g.: vector-scale --scale=-1.0 1.vec -\n"
        "       vector-scale --scale=-2.0 ark:vec.ark ark,t:-\n"
        "See also: copy-vector, vector-sum\n";

    ParseOptions po(usage);
    BaseFloat scale = 1.0;
    bool binary = false;

    po.Register("binary", &binary, "If true, write output as binary "
                "not relevant for archives");
    po.Register("scale", &scale, "Scaling factor for vectors");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string vector_in_fn = po.GetArg(1);
    std::string vector_out_fn = po.GetArg(2);

    if (ClassifyWspecifier(vector_in_fn, NULL, NULL, NULL) != kNoWspecifier) {
      if (ClassifyRspecifier(vector_in_fn, NULL, NULL) == kNoRspecifier) {
        KALDI_ERR << "Cannot mix archives and regular files";
      }
      BaseFloatVectorWriter vec_writer(vector_out_fn);
      SequentialBaseFloatVectorReader vec_reader(vector_in_fn);
      for (; !vec_reader.Done(); vec_reader.Next()) {
        Vector<BaseFloat> vec(vec_reader.Value());
        vec.Scale(scale);
        vec_writer.Write(vec_reader.Key(), vec);
      }
    } else {
      if (ClassifyRspecifier(vector_in_fn, NULL, NULL) != kNoRspecifier) {
        KALDI_ERR << "Cannot mix archives and regular files";
      }
      bool binary_in;
      Input ki(vector_in_fn, &binary_in);
      Vector<BaseFloat> vec;
      vec.Read(ki.Stream(), binary_in);
      vec.Scale(scale);
      Output ko(vector_out_fn, binary);
      vec.Write(ko.Stream(), binary);
    }

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


