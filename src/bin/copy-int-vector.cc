// bin/copy-int-vector.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "matrix/kaldi-vector.h"
#include "transform/transform-common.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Copy vectors of integers, or archives of vectors of integers \n"
        "(e.g. alignments)\n"
        "\n"
        "Usage: copy-int-vector [options] (vector-in-rspecifier|vector-in-rxfilename) (vector-out-wspecifier|vector-out-wxfilename)\n"
        " e.g.: copy-int-vector --binary=false foo -\n"
        "   copy-int-vector ark:1.ali ark,t:-\n";
    
    bool binary = true;
    ParseOptions po(usage);

    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }


    std::string vector_in_fn = po.GetArg(1),
        vector_out_fn = po.GetArg(2);

    // all these "fn"'s are either rspecifiers or filenames.

    bool in_is_rspecifier =
        (ClassifyRspecifier(vector_in_fn, NULL, NULL)
         != kNoRspecifier),
        out_is_wspecifier =
        (ClassifyWspecifier(vector_out_fn, NULL, NULL, NULL)
         != kNoWspecifier);

    if (in_is_rspecifier != out_is_wspecifier)
      KALDI_ERR << "Cannot mix archives with regular files (copying vectors)";
    
    if (!in_is_rspecifier) {
      std::vector<int32> vec;
      {
        bool binary_in;
        Input ki(vector_in_fn, &binary_in);
        ReadIntegerVector(ki.Stream(), binary_in, &vec);
      }
      Output ko(vector_out_fn, binary);
      WriteIntegerVector(ko.Stream(), binary, vec);
      KALDI_LOG << "Copied vector to " << vector_out_fn;
      return 0;
    } else {
      int num_done = 0;
      Int32VectorWriter writer(vector_out_fn);
      SequentialInt32VectorReader reader(vector_in_fn);
      for (; !reader.Done(); reader.Next(), num_done++)
        writer.Write(reader.Key(), reader.Value());
      KALDI_LOG << "Copied " << num_done << " vectors of int32.";
      return (num_done != 0 ? 0 : 1);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


