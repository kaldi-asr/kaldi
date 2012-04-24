// bin/copy-vector.cc

// Copyright 2009-2011  Microsoft Corporation

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
        "Copy vectors, or archives of vectors (e.g. transition-accs; speaker vectors)\n"
        "\n"
        "Usage: copy-vector [options] (vector-in-rspecifier|vector-in-rxfilename) (vector-out-wspecifier|vector-out-wxfilename)\n"
        " e.g.: copy-vector --binary=false 1.mat -\n"
        "   copy-vector ark:2.trans ark,t:-\n";
    
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
        out_is_rspecifier =
        (ClassifyRspecifier(vector_out_fn, NULL, NULL)
         != kNoRspecifier);

    if (in_is_rspecifier != out_is_rspecifier)
      KALDI_ERR << "Cannot mix archives with regular files (copying vectors)\n";
    
    if (!in_is_rspecifier) {
      Vector<BaseFloat> mat;
      ReadKaldiObject(vector_in_fn, &mat);
      Output ko(vector_out_fn, binary);
      mat.Write(ko.Stream(), binary);
      KALDI_LOG << "Copied vector to " << vector_out_fn;
      return 0;
    } else {
      int num_done = 0;
      BaseFloatVectorWriter writer(vector_out_fn);
      SequentialBaseFloatVectorReader reader(vector_in_fn);
      for (; !reader.Done(); reader.Next(), num_done++)
        writer.Write(reader.Key(), reader.Value());
      KALDI_LOG << "Copied " << num_done << " vectors.";
      return (num_done != 0 ? 0 : 1);
    }
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


