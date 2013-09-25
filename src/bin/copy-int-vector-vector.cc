// bin/copy-int-vector-vector.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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



int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Copy vectors of vectors of integers, or archives thereof\n"
        "\n"
        "Usage: copy-int-vector-vector [options] vector-in-(rspecifier|rxfilename) "
        "vector-out-(wspecifierwxfilename)\n";
        
    bool binary = true;
    ParseOptions po(usage);

    po.Register("binary", &binary, "Write in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }    
    
    std::string in_fn = po.GetArg(1),
        out_fn = po.GetArg(2);

    bool archive_in = (ClassifyRspecifier(in_fn, NULL, NULL) != kNoRspecifier),
        archive_out = (ClassifyRspecifier(out_fn, NULL, NULL) != kNoRspecifier);
    
    if (archive_in != archive_out)
      KALDI_ERR << "Cannot mix Tables/archives and non-Trables.\n";
    
    if (archive_in) {
      int num_done = 0;
      Int32VectorVectorWriter writer(out_fn);
      SequentialInt32VectorVectorReader reader(in_fn);
      for (; !reader.Done(); reader.Next(), num_done++)
        writer.Write(reader.Key(), reader.Value());
      KALDI_LOG << "Copied " << num_done << " items.";
      return (num_done != 0 ? 0 : 1);
    } else {
      KALDI_ERR << "Non-archive reading and writing of vector<vector<int32> > "
          "not yet implemented.";
      // There doesn't seem to be a standard way of writing them, when
      // not appearing in tables.
      /*  std::vector<std::vector<int32> > vec;
          {
          bool binary_in;
          Input ki(in_fn, &binary_in);
          ReadIntegerVectorVector(ki.Stream(), binary_in, &vec);
          }
          Output ko(out_fn, binary);
          WriteIntegerVectorVector(ko.Stream(), binary, vec);
          KALDI_LOG << "Copied vector<vector<int32> > to " << vector_out_fn; */
      return 0;
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


