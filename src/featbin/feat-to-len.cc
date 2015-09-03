// featbin/feat-to-len.cc

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
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Reads an archive of features and writes a corresponding archive\n"
        "that maps utterance-id to utterance length in frames, or (with\n"
        "one argument) print to stdout the total number of frames in the\n"
        "input archive.\n"
        "Usage: feat-to-len [options] <in-rspecifier> [<out-wspecifier>]\n"
        "e.g.: feat-to-len scp:feats.scp ark,t:feats.lengths\n"
        "or: feat-to-len scp:feats.scp\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 1 && po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    if (po.NumArgs() == 2) {
      std::string rspecifier = po.GetArg(1);
      std::string wspecifier = po.GetArg(2);

      Int32Writer length_writer(wspecifier);

      SequentialBaseFloatMatrixReader matrix_reader(rspecifier);
      for (; !matrix_reader.Done(); matrix_reader.Next())
        length_writer.Write(matrix_reader.Key(), matrix_reader.Value().NumRows());
    } else {
      int64 tot = 0;
      std::string rspecifier = po.GetArg(1);
      SequentialBaseFloatMatrixReader matrix_reader(rspecifier);
      for (; !matrix_reader.Done(); matrix_reader.Next())
        tot += matrix_reader.Value().NumRows();
      std::cout << tot << std::endl;
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


