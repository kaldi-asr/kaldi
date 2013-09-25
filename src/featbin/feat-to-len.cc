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
        "that maps utterance-id to utterance length in frames.\n"
        "Usage: feat-to-len [options] in-rspecifier out-wspecifier\n"
        "e.g.: feat-to-len scp:feats.scp ark,t:feats.lengths\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    Int32Writer length_writer(wspecifier);

    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
    for (; !kaldi_reader.Done(); kaldi_reader.Next())
      length_writer.Write(kaldi_reader.Key(), kaldi_reader.Value().NumRows());
    
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


