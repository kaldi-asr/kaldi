// featbin/copy-feats.cc

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
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Copy features [and possibly change format]\n"
        "Usage: copy-feats [options] in-rspecifier out-wspecifier\n";

    ParseOptions po(usage);
    bool htk_in = false;
    bool sphinx_in = false;
    po.Register("htk-in", &htk_in, "Read input as HTK features");
    po.Register("sphinx-in", &sphinx_in, "Read input as Sphinx features");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    if (htk_in) {
      SequentialTableReader<HtkMatrixHolder> htk_reader(rspecifier);
      for (; !htk_reader.Done(); htk_reader.Next())
        kaldi_writer.Write(htk_reader.Key(), htk_reader.Value().first);
    } else if (sphinx_in) {
      SequentialTableReader<SphinxMatrixHolder<> > sphinx_reader(rspecifier);
      for (; !sphinx_reader.Done(); sphinx_reader.Next())
        kaldi_writer.Write(sphinx_reader.Key(), sphinx_reader.Value());
    } else {
      SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
      for (; !kaldi_reader.Done(); kaldi_reader.Next())
        kaldi_writer.Write(kaldi_reader.Key(), kaldi_reader.Value());
    }
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


