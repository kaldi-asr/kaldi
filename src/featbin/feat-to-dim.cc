// featbin/feat-to-dim.cc

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
        "Reads an archive of features.  If second argument is wxfilename, writes\n"
        "the feature dimension of the first feature file; if second argument is\n"
        "wspecifier, writes an archive of the feature dimension, indexed by utterance\n"
        "id.\n"
        "Usage: feat-to-dim [options] <feat-rspecifier> (<dim-wspecifier>|<dim-wxfilename>)\n"
        "e.g.: feat-to-dim scp:feats.scp -\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier_or_wxfilename = po.GetArg(2);

    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
      
    if (ClassifyWspecifier(wspecifier_or_wxfilename, NULL, NULL, NULL)
        != kNoWspecifier) {
      Int32Writer dim_writer(wspecifier_or_wxfilename);
      for (; !kaldi_reader.Done(); kaldi_reader.Next())
        dim_writer.Write(kaldi_reader.Key(), kaldi_reader.Value().NumCols());
    } else {
      if (kaldi_reader.Done())
        KALDI_ERR << "Could not read any features (empty archive?)";
      Output ko(wspecifier_or_wxfilename, false); // text mode.
      ko.Stream() << kaldi_reader.Value().NumCols() << "\n";
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


