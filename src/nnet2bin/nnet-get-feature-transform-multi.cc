// nnet2bin/nnet-get-feature-transform-multi.cc

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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
#include "nnet2/get-feature-transform.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Get feature-projection transform using stats obtained with acc-lda.\n"
        "The file <index-list> contains a series of line, each containing a list\n"
        "of integer indexes.  For each line we create a transform of the same type\n"
        "as nnet-get-feature-transform would produce, taking as input just the\n"
        "listed feature dimensions.  The output transform will be the concatenation\n"
        "of all these transforms.  The output-dim will be the number of integers in\n"
        "the file <index-list> (the individual transforms are not dimension-reducing).\n"
        "Do not set the --dim option."
        "Usage:  nnet-get-feature-transform-multi [options] <index-list> <lda-acc-1> <lda-acc-2> ... <lda-acc-n> <matrix-out>\n";

    bool binary = true;

    FeatureTransformEstimateOptions opts;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write accumulators in binary mode.");
    opts.Register(&po);
    po.Read(argc, argv);
    
    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    FeatureTransformEstimateMulti fte;
    std::string index_list_rxfilename = po.GetArg(1);
    std::string projection_wxfilename = po.GetArg(po.NumArgs());

    std::vector<std::vector<int32> > indexes;
    {
      Input ki(index_list_rxfilename);
      std::string line;
      while (getline(ki.Stream(), line)) {
        std::vector<int32> this_indexes;
        if (!SplitStringToIntegers(line, " \t\n\r",
                                   true, &this_indexes) ||
            line.empty()) {
          KALDI_ERR << "Bad line in index-list file: line is " << line;
        }
        indexes.push_back(this_indexes);
      }
      if (indexes.empty())
        KALDI_ERR << "Empty index-list file "
                  << PrintableRxfilename(index_list_rxfilename);
    }
    
    for (int32 i = 2; i < po.NumArgs(); i++) {
      bool binary_in, add = true;
      Input ki(po.GetArg(i), &binary_in);
      fte.Read(ki.Stream(), binary_in, add);
    }

    Matrix<BaseFloat> mat;
    fte.Estimate(opts, indexes, &mat);
    WriteKaldiObject(mat, projection_wxfilename, binary);

    KALDI_LOG << "Wrote transform to "
              << PrintableWxfilename(projection_wxfilename);
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


