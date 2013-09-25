// featbin/add-deltas.cc

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
#include "feat/feature-functions.h"
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Add deltas (typically to raw mfcc or plp features\n"
        "Usage: add-deltas [options] in-rspecifier out-wspecifier\n";
    DeltaFeaturesOptions opts;
    int32 truncate = 0;
    ParseOptions po(usage);
    po.Register("truncate", &truncate, "If nonzero, first truncate features to this dimension.");
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    BaseFloatMatrixWriter feat_writer(wspecifier);
    SequentialBaseFloatMatrixReader feat_reader(rspecifier);
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats  = feat_reader.Value();

      if (feats.NumRows() == 0) {
        KALDI_WARN << "Empty feature matrix for key " << key;
        continue;
      }
      Matrix<BaseFloat> new_feats;
      if (truncate != 0) {
        if (truncate > feats.NumCols())
          KALDI_ERR << "Cannot truncate features as dimension " << feats.NumCols()
                    << " is smaller than truncation dimension.";
        SubMatrix<BaseFloat> feats_sub(feats, 0, feats.NumRows(), 0, truncate);
        ComputeDeltas(opts, feats_sub, &new_feats);
      } else
        ComputeDeltas(opts, feats, &new_feats);
      feat_writer.Write(key, new_feats);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


