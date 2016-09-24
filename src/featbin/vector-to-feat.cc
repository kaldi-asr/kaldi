// featbin/vector-to-feat.cc

// Copyright 2015   Vimal Manohar

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
        "Convert a vector into a single feature so that it can be appended \n"
        "to other feature matrices\n"
        "Usage: vector-to-feats <vector-rspecifier> <feature-wspecifier>\n"
        "or:   vector-to-feats <vector-rxfilename> <feature-wxfilename>\n"
        "e.g.: vector-to-feats scp:weights.scp ark:weight_feats.ark\n"
        " or: vector-to-feats weight_vec feat_mat\n"
        "See also: copy-feats, copy-matrix, paste-feats, \n"
        "subsample-feats, splice-feats\n";

    ParseOptions po(usage);
    bool compress = false, binary = true;

    po.Register("binary", &binary, "Binary-mode output (not relevant if writing "
                "to archive)");
    po.Register("compress", &compress, "If true, write output in compressed form"
                "(only currently supported for wxfilename, i.e. archive/script,"
                "output)");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_done = 0;

    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL) != kNoRspecifier) {
      std::string vector_rspecifier = po.GetArg(1);
      std::string feature_wspecifier = po.GetArg(2);

      SequentialBaseFloatVectorReader vector_reader(vector_rspecifier);
      BaseFloatMatrixWriter feat_writer(feature_wspecifier);
      CompressedMatrixWriter compressed_feat_writer(feature_wspecifier);

      for (; !vector_reader.Done(); vector_reader.Next(), ++num_done) {
        const Vector<BaseFloat> &vec = vector_reader.Value();
        Matrix<BaseFloat> feat(vec.Dim(), 1);
        feat.CopyColFromVec(vec, 0);

        if (!compress)
          feat_writer.Write(vector_reader.Key(), feat);
        else
          compressed_feat_writer.Write(vector_reader.Key(),
                                       CompressedMatrix(feat));
      }
      KALDI_LOG  << "Converted " << num_done << " vectors into features";
      return (num_done != 0 ? 0 : 1);
    }

    KALDI_ASSERT(!compress && "Compression not yet supported for single files");

    std::string vector_rxfilename = po.GetArg(1),
                feature_wxfilename = po.GetArg(2);

    Vector<BaseFloat> vec;
    ReadKaldiObject(vector_rxfilename, &vec);

    Matrix<BaseFloat> feat(vec.Dim(), 1);
    feat.CopyColFromVec(vec, 0);

    WriteKaldiObject(feat, feature_wxfilename, binary);

    KALDI_LOG << "Converted vector " << PrintableRxfilename(vector_rxfilename)
              << " to " << PrintableWxfilename(feature_wxfilename);
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

