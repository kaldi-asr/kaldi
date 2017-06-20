// bin/align-equal-compiled.cc

// Copyright 2009-2013  Microsoft Corporation
//                      Johns Hopkins University (Author: Daniel Povey)

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
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/training-graph-compiler.h"


/** @brief Write an equally spaced alignment (for getting training started).
*/
int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =  "Write an equally spaced alignment (for getting training started)"
        "Usage:  align-equal-compiled <graphs-rspecifier> <features-rspecifier> <alignments-wspecifier>\n"
        "e.g.: \n"
        " align-equal-compiled 1.fsts scp:train.scp ark:equal.ali\n";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        fst_rspecifier = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignment_wspecifier = po.GetArg(3);


    SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);

    int32 done = 0, no_feat = 0, error = 0;

    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string key = fst_reader.Key();
      if (!feature_reader.HasKey(key)) {
        KALDI_WARN << "No features for utterance " << key;
        no_feat++;
      } else {
        const Matrix<BaseFloat> &features = feature_reader.Value(key);
        VectorFst<StdArc> decode_fst(fst_reader.Value());
        fst_reader.FreeCurrent();  // this stops copy-on-write of the fst
        // by deleting the fst inside the reader, since we're about to mutate
        // the fst by adding transition probs.

        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << key;
          error++;
          continue;
        }
        if (decode_fst.Start() == fst::kNoStateId) {
          KALDI_WARN << "Empty decoding graph for " << key;
          error++;
          continue;
        }

        VectorFst<StdArc> path;
        int32 rand_seed = StringHasher()(key); // StringHasher() produces new anonymous
        // object of type StringHasher; we then call operator () on it, with "key".
        if (EqualAlign(decode_fst, features.NumRows(), rand_seed, &path) ) {
          std::vector<int32> aligned_seq, words;
          StdArc::Weight w;
          GetLinearSymbolSequence(path, &aligned_seq, &words, &w);
          KALDI_ASSERT(aligned_seq.size() == features.NumRows());
          alignment_writer.Write(key, aligned_seq);
          done++;
        } else {
          KALDI_WARN << "AlignEqual: did not align utterence " << key;
          error++;
        }
      }
    }

    if (done != 0 && no_feat == 0 && error == 0) {
      KALDI_LOG << "Success: done " << done << " utterances.";
    } else {
      KALDI_WARN << "Computed " << done << " alignments; " << no_feat
                 << " lacked features, " << error
                 << " had other errors.";
    }
    if (done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


