// latbin/lattice-depth.cc

// Copyright 2013  Ehsan Variani
//           2013  Johns Hopkins University (Author: Daniel Povey)

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
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    using fst::VectorFst;
    using fst::StdArc;
    typedef StdArc::StateId StateId;

    const char *usage =
        "Compute the lattice depths in terms of the average number of arcs that\n"
        "cross a frame.  See also lattice-depth-per-frame\n"
        "Usage: lattice-depth <lattice-rspecifier> [<depth-wspecifier>]\n"
        "E.g.: lattice-depth ark:- ark,t:-\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() < 1 || po.NumArgs() > 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1);
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);

    std::string depth_wspecifier = po.GetOptArg(2);
    BaseFloatWriter lats_depth_writer(depth_wspecifier);

    int64 num_done = 0;
    double sum_depth = 0.0, total_t = 0.0;
    for (; !clat_reader.Done(); clat_reader.Next()) {
      CompactLattice clat = clat_reader.Value();
      std::string key = clat_reader.Key();

      TopSortCompactLatticeIfNeeded(&clat);

      int32 t;
      BaseFloat depth = CompactLatticeDepth(clat, &t);

      if (depth_wspecifier != "")
        lats_depth_writer.Write(key, depth);

      sum_depth += depth * t;
      total_t += t;
      num_done++;
    }
    KALDI_LOG << "Done " << num_done << " lattices.";
    // Warning: the script egs/s5/*/steps/oracle_wer.sh parses the next line.
    KALDI_LOG << "Overall density is " << (sum_depth / total_t) << " over "
              << total_t << " frames.";
    if (num_done != 0) return 0;
    else return 1;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
