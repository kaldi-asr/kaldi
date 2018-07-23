// latbin/lattice-depth-per-frame.cc

// Copyright 2013  Ehsan Variani
//      2013,2016  Johns Hopkins University (Author: Daniel Povey)

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
        "For each lattice, compute a vector of length (num-frames) saying how\n"
        "may arcs cross each frame.  See also lattice-depth\n"
        "Usage: lattice-depth-per-frame <lattice-rspecifier> <depth-wspecifier> [<lattice-wspecifier>]\n"
        "The final <lattice-wspecifier> allows you to write the input lattices out\n"
        "in case you want to do something else with them as part of the same pipe.\n"
        "E.g.: lattice-depth-per-frame ark:- ark,t:-\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1);
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);

    std::string depth_wspecifier = po.GetArg(2);
    Int32VectorWriter lats_depth_writer(depth_wspecifier);

    std::string lattice_wspecifier = po.GetOptArg(3);
    CompactLatticeWriter clat_writer(lattice_wspecifier);

    int64 num_done = 0;

    for (; !clat_reader.Done(); clat_reader.Next()) {
      CompactLattice clat = clat_reader.Value();
      std::string key = clat_reader.Key();

      TopSortCompactLatticeIfNeeded(&clat);

      std::vector<int32> depth_per_frame;
      CompactLatticeDepthPerFrame(clat, &depth_per_frame);

      lats_depth_writer.Write(key, depth_per_frame);

      if (!lattice_wspecifier.empty())
        clat_writer.Write(key, clat);

      num_done++;
    }
    KALDI_LOG << "Done " << num_done << " lattices.";
    if (num_done != 0) return 0;
    else return 1;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
