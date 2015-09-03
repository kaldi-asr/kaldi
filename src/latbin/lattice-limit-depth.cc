// latbin/lattice-limit-depth.cc

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
        "Limit the number of arcs crossing any frame, to a specified maximum.\n"
        "Requires an acoustic scale, because forward-backward Viterbi probs are\n"
        "needed, which will be affected by this.\n"
        "\n"
        "Usage: lattice-limit-depth [options] <lattice-rspecifier> <lattice-wspecifier>\n"
        "E.g.: lattice-limit-depth --max-arcs-per-frame=1000 --acoustic-scale=0.1 ark:- ark:-\n";

    ParseOptions po(usage);

    int32 max_arcs_per_frame = 1000;
    BaseFloat acoustic_scale = 1.0;

    po.Register("max-arcs-per-frame", &max_arcs_per_frame,
                "Maximum number of arcs that are allowed to cross any given "
                "frame");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(acoustic_scale != 0.0);
    
    std::string lats_rspecifier = po.GetArg(1),
        lats_wspecifier = po.GetArg(2);
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    CompactLatticeWriter clat_writer(lats_wspecifier);

    int64 num_done = 0;
    double sum_depth_in = 0.0, sum_depth_out = 0.0, total_t = 0.0;
    for (; !clat_reader.Done(); clat_reader.Next()) {
      CompactLattice clat = clat_reader.Value();
      std::string key = clat_reader.Key();
      
      fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &clat);
      
      TopSortCompactLatticeIfNeeded(&clat);

      int32 t;
      BaseFloat depth_in = CompactLatticeDepth(clat, &t);

      CompactLatticeLimitDepth(max_arcs_per_frame, &clat);

      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale),
                        &clat);
      
      BaseFloat depth_out = CompactLatticeDepth(clat);
      
      KALDI_VLOG(2) << "For key " << key << ", depth changed from "
                    << depth_in << " to " << depth_out <<  " over "
                    << t << " frames.";

      total_t += t;
      sum_depth_in += t * depth_in;
      sum_depth_out += t * depth_out;

      clat_writer.Write(key, clat);

      num_done++;
    }
    KALDI_LOG << "Done " << num_done << " lattices.";
    KALDI_LOG << "Overall density changed from " << (sum_depth_in / total_t)
              << " to " << (sum_depth_out / total_t);
    if (num_done != 0) return 0;
    else return 1;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
