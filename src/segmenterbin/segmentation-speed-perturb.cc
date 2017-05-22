// segmenterbin/segmentation-speed-perturb.cc

// Copyright 2017   Vimal Manohar (Johns Hopkins University)

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

#include <sstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "segmenter/segmentation.h"
#include "segmenter/segmentation-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Creates copies of segmentations with different speed perturbations "
        "applied.\n"
        "Usage: segmentation-speed-perturb [options] <segmentation-rspecifier> "
        "<segmentation-wspecifier>\n"
        " e.g.: segmentation-speed-perturb ark:1.seg ark:1.sp.seg\n";

    std::string speeds = "0.9:1.0:1.1";
    std::string prefix = "sp";

    ParseOptions po(usage);
    
    po.Register("prefix", &prefix, "Prefix to apply to the key "
                "to index the perturbed segments");
    po.Register("speeds", &speeds, "Colon separated list of "
                "speed perturbation factors to be applied.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string segmentation_rspecifier = po.GetArg(1),
                segmentation_wspecifier  = po.GetArg(2);

    int32 num_done = 0, num_err = 0;

    SegmentationWriter writer(segmentation_wspecifier);
    SequentialSegmentationReader reader(segmentation_rspecifier);

    std::vector<BaseFloat> speeds_vec;
    SplitStringToFloats(speeds, ":,", false, &speeds_vec);

    for (; !reader.Done(); reader.Next()) {
      const std::string &key = reader.Key();
      const Segmentation &seg = reader.Value();

      for (std::vector<BaseFloat>::const_iterator it = speeds_vec.begin();
            it != speeds_vec.end(); ++it, num_done++) {
        if (*it == 1.0) {
          writer.Write(key, seg);
          continue;
        }

        KALDI_ASSERT(*it > 0.1 && *it < 10);

        Segmentation this_seg(seg);
        ScaleFrameShift(1.0 / *it, &this_seg);

        std::ostringstream oss;
        oss << prefix << *it << "-" << key;

        writer.Write(oss.str(), this_seg);
      }
    }
    
    KALDI_LOG << "Speed perturbed " << num_done << " segmentation";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


