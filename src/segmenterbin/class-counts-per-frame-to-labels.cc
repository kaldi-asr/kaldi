// segmenterbin/class-counts-per-frame-to-labels.cc

// Copyright 2016  Vimal Manohar

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
#include "hmm/posterior.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Converts class-counts-per-frame in the format of vectors of vectors of "
        "integers into labels for overlapping SAD.\n"
        "If there is a junk-label in the classes in the frame, then the label "
        "for the frame is set to the junk-label no matter what other labels "
        "are present.\n"
        "If there is only a 0 (silence) in the classes in the frame, then the "
        "label for the frame is set to 0.\n"
        "If there is only one non-zero non-junk class, then the label is set "
        "to 1.\n"
        "Otherwise, the label is set to 2 (overlapping speakers)\n"
        "\n"
        "Usage: class-counts-per-frame-to-labels [options] "
        "<class-counts-posterior-rspecifier> <vector-out-wspecifier>\n";
     
    int32 junk_label = -1;
    ParseOptions po(usage);

    po.Register("junk-label", &junk_label,
                "The label used for segments that are junk. If a frame has "
                "a junk label, it will be considered junk segment, no matter "
                "what other labels the frame contains. Also frames with no "
                "classes seen are labeled junk.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }    
    
    std::string in_fn = po.GetArg(1),
        out_fn = po.GetArg(2);

    int num_done = 0;
    Int32VectorWriter writer(out_fn);
    SequentialPosteriorReader reader(in_fn);
    for (; !reader.Done(); reader.Next(), num_done++) {
      const Posterior &class_counts_per_frame = reader.Value();
      std::vector<int32> labels(class_counts_per_frame.size(), junk_label);

      for (size_t i = 0; i < class_counts_per_frame.size(); i++) {
        const std::vector<std::pair<int32, BaseFloat> > &class_counts = 
          class_counts_per_frame[i];

        if (class_counts.size() == 0) {
          labels[i] = junk_label;
        } else {
          bool silence_found = false;
          std::vector<std::pair<int32, BaseFloat> >::const_iterator it =
            class_counts.begin();
          int32 class_counts_in_frame = 0;
          for (; it != class_counts.end(); ++it) {
            KALDI_ASSERT(it->second > 0);
            if (it->first == 0) {
              silence_found = true;
            } else {
              class_counts_in_frame += static_cast<int32>(it->second);
              if (it->first == junk_label) {
                labels[i] = junk_label;
                break;
              }
            }
          }

          if (class_counts_in_frame == 0) {
            KALDI_ASSERT(silence_found);
            labels[i] = 0;
          } else if (class_counts_in_frame == 1) {
            labels[i] = 1;
          } else {
            labels[i] = 2;
          }
        }
      }
      writer.Write(reader.Key(), labels);
    }
    KALDI_LOG << "Copied " << num_done << " items.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



