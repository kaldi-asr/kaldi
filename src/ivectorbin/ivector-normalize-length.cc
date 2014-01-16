// ivectorbin/ivector-normalize-length.cc

// Copyright 2013  Daniel Povey

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
#include "gmm/am-diag-gmm.h"
#include "ivector/ivector-extractor.h"
#include "thread/kaldi-task-sequence.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Normalize length of iVectors to equal sqrt(feature-dimension)\n"
        "\n"
        "Usage:  ivector-normalize-length [options] <ivector-rspecifier>"
        "<ivector-wspecifier>\n"
        "e.g.: \n"
        " ivector-normalize-length ark:ivectors.ark ark:normalized_ivectors.ark\n";
    
    ParseOptions po(usage);
    bool normalize = true;

    po.Register("normalize", &normalize,
                "Set this to false to disable normalization");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string ivector_rspecifier = po.GetArg(1),
        ivector_wspecifier = po.GetArg(2);


    int32 num_done = 0;
    
    double tot_ratio = 0.0, tot_ratio2 = 0.0;

    SequentialBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    BaseFloatVectorWriter ivector_writer(ivector_wspecifier);

    
    for (; !ivector_reader.Done(); ivector_reader.Next()) {
      std::string key = ivector_reader.Key();
      Vector<BaseFloat> ivector = ivector_reader.Value();
      BaseFloat norm = ivector.Norm(2.0);
      BaseFloat ratio = norm / sqrt(ivector.Dim()); // how much larger it is
                                                    // than it would be, in
                                                    // expectation, if normally
                                                    // distributed.
      KALDI_VLOG(2) << "Ratio for key " << key << " is " << ratio;
      if (ratio == 0.0) {
        KALDI_WARN << "Zero iVector";
      } else {
        if (normalize) ivector.Scale(1.0 / ratio);
      }
      ivector_writer.Write(key, ivector);
      tot_ratio += ratio;
      tot_ratio2 += ratio * ratio;
      num_done++;
    }

    KALDI_LOG << "Processed " << num_done << " iVectors.";
    if (num_done != 0) {
      BaseFloat avg_ratio = tot_ratio / num_done,
          ratio_stddev = sqrt(tot_ratio2 / num_done - avg_ratio * avg_ratio);
      KALDI_LOG << "Average ratio of iVector to expected length was "
                << avg_ratio << ", standard deviation was " << ratio_stddev;
    }      
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
