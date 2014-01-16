// ivectorbin/ivector-compute-dot-products.cc

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
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Computes dot-products between iVectors; useful in application of an\n"
        "iVector-based system.  The 'trials-file' has lines of the form\n"
        "<key1> <key2>\n"
        "and the output will have the form\n"
        "<key1> <key2> [<dot-product>]\n"
        "(if either key could not be found, the dot-product field in the output\n"
        "will be absent, and this program will print a warning)\n"
        "\n"
        "Usage:  ivector-compute-dot-products [options] <trials-in> "
        "<ivector1-rspecifier> <ivector2-rspecifier> <scores-out>\n"
        "e.g.: \n"
        " ivector-compute-dot-products trials ark:train_ivectors.scp ark:test_ivectors.scp trials.scored\n";
    
    ParseOptions po(usage);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string trials_rxfilename = po.GetArg(1),
        ivector1_rspecifier = po.GetArg(2),
        ivector2_rspecifier = po.GetArg(3),
        scores_wxfilename = po.GetArg(4);


    int64 num_done = 0, num_err = 0;
    
    RandomAccessBaseFloatVectorReader ivector1_reader(ivector1_rspecifier);
    RandomAccessBaseFloatVectorReader ivector2_reader(ivector2_rspecifier);
    
    Input ki(trials_rxfilename);

    bool binary = false;
    Output ko(scores_wxfilename, binary);
    double sum = 0.0, sumsq = 0.0;

    std::string line;
    while (std::getline(ki.Stream(), line)) {
      std::vector<std::string> fields;
      SplitStringToVector(line, " \t\n\r", true, &fields);
      if (fields.size() != 2) {
        KALDI_ERR << "Bad line " << (num_done + num_err) << " in input "
                  << "(expected two fields: key1 key2): " << line;
      }
      std::string key1 = fields[0], key2 = fields[1];
      if (!ivector1_reader.HasKey(key1)) {
        KALDI_WARN << "Key " << key1 << " not present in 1st table of ivectors.";
        num_err++;
        continue;
      }
      if (!ivector2_reader.HasKey(key2)) {
        KALDI_WARN << "Key " << key2 << " not present in 2nd table of ivectors.";
        num_err++;
        continue;
      }
      const Vector<BaseFloat> &ivector1 = ivector1_reader.Value(key1),
          &ivector2 = ivector2_reader.Value(key2);
      // The following will crash if the dimensions differ, but
      // they would likely also differ for all the ivectors so it's probably
      // best to just crash.
      BaseFloat dot_prod = VecVec(ivector1, ivector2);
      sum += dot_prod;
      sumsq += dot_prod * dot_prod;
      num_done++;
      ko.Stream() << key1 << ' ' << key2 << ' ' << dot_prod << std::endl;
    }
    
    if (num_done != 0) {
      BaseFloat mean = sum / num_done, scatter = sumsq / num_done,
          variance = scatter - mean * mean, stddev = sqrt(variance);
      KALDI_LOG << "Mean dot-product was " << mean << ", standard deviation was "
                << stddev;
    }
    KALDI_LOG << "Processed " << num_done << " trials " << num_err
              << " had errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
