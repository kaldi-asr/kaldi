// featbin/process-kaldi-pitch-feats.cc

// Copyright 2013   Pegah Ghahremani
//                  Johns Hopkins University (author: Daniel Povey)
//           2014   IMSL, PKU-HKUST (author: Wei Shi)
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
#include "feat/pitch-functions.h"
#include "feat/wave-reader.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Post-process Kaldi pitch features, consisting of pitch and NCCF, into\n"
        "features suitable for input to ASR system.  Default setup produces\n"
        "3-dimensional features consisting of (pov-feature, pitch-feature,\n"
        "delta-pitch-feature), where pov-feature is warped NCCF, pitch-feature\n"
        "is log-pitch with POV-weighted mean subtraction over 1.5 second window,\n"
        "and delta-pitch-feature is delta feature computed on raw log pitch.\n"
        "In general, you can select from four features: (pov-feature, \n"
        "pitch-feature, delta-pitch-feature, raw-log-pitch), produced in that \n"
        "order, by setting the boolean options (--add-pov-feature, \n"
        "--add-normalized-log-pitch, --add-delta-pitch and --add-raw-log-pitch)\n"
        "\n"
        "Usage: process-kaldi-pitch-feats [options...] <feat-rspecifier> <feats-wspecifier>\n"
        "\n"
        "e.g.: compute-kaldi-pitch-feats [args] ark:- | process-kaldi-pitch-feats ark:- ark:feats.ark\n"
        "\n"
        "See also: compute-kaldi-pitch-feats, compute-and-process-kaldi-pitch-feats\n";

    ParseOptions po(usage);

    int32 srand_seed = 0;
    
    ProcessPitchOptions process_opts;
    process_opts.Register(&po);

    po.Register("srand", &srand_seed, "Seed for random number generator, used to "
                "add noise to delta-log-pitch features");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    srand(srand_seed);
    
    std::string feat_rspecifier = po.GetArg(1),
        feat_wspecifier = po.GetArg(2);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);

    int32 num_done = 0;
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &features = feat_reader.Value();

      Matrix<BaseFloat> processed_feats(features);
      ProcessPitch(process_opts, features, &processed_feats);

      feat_writer.Write(utt, processed_feats);
      num_done++;
    }
    KALDI_LOG << "Post-processed pitch for " << num_done << " utterances.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

