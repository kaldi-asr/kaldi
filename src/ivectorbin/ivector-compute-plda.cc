// ivectorbin/ivector-mean.cc

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
#include "ivector/plda.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Computes a Plda object (for Probabilistic Linear Discriminant Analysis)\n"
        "from a set of iVectors.  Uses speaker information from a spk2utt file\n"
        "to compute within and between class variances.\n"
        "\n"
        "Usage:  ivector-compute-plda [options] <spk2utt-rspecifier> <ivector-rspecifier> "
        "<plda-out>\n"
        "e.g.: \n"
        " ivector-compute-plda ark:spk2utt ark,s,cs:ivectors.ark plda\n";
    
    ParseOptions po(usage);

    bool binary = true;
    PldaEstimationConfig plda_config;

    plda_config.Register(&po);

    po.Register("binary", &binary, "Write output in binary mode");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string spk2utt_rspecifier = po.GetArg(1),
        ivector_rspecifier = po.GetArg(2),
        plda_wxfilename = po.GetArg(3);
    
    int64 num_spk_done = 0, num_spk_err = 0,
        num_utt_done = 0, num_utt_err = 0;

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    
    PldaStats plda_stats;
    
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();
      if (uttlist.empty()) {
        KALDI_ERR << "Speaker with no utterances.";
      }
      std::vector<Vector<BaseFloat> > ivectors;
      ivectors.reserve(uttlist.size());

      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!ivector_reader.HasKey(utt)) {
          KALDI_WARN << "No iVector present in input for utterance " << utt;
          num_utt_err++;
        } else {
          ivectors.resize(ivectors.size() + 1);
          ivectors.back() = ivector_reader.Value(utt);
          num_utt_done++;
        }
      }
      if (ivectors.size() == 0) {
        KALDI_WARN << "Not producing output for speaker " << spk
                   << " since no utterances had iVectors";
        num_spk_err++;
      } else {
        Matrix<double> ivector_mat(ivectors.size(), ivectors[0].Dim());
        for (size_t i = 0; i < ivectors.size(); i++)
          ivector_mat.Row(i).CopyFromVec(ivectors[i]);
        double weight = 1.0; // The code supports weighting but
                             // we don't support this at the command-line
                             // level yet.
        plda_stats.AddSamples(weight, ivector_mat);
        num_spk_done++;
      }
    }
      
    KALDI_LOG << "Accumulated stats from " << num_spk_done << " speakers ("
              << num_spk_err << " with no utterances), consisting of "
              << num_utt_done << " utterances (" << num_utt_err
              << " absent from input).";
    
    if (num_spk_done == 0)
      KALDI_ERR << "No stats accumulated, unable to estimate PLDA.";
    if (num_spk_done == num_utt_done)
      KALDI_ERR << "No speakers with multiple utterances, "
                << "unable to estimate PLDA.";
    
    plda_stats.Sort();
    PldaEstimator plda_estimator(plda_stats);
    Plda plda;
    plda_estimator.Estimate(plda_config, &plda);

    WriteKaldiObject(plda, plda_wxfilename, binary);
    
    return (num_spk_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
