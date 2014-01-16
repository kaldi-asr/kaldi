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


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Averages iVectors over all the utterances of each speaker\n"
        "Input is a spk2utt file and a set of iVectors indexed by\n"
        "utterance; output is iVectors indexed by speaker.  If 4\n"
        "arguments are given, the last one is a table for the number"
        "of utterances per speaker (can be useful for PLDA)\n"
        "\n"
        "Usage: ivector-mean <spk2utt-rspecifier> <ivector-rspecifier> "
        "<ivector-wspecifier> [<num-utt-wspecifier>]\n"
        "e.g.: ivector-mean data/spk2utt exp/ivectors.ark exp/spk_ivectors.ark exp/spk_num_utts.ark\n";
    
    ParseOptions po(usage);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string spk2utt_rspecifier = po.GetArg(1),
        ivector_rspecifier = po.GetArg(2),
        ivector_wspecifier = po.GetArg(3),
        num_utts_wspecifier = po.GetOptArg(4);
    
    double spk_sumsq = 0.0;
    Vector<double> spk_sum;
    
    int64 num_spk_done = 0, num_spk_err = 0,
        num_utt_done = 0, num_utt_err = 0;

    RandomAccessBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    BaseFloatVectorWriter ivector_writer(ivector_wspecifier);
    Int32Writer num_utts_writer(num_utts_wspecifier);
    
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();
      if (uttlist.empty()) {
        KALDI_ERR << "Speaker with no utterances.";
      }
      Vector<BaseFloat> spk_mean;
      int32 utt_count = 0;
      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!ivector_reader.HasKey(utt)) {
          KALDI_WARN << "No iVector present in input for utterance " << utt;
          num_utt_err++;
        } else {
          if (utt_count == 0) {
            spk_mean = ivector_reader.Value(utt);
          } else {
            spk_mean.AddVec(1.0, ivector_reader.Value(utt));
          }
          num_utt_done++;
          utt_count++;
        }
      }
      if (utt_count == 0) {
        KALDI_WARN << "Not producing output for speaker " << spk
                   << " since no utterances had iVectors";
        num_spk_err++;
      } else {
        spk_mean.Scale(1.0 / utt_count);
        ivector_writer.Write(spk, spk_mean);
        if (num_utts_wspecifier != "")
          num_utts_writer.Write(spk, utt_count);
        num_spk_done++;
        spk_sumsq += VecVec(spk_mean, spk_mean);
        if (spk_sum.Dim() == 0)
          spk_sum.Resize(spk_mean.Dim());
        spk_sum.AddVec(1.0, spk_mean);
      }
    }
      
    KALDI_LOG << "Computed mean of " << num_spk_done << " speakers ("
              << num_spk_err << " with no utterances), consisting of "
              << num_utt_done << " utterances (" << num_utt_err
              << " absent from input).";

    if (num_spk_done != 0) {
      spk_sumsq /= num_spk_done;
      spk_sum.Scale(1.0 / num_spk_done);
      double mean_length = spk_sum.Norm(2.0),
          spk_length = sqrt(spk_sumsq),
          norm_spk_length = spk_length / sqrt(spk_sum.Dim());
      KALDI_LOG << "Norm of mean of speakers is " << mean_length
                << ", root-mean-square speaker-iVector length divided by "
                << "sqrt(dim) is " << norm_spk_length;
    }

    return (num_spk_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
