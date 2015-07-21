// bin/compute-atwv.cc

// Copyright (c) 2015, Johns Hopkins University (Yenda Trmal<jtrmal@gmail.com>)

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


#include <algorithm>
#include <iomanip>      // std::setw

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/stl-utils.h"
#include "kws/kws-scoring.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    typedef kaldi::int32 int32;
    typedef kaldi::uint32 uint32;
    typedef kaldi::uint64 uint64;

    const char *usage = "Computes the Actual Term-Weighted Value and prints it."
        "\n"
        "Usage: compute-atwv [options]  ref-rspecifier hyp-rspecifier [alignment csv]\n"
        " e.g.: compute-atwv ark:ref.1 ark:hyp.1 ali.csv\n"
        "\n"
        "where the alignment format is compatible with the alignment produced\n"
        "using the F4DE tool -- you are responsible for mapping the utterance\n"
        "identifiers and the term string to the correct ones - use the script\n"
        "utils/int2sym.pl and the utterance/keyword maps\n";

    ParseOptions po(usage);
    KwsTermsAlignerOptions ali_opts;
    TwvMetricsOptions twv_opts;
    int frames_per_sec = 100;

    ali_opts.Register(&po);
    twv_opts.Register(&po);
    po.Register("frames-per-sec", &frames_per_sec,
        "Number of feature vector frames per second. This is used only when"
        "writing the alignment to a file");

    po.Read(argc, argv);

    if ((po.NumArgs() < 3) || (po.NumArgs() > 4)) {
      po.PrintUsage();
      exit(1);
    }

    if (!kaldi::ConvertStringToReal(po.GetArg(1), &twv_opts.audio_duration)) {
      KALDI_ERR << "The duration parameter is not a number";
    }
    if (twv_opts.audio_duration <= 0) {
      KALDI_ERR << "The duration is either negative or zero";
    }

    KwsTermsAligner aligner(ali_opts);
    TwvMetrics twv_scores(twv_opts);

    std::string ref_rspecifier = po.GetArg(2),
        hyp_rspecifier = po.GetArg(3),
        ali_output = po.GetOptArg(4);

    kaldi::SequentialTableReader< kaldi::BasicVectorHolder<double> >
        ref_reader(ref_rspecifier);

    for (; !ref_reader.Done(); ref_reader.Next()) {
      std::string kwid = ref_reader.Key();
      std::vector<double> vals = ref_reader.Value();
      if (vals.size() != 4) {
        KALDI_ERR << "Incorrect format of the reference file"
          << " -- 4 entries expected, " << vals.size() << " given!\n"
          << "Key: " << kwid << std::endl;
      }
      KwsTerm inst(kwid, vals);
      aligner.AddRef(inst);
    }

    kaldi::SequentialTableReader< kaldi::BasicVectorHolder<double> >
        hyp_reader(hyp_rspecifier);

    for (; !hyp_reader.Done(); hyp_reader.Next()) {
      std::string kwid = hyp_reader.Key();
      std::vector<double> vals = hyp_reader.Value();
      if (vals.size() != 4) {
        KALDI_ERR << "Incorrect format of the hypotheses file"
          << " -- 4 entries expected, " << vals.size() << " given!\n"
          << "Key: " << kwid << std::endl;
      }
      KwsTerm inst(kwid, vals);
      aligner.AddHyp(inst);
    }

    KALDI_LOG << "Read " << aligner.nof_hyps() << " hypotheses";
    KALDI_LOG << "Read " << aligner.nof_refs() << " references";
    KwsAlignment ali = aligner.AlignTerms();

    if (ali_output != "") {
      std::fstream fs;
      fs.open(ali_output.c_str(), std::fstream::out);
      ali.WriteCsv(fs, frames_per_sec);
      fs.close();
    }

    TwvMetrics scores(twv_opts);
    scores.AddAlignment(ali);

    std::cout << "aproximate ATWV = "
      << std::fixed << std::setprecision(4)
      << scores.Atwv() << std::endl;
    std::cout << "aproximate STWV = "
      << std::fixed << std::setprecision(4)
      << scores.Stwv() << std::endl;

    float mtwv, mtwv_threshold, otwv;
    scores.GetOracleMeasures(&mtwv, &mtwv_threshold, &otwv);

    std::cout << "aproximate MTWV = "
      << std::fixed << std::setprecision(4)
      << mtwv << std::endl;
    std::cout << "aproximate MTWV threshold = "
      << std::fixed << std::setprecision(4)
      << mtwv_threshold << std::endl;
    std::cout << "aproximate OTWV = "
      << std::fixed << std::setprecision(4)
      << otwv << std::endl;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

