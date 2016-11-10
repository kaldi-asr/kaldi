// kwsbin/compute-atwv.cc

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
        "Usage: compute-atwv [options] <nof-trials> <ref-rspecifier> <hyp-rspecifier> [alignment-csv-filename]\n"
        " e.g.: compute-atwv 32485.4 ark:ref.1 ark:hyp.1 ali.csv\n"
        "   or: compute-atwv 32485.4 ark:ref.1 ark:hyp.1\n"
        "\n"
        "NOTES: \n"
        "  a) the number of trials is usually equal to the size of the searched\n"
        "     collection in seconds\n"
        "  b  the ref-rspecifier/hyp-rspecifier are the kaldi IO specifiers for both\n"
        "     the reference and the hypotheses (found hits), respectively.\n"
        "     The format is the same for both of them. Each line is of \n"
        "     the following format\n"
        "\n"
        "     <KW-ID> <utterance-id> <start-frame> <end-frame> <score>\n\n"
        "     e.g.:\n\n"
        "     KW106-189 348 459 560 0.8\n"
        "\n"
        "  b) the alignment-csv-filename is an optional parameter. If present,\n"
        "     the alignment i.e. detailed information about what hypotheses match\n"
        "     up with which reference entries will be generated. The alignemnt\n"
        "     file format is equivalent to the alignment file produced using\n"
        "     the F4DE tool. However, we do not set some fields and the utterance\n"
        "     identifiers are numeric. You can use the script utils/int2sym.pl\n"
        "     and the utterance/keyword maps to convert the numerical ids into text\n"
        "  c) the scores are expected to be probabilities. Please note that\n"
        "     the output from the kws-search is in -log(probability).\n"
        "  d) compute-atwv does not perform any score normalization (it's just\n"
        "     for scoring purposes). Without score normalization/calibration\n"
        "     the performance of the search will be quite poor.\n";

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

