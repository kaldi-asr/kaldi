// kwsbin/generate-proxy-keywords.cc

// Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)

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
#include "fstext/fstext-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    typedef kaldi::int32 int32;
    typedef kaldi::uint64 uint64;
    typedef StdArc::StateId StateId;
    typedef StdArc::Weight Weight;

    const char *usage =
        "Convert the keywords into in-vocabulary words using the given phone level edit distance\n"
        "fst (E.fst). The large lexicon (L2.fst) and inverted small lexicon (L1'.fst) are also\n"
        "expected to present. We actually use the composed FST L2xE.fst, to be more efficient.\n"
        "Ideally we should have used L2xExL1'.fst but this could be quite computationally expensive.\n"
        "Keywords.int is in the transcription format.\n"
        "\n"
        "Usage: generate-proxy-keywords [options]  L2xE.fst L1'.fst keyword-rspecifier fsts-wspecifier\n"
        " e.g.: generate-proxy-keywords L2xE.fst L1'.fst ark:keywords.int ark:proxy.fsts\n";

    ParseOptions po(usage);

    int32 nBest = 100;
    double cost_threshold = 1;
    po.Register("nBest", &nBest, "n best possible in-vocabulary proxy keywords.");
    po.Register("cost-threshold", &cost_threshold, "Cost threshold.");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string L2xE_filename = po.GetArg(1),
        L1_filename = po.GetArg(2),
        transcript_rspecifier = po.GetArg(3),
        fst_wspecifier = po.GetArg(4);


    VectorFst<StdArc> *L2xE = ReadFstKaldi(L2xE_filename);
    VectorFst<StdArc> *L1 = ReadFstKaldi(L1_filename);
    SequentialInt32VectorReader transcript_reader(transcript_rspecifier);
    TableWriter<VectorFstHolder> fst_writer(fst_wspecifier);

    // Start processing the keywords
    int32 n_done = 0;
    for (; !transcript_reader.Done(); transcript_reader.Next()) {
      std::string key = transcript_reader.Key();
      std::vector<int32> transcript = transcript_reader.Value();
      transcript_reader.FreeCurrent();

      KALDI_LOG << "Processing " << key;

      VectorFst<StdArc> fst;
      VectorFst<StdArc> tmp;
      MakeLinearAcceptor(transcript, &fst);

      KALDI_VLOG(1) << "Compose(KW, L2xE)";
      ArcSort(&fst, OLabelCompare<StdArc>());
      Compose(fst, *L2xE, &tmp);
      KALDI_VLOG(1) << "Compose(KWxL2xE, L1')";
      ArcSort(&tmp, OLabelCompare<StdArc>());
      Compose(tmp, *L1, &fst);
      KALDI_VLOG(1) << "Project";
      Project(&fst, PROJECT_OUTPUT);
      KALDI_VLOG(1) << "Prune";
      Prune(&fst, cost_threshold);
      if (nBest > 0) {
        KALDI_VLOG(1) << "Shortest Path";
        ShortestPath(fst, &tmp, nBest, true, true);
      } else {
        tmp = fst;
      }
      KALDI_VLOG(1) << "Remove epsilon";
      RmEpsilon(&tmp);
      KALDI_VLOG(1) << "Determinize";
      Determinize(tmp, &fst);
      ArcSort(&fst, fst::OLabelCompare<StdArc>());

      fst_writer.Write(key, fst);

      n_done++;
    }

    delete L1;
    delete L2xE;
    KALDI_LOG << "Done " << n_done << " keywords";
    return (n_done != 0 ? 0 : 1);    
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
