// decoder/kaldi-decoder-test.cc

// Copyright 2009-2011  Lukas Burget, Mirko Hannemann

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
#include "matrix/matrix-lib.h"
#include "gmm/am-diag-gmm.h"
#include "./decodable-am-diag-gmm.h"
#include "decoder/kaldi-decoder.h"
#include "itf/decodable-itf.h"

typedef fst::ConstFst<fst::StdArc> FstType;
// typedef fst::VectorFst<fst::StdArc> FstType;


int main(int argc, char *argv[]) {
  int retval = 0;

  const char *fea_file = "./test_utterance_4.plp";
  const char *am_file = "./am_diag_gmm";
  const char *rec_net = "./reconet_right.fst";

// const char *fea_file = "/mnt/matylda4/burget/UBM-ASR/branches/clean/src/decoder/tmp/fea/en_6179_059223_059708_B0.plp";
// const char *am_file = "/mnt/matylda4/burget/UBM-ASR/branches/clean/src/decoder/tmp/MMF.Kaldi";
// const char *rec_net = "/mnt/matylda4/burget/UBM-ASR/branches/clean/src/decoder/tmp/reconet.fst";

  try {
    std::ifstream                     is_am;
    kaldi::Matrix<kaldi::BaseFloat>   feature_matrix;
    kaldi::AmDiagGmm       acoustic_model;

    // read data from disk
    is_am.open(am_file);
    KALDI_ASSERT(is_am.good());
    acoustic_model.Read(is_am, false);
    {
      std::ifstream is(fea_file);
      ReadHtk(is, &feature_matrix, NULL);
    }
    fst::StdFst *readfst = fst::StdFst::Read(rec_net);
    FstType recognition_net(*readfst);

    kaldi::DecodableAmDiagGmmUnmapped decodable(acoustic_model, feature_matrix);
    kaldi::KaldiDecoderOptions options;

    const char *usage =
        "Decode features using GMM-based model.\n"
        "Usage: kaldi-decoder-test [options] "
          "model-in fst-in feature-file words-list\n";

    kaldi::ParseOptions po(usage);
    options.Register(&po, true);  // true == include obscure settings.
    po.Read(argc, argv);

    // if (po.NumArgs() != 4) {
    //  po.PrintUsage();
    //  exit(1);
    // }

    // std::string model_in_filename = po.GetArg(1),
    //    fst_in_filename = po.GetArg(2),
    //    feature_rspecifier = po.GetArg(3),
    //    words_wspecifier = po.GetArg(4);


    kaldi::KaldiDecoder<kaldi::DecodableAmDiagGmmUnmapped, FstType>
        decoder(options);

    decoder.SetMaxActiveTokens(16000);
    decoder.SetBeamPruning(200.0);  // 200.0
    decoder.SetLmScale(1.0);  // 13.0
    decoder.SetWordPenalty(0.0);  // -5.0

    fst::VectorFst<fst::StdArc>* word_links =
        decoder.Decode(recognition_net, &decodable);

    fst::StdArc::StateId s = word_links->Start();
    std::string word;
    while ((s != fst::kNoStateId) &&
          (word_links->Final(s) == fst::StdArc::Weight::Zero())) {
          for (fst::ArcIterator<fst::Fst<fst::StdArc> > aiter(*word_links, s);
            !aiter.Done(); aiter.Next()) {
        // look at first emitting arc -> arc.ilabel
        const fst::StdArc &arc = aiter.Value();
        word = recognition_net.OutputSymbols()->Find(arc.olabel);
        // not possible without having symbol lists

        // std::cerr << s <<  "->" << arc.nextstate << " " << arc.ilabel << ":"
        //       << arc.olabel << "( " << word << " )/" << arc.weight << '\n';
        std::cerr << word << " " << arc.weight << '\n';
        s = arc.nextstate;  // move forward ...
      }  // for: links loop
      if (word_links->Final(s) != fst::StdArc::Weight::Zero()) break;
    }  // while: state loop
    delete(readfst);
    delete(word_links);
  }
  catch(const std::exception& rExc) {
    std::cerr << "Exception thrown" << '\n';
    std::cerr << rExc.what() << '\n';
    retval = -1;
  }
  return retval;
}
