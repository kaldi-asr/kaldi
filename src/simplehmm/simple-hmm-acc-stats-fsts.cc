// simplehmmbin/simple-hmm-acc-stats-fsts.cc

// Copyright 2016   Vimal Manohar (Johns Hopkins University)

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
#include "simplehmm/simple-hmm.h"
#include "hmm/hmm-utils.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Accumulate stats for simple HMM models from FSTs directly.\n"
        "Usage:   simple-hmm-acc-stats-fsts [options] <model-in> <graphs-rspecifier> "
        "<likes-rspecifier> <pdf2class_rxfilename> <stats-out>\n"
        "e.g.: \n"
        " simple-hmm-acc-stats-fsts 1.mdl ark:graphs.fsts scp:likes.scp pdf2class_map 1.stats\n";

    ParseOptions po(usage);

    BaseFloat acoustic_scale = 1.0;
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;

    po.Register("transition-scale", &transition_scale,
                "Transition-probability scale [relative to acoustics]");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("self-loop-scale", &self_loop_scale,
                "Scale of self-loop versus non-self-loop log probs [relative to acoustics]");
    po.Read(argc, argv);


    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        fst_rspecifier = po.GetArg(2),
        likes_rspecifier = po.GetArg(3),
        pdf2class_map_rxfilename = po.GetArg(4),
        accs_wxfilename = po.GetArg(5);

    simple_hmm::SimpleHmm model;
    ReadKaldiObject(model_in_filename, &model);

    SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
    RandomAccessBaseFloatMatrixReader likes_reader(likes_rspecifier);

    std::vector<int32> pdf2class;
    {
      Input ki(pdf2class_map_rxfilename);
      std::string line;
      while (std::getline(ki.Stream(), line)) {
        std::vector<std::string> parts;
        SplitStringToVector(line, " ", true, &parts);
        if (parts.size() != 2) {
          KALDI_ERR << "Invalid line " << line
                    << " in pdf2class-map " << pdf2class_map_rxfilename;
        }
        int32 pdf_id = std::atoi(parts[0].c_str()),
              class_id = std::atoi(parts[1].c_str());

        if (pdf_id != pdf2class.size())
          KALDI_ERR << "pdf2class-map is not sorted or does not contain "
                    << "pdf " << pdf_id - 1 << " in " 
                    << pdf2class_map_rxfilename;
        
        if (pdf_id < pdf2class.size()) 
          KALDI_ERR << "Duplicate pdf " << pdf_id
                    << " in pdf2class-map " << pdf2class_map_rxfilename;

        pdf2class.push_back(class_id);
      }
    }

    int32 num_done = 0, num_err = 0;
    double tot_like = 0.0, tot_t = 0.0;
    int64 frame_count = 0;

    Vector<double> transition_accs;
    model.InitStats(&transition_accs);

    SimpleHmmComputation computation(model, pdf2class_map);

    for (; !fst_reader.Done(); fst_reader.Next()) {
      const std::string &utt = fst_reader.Key();

      if (!likes_reader.HasKey(utt)) {
        num_err++;
        KALDI_WARN << "No likes for utterance " << utt;
        continue;
      }

      const Matrix<BaseFloat> &likes = likes_reader.Value(utt);
      VectorFst<StdArc> decode_fst(fst_reader.Value());
      fst_reader.FreeCurrent();  // this stops copy-on-write of the fst
      // by deleting the fst inside the reader, since we're about to mutate
      // the fst by adding transition probs.

      if (likes.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_err++;
        continue;
      }

      if (likes.NumCols() != pdf2class.size()) {
        KALDI_ERR << "Mismatch in pdf dimension in log-likelihood matrix "
                  << "and pdf2class map; " << likes.NumCols() << " vs "
                  << pdf2class.size();
      }

      // Add transition-probs to the FST.
      AddTransitionProbs(model, transition_scale, self_loop_scale,
                         &decode_fst);

      BaseFloat tot_like_this_utt = 0.0, tot_weight = 0.0;
      if (!computation.Compute(decode_fst, likes, acoustic_scale,
                               &transition_accs,
                               &tot_like_this_utt, &tot_weight)) {
        KALDI_WARN << "Failed to do computation for utterance " << utt;
        num_err++;
      }
      tot_like += tot_like_this_utt;
      tot_t += tot_weight;
      frame_count += likes.NumRows();

      num_done++;
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors.";

    KALDI_LOG << "Overall avg like per frame = "
              << (tot_like/tot_t) << " over " << tot_t << " frames.";

    {
      Output ko(accs_wxfilename, binary);
      transition_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


