// durmodbin/nnet3-durmodel-estimate-avg-logprobs.cc

// Copyright 2015 Hossein Hadian

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
#include "hmm/hmm-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "hmm/transition-model.h"
#include "durmod/kaldi-durmod.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using kaldi::CompactLatticeArc;

    const char *usage =
      "This program estimates the average logprobs for each phone-in-context over\n"
      "some training examples so that they are later subtracted from logprobs\n"
      "during the rescoring stage.\n"
      "Usage: nnet3-durmodel-estimate-avg-logprobs [options] <nnet3-dur-model> "
      "<trans-model> <alignments-rspecifier> <output-file>\n"
      "e.g.: \n"
      "nnet3-durmodel-estimate-avg-logprobs nnet_durmodel.mdl final.mdl "
      "ark:ali.1 logprobs.data\n";

    int32 srand_seed = 0;
    int32 num_examples_to_process = 1000000;
    bool binary_write = true;
    int left_context = 1, right_context = 0;

    ParseOptions po(usage);
    po.Register("left-context", &left_context, "left context");
    po.Register("right-context", &right_context, "right context");
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    po.Register("num-examples", &num_examples_to_process, "Number of phone "
                "examples to compute logprobs for.");
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    srand(srand_seed);
    TransitionModel trans_model;
    std::string nnet_durmodel_filename = po.GetArg(1),
                model_filename = po.GetArg(2),
                alignments_rspecifier = po.GetArg(3),
                out_filename = po.GetOptArg(4);

    ReadKaldiObject(model_filename, &trans_model);
    NnetPhoneDurationModel nnet_durmodel;
    ReadKaldiObject(nnet_durmodel_filename, &nnet_durmodel);
    SequentialInt32VectorReader reader(alignments_rspecifier);

    KALDI_ASSERT(left_context <= nnet_durmodel.LeftContext());
    KALDI_ASSERT(right_context <= nnet_durmodel.RightContext());

    unordered_map<std::vector<int32>, BaseFloat, VectorHasher<int32> > context_to_logprob;
    unordered_map<std::vector<int32>, int32, VectorHasher<int32> > context_to_count;

    NnetPhoneDurationScoreComputer durmodel_scorer(nnet_durmodel);
    int num_examples_processed = 0;
    std::ofstream ff(out_filename.c_str());
    for (; !reader.Done(); reader.Next()) {
      //      if (WithProb(0.5))  // to shuffle a little
      //        continue;
      std::string key = reader.Key();
      const std::vector<int32> &alignment = reader.Value();

      std::vector<std::vector<int32> > split;
      SplitToPhones(trans_model, alignment, &split);
      std::vector<std::pair<int32, int32> > pairs;

      for (int i = 0; i < nnet_durmodel.LeftContext(); i++)
        pairs.push_back(std::make_pair(0, 0));
      for (int i = 0; i < split.size(); i++) {
        KALDI_ASSERT(split[i].size() > 0);
        int32 phone = trans_model.TransitionIdToPhone(split[i][0]);
        int32 num_repeats = split[i].size();
        KALDI_ASSERT(num_repeats != 0);
        pairs.push_back(std::make_pair(phone, num_repeats));
      }
      for (int i = 0; i < nnet_durmodel.RightContext(); i++)
        pairs.push_back(std::make_pair(0, 0));
      ff << key << " ";
      for (int i = nnet_durmodel.LeftContext();
           i < pairs.size() - nnet_durmodel.RightContext(); i++) {
        const std::vector<std::pair<int32, int32> > phone_dur_context
                         (pairs.begin() + i - nnet_durmodel.LeftContext(),
                          pairs.begin() + i + nnet_durmodel.RightContext() + 1);
        BaseFloat logprob = durmodel_scorer.GetLogProb(phone_dur_context);
        ff << phone_dur_context[nnet_durmodel.LeftContext()].first << ":" << logprob << " ";
        num_examples_processed++;
      }
      ff << "\n";
      //      if (num_examples_processed > num_examples_to_process)
      //        break;
    }

    KALDI_LOG << "Wrote logprobs for " << num_examples_processed
              << " phones. ";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

