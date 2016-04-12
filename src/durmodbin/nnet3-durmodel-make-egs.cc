// durmodbin/nnet3-durmodel-make-egs.cc
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
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "util/common-utils.h"
#include "tree/build-tree.h"
#include "util/parse-options.h"
#include "durmod/kaldi-durmod.h"
int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  using nnet3::NnetExample;

  try {
    const char *usage =
        "Prepare nnet3 training examples for the phone duration model.\n"
        "Usage:  nnet3-durmodel-make-egs [options] <nnet-dur-model> "
        "<trans-model> <alignments-rspecifier> <egs-wspecifier>\n"
        "e.g.: \n"
        "  nnet3-durmodel-make-egs 0.mdl final.mdl egs:ali.1 ark:1.egs";
    ParseOptions po(usage);
    po.Read(argc, argv);
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    std::string nnet_durmodel_filename = po.GetArg(1),
                transmodel_filename = po.GetArg(2),
                alignments_rspecifier = po.GetArg(3),
                examples_wspecifier = po.GetArg(4);
    TransitionModel trans_model;
    ReadKaldiObject(transmodel_filename, &trans_model);
    NnetPhoneDurationModel nnet_durmodel;
    ReadKaldiObject(nnet_durmodel_filename, &nnet_durmodel);
    SequentialInt32VectorReader reader(alignments_rspecifier);
    TableWriter<KaldiObjectHolder<NnetExample> >
                                           example_writer(examples_wspecifier);
    int32 n_done = 0;
    int32 n_egs_done = 0;
    PhoneDurationFeatureMaker feature_maker(nnet_durmodel.GetDurationModel());


    for (; !reader.Done(); reader.Next()) {
      std::string key = reader.Key();
      const std::vector<int32> &alignment = reader.Value();

      std::vector<std::vector<int32> > split;
      SplitToPhones(trans_model, alignment, &split);
      std::vector<std::pair<int32, int32> > pairs;

      for (size_t i = 0; i < split.size(); i++) {
        KALDI_ASSERT(split[i].size() > 0);
        int32 phone = trans_model.TransitionIdToPhone(split[i][0]);
        int32 num_repeats = split[i].size();
        KALDI_ASSERT(num_repeats != 0);
        pairs.push_back(std::make_pair(phone, num_repeats));
      }

      std::vector<NnetExample> egs;
      AlignmentToNnetExamples(nnet_durmodel, feature_maker, pairs, &egs);
      n_egs_done += egs.size();
      for (int i = 0; i < egs.size(); i++)
        example_writer.Write(key, egs[i]);
      n_done++;
    }

    KALDI_LOG << "Wrote " << n_egs_done << " examples.";
    KALDI_LOG << "Done " << n_done << " utterances.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
