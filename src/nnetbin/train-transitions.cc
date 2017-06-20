// nnetbin/train-transitions.cc

// Copyright 2015  Brno University of Technology (author: Karel Vesely)
//           2012  Johns Hopkins University (author:  Daniel Povey)

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
#include "tree/context-dep.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Train the transition probabilities in transition-model "
        "(used in nnet1 recipe).\n"
        "\n"
        "Usage: train-transitions [options] "
        "<trans-model-in> <alignments-rspecifier> <trans-model-out>\n"
        "e.g.: train-transitions 1.mdl \"ark:gunzip -c ali.*.gz|\" 2.mdl\n";

    bool binary_write = true;
    MleTransitionUpdateConfig transition_update_config;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    transition_update_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string trans_model_rxfilename = po.GetArg(1),
        ali_rspecifier = po.GetArg(2),
        trans_model_wxfilename = po.GetArg(3);

    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(trans_model_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
    }

    Vector<double> transition_accs;
    trans_model.InitStats(&transition_accs);

    int32 num_done = 0;
    SequentialInt32VectorReader ali_reader(ali_rspecifier);
    for (; !ali_reader.Done(); ali_reader.Next()) {
      const std::vector<int32> alignment(ali_reader.Value());
      for (size_t i = 0; i < alignment.size(); i++) {
        int32 tid = alignment[i];
        BaseFloat weight = 1.0;
        trans_model.Accumulate(weight, tid, &transition_accs);
      }
      num_done++;
    }
    KALDI_LOG << "Accumulated transition stats from " << num_done
              << " utterances.";

    {
      BaseFloat objf_impr, count;
      trans_model.MleUpdate(transition_accs, transition_update_config,
                            &objf_impr, &count);
      KALDI_LOG << "Transition model update: average " << (objf_impr/count)
                << " log-like improvement per frame over " << count
                << " frames.";
    }

    {
      Output ko(trans_model_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Trained transition model and wrote it to "
              << trans_model_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
