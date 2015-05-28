// bin/post-to-tacc.cc

// Copyright 2009-2011 Chao Weng  Microsoft Corporation
//           2015   Minhua Wu

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
#include "hmm/transition-model.h"
#include "hmm/posterior.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;  

    const char *usage =
        "From posteriors, compute transition-accumulators\n"
        "The output is a vector of counts/soft-counts, indexed by transition-id)\n"
        "Note: the model is only read in order to get the size of the vector\n"
        "\n"
        "Usage: post-to-tacc [options] <model> <post-rspecifier> <accs>\n"
        " e.g.: post-to-tacc --binary=false 1.mdl \"ark:ali-to-post 1.ali|\" 1.tacc\n";

    bool binary = true;
    bool per_pdf = false;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode.");
    po.Register("per-pdf", &per_pdf, "if ture, accumulate counts per pdf-id"
                " rather than transition-id. (default: false)");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
      
    std::string model_rxfilename = po.GetArg(1),
        post_rspecifier = po.GetArg(2),
        accs_wxfilename = po.GetArg(3);

    kaldi::SequentialPosteriorReader posterior_reader(post_rspecifier);
    
    int32 num_transition_ids;
    
      bool binary_in;
      Input ki(model_rxfilename, &binary_in);
      TransitionModel trans_model;
      trans_model.Read(ki.Stream(), binary_in);
      num_transition_ids = trans_model.NumTransitionIds();
    
    Vector<double> transition_accs(num_transition_ids+1); // +1 because they're
    // 1-based; position zero is empty.  We'll write as float.
    int32 num_done = 0;      
    
    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      const kaldi::Posterior &posterior = posterior_reader.Value();
      int32 num_frames = static_cast<int32>(posterior.size());
      for (int32 i = 0; i < num_frames; i++) {
        for (int32 j = 0; j < static_cast<int32>(posterior[i].size()); j++) {
          int32 tid = posterior[i][j].first;
          if (tid <= 0 || tid > num_transition_ids)
            KALDI_ERR << "Invalid transition-id " << tid
                      << " encountered for utterance "
                      << posterior_reader.Key();
          transition_accs(tid) += posterior[i][j].second;
        }
      }
      num_done++;
    }

    if (per_pdf) {
      KALDI_LOG << "accumulate counts per pdf-id";
      int32 num_pdf_ids = trans_model.NumPdfs();
      Vector<double> pdf_accs(num_pdf_ids);
      for (int32 i = 1; i < num_transition_ids; i++) {
        int32 pid = trans_model.TransitionIdToPdf(i);
        pdf_accs(pid) += transition_accs(i);
      }
      Vector<BaseFloat> pdf_accs_float(pdf_accs);
      Output ko(accs_wxfilename, binary);
      pdf_accs_float.Write(ko.Stream(), binary);
    } else {
      Vector<BaseFloat> transition_accs_float(transition_accs);
      Output ko(accs_wxfilename, binary);
      transition_accs_float.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Done computing transition stats over "
              << num_done << " utterances; wrote stats to "
              << accs_wxfilename;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

