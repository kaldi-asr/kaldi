// bin/post-to-phone-post.cc

// Copyright 2012-2013  Johns Hopkins University (author: Daniel Povey)
//                2019  Daniel Povey

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
        "Convert posteriors (or pdf-level posteriors) to phone-level posteriors\n"
        "See also: post-to-pdf-post, post-to-weights, get-post-on-ali\n"
        "\n"
        "First, the usage when your posteriors are on transition-ids (the normal case):\n"
        "Usage: post-to-phone-post [options] <model> <post-rspecifier> <phone-post-wspecifier>\n"
        " e.g.: post-to-phone-post --binary=false 1.mdl \"ark:ali-to-post 1.ali|\" ark,t:-\n"
        "\n"
        "Next, the usage when your posteriors are on pdfs (e.g. if they are neural-net\n"
        "posteriors)\n"
        "post-to-phone-post --transition-id-counts=final.tacc 1.mdl ark:pdf_post.ark ark,t:-\n"
        "See documentation of --transition-id-counts option for more details.";

    std::string tacc_rxfilename;

    ParseOptions po(usage);

    po.Register("transition-id-counts", &tacc_rxfilename, "Rxfilename where vector of counts\n"
                "for transition-ids can be read (would normally come from training data\n"
                "alignments, e.g. from ali-to-post and then post-to-tacc with --per-pdf=false)\n");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        post_rspecifier = po.GetArg(2),
        phone_post_wspecifier = po.GetArg(3);

    kaldi::SequentialPosteriorReader posterior_reader(post_rspecifier);
    kaldi::PosteriorWriter posterior_writer(phone_post_wspecifier);

    TransitionModel trans_model;
    {
      bool binary_in;
      Input ki(model_rxfilename, &binary_in);
      trans_model.Read(ki.Stream(), binary_in);
    }
    int32 num_done = 0;


    if (tacc_rxfilename.empty()) {
      // Input is transition-ids
      for (; !posterior_reader.Done(); posterior_reader.Next()) {
        const kaldi::Posterior &posterior = posterior_reader.Value();
        kaldi::Posterior phone_posterior;
        ConvertPosteriorToPhones(trans_model, posterior, &phone_posterior);
        posterior_writer.Write(posterior_reader.Key(), phone_posterior);
        num_done++;
      }
    } else {
      Vector<BaseFloat> transition_counts;
      ReadKaldiObject(tacc_rxfilename, &transition_counts);
      int32 num_pdfs = trans_model.NumPdfs(),
          num_tids = trans_model.NumTransitionIds();
      if (transition_counts.Dim() != num_tids + 1) {
        KALDI_ERR << "Wrong size for transition counts in " << tacc_rxfilename
                  << ", expected " << num_tids << " + 1, got "
                  << transition_counts.Dim();
      }
      // Maps from pdf-id to a map from phone -> count associated with that
      // phone.
      std::vector<std::unordered_map<int32, BaseFloat> > pdf_to_phones(num_pdfs);

      for (int32 i = 1; i <= num_tids; i++) {
        BaseFloat count = transition_counts(i);
        int32 phone = trans_model.TransitionIdToPhone(i),
            pdf_id = trans_model.TransitionIdToPdf(i);
        // Relying on C++11 value-initialization thingies that should make the
        // map's elements default to zero.
        pdf_to_phones[pdf_id][phone] += count;
      }

      for (int32 i = 0; i < num_pdfs; i++) {
        BaseFloat denominator = 0.0;
        for (auto p: pdf_to_phones[i])
          denominator += p.second;
        for (auto iter = pdf_to_phones[i].begin(); iter != pdf_to_phones[i].end();
             ++iter) {
          if (denominator != 0.0)
            iter->second /= denominator;
          else
            iter->second = 1.0 / pdf_to_phones[i].size();
        }
      }

      // Input is pdf-ids
      for (; !posterior_reader.Done(); posterior_reader.Next()) {
        const kaldi::Posterior &posterior = posterior_reader.Value();
        int32 T = posterior.size();
        kaldi::Posterior phone_posterior(T);
        std::unordered_map<int32, BaseFloat> phone_to_count;
        for (int32 t = 0; t < T; t++) {
          phone_to_count.clear();
          for (auto p : posterior[t]) {
            int32 pdf_id = p.first;
            BaseFloat count = p.second;
            if (pdf_id < 0 || pdf_id >= num_pdfs)
              KALDI_ERR << "pdf-id on input out of range, expected [0.." << (num_pdfs-1)
                        << ", got: " << pdf_id;
            for (auto q: pdf_to_phones[pdf_id]) {
              int32 phone = q.first;
              BaseFloat prob = q.second;
              if (prob != 0.0)
                phone_to_count[phone] += count * prob;
            }
          }
          for (auto p : phone_to_count) {
            phone_posterior[t].push_back(
                std::pair<int32, BaseFloat>(p.first, p.second));
          }
        }
        posterior_writer.Write(posterior_reader.Key(), phone_posterior);
        num_done++;
      }
    }
    KALDI_LOG << "Done converting posteriors to phone posteriors for "
              << num_done << " utterances.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
