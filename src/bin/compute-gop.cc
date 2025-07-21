// bin/compute-gop.cc

// Copyright 2019  Junbo Zhang

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

/**
   This code computes Goodness of Pronunciation (GOP) and extracts phone-level
   pronunciation feature for mispronunciations detection tasks, the reference:

   "Improved mispronunciation detection with deep neural network trained acoustic
   models and transfer learning based logistic regression classifiers"
   by Hu et al., Speech Comunication, 2015.

   GOP is widely used to detect mispronunciations. The DNN-based GOP was defined
   as the log phone posterior ratio between the canonical phone and the one with
   the highest score.

   To compute GOP, we need to compute Log Phone Posterior (LPP):
     LPP(p) = \log p(p|\mathbf o; t_s,t_e)
   where {\mathbf o} is the input observations, p is the canonical phone,
   {t_s, t_e} are the start and end frame indexes.

   LPP could be calculated as the average of the frame-level LPP, i.e. p(p|o_t):
     LPP(p) = \frac{1}{t_e-t_s+1} \sum_{t=t_s}^{t_e}\log p(p|o_t)
     p(p|o_t) = \sum_{s \in p} p(s|o_t)
   where s is the senone label, {s|s \in p} is the states belonging to those
   triphones whose current phone is p.

   GOP is extracted from LPP:
     GOP(p) = \log \frac{LPP(p)}{\max_{q\in Q} LPP(q)}

   An array of a GOP-based feature for each phone is extracted as well, which
   could be used to train a classifier to detect mispronunciations. Normally the
   classifier-based approach archives better performance than the GOP-based approach.

   The GOP-based feature is defined as:
     {[LPP(p_1),\cdots,LPP(p_M), LPR(p_1|p_i), \cdots, LPR(p_j|p_i),\cdots]}^T

   where the Log Posterior Ratio (LPR) between phone p_j and p_i is defined as:
     LPR(p_j|p_i) = \log p(p_j|\mathbf o; t_s, t_e) - \log p(p_i|\mathbf o; t_s, t_e)
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "hmm/tree-accu.h"
#include "hmm/posterior.h"

namespace kaldi {

int32 PhoneNum(const std::vector<std::set<int32> > &pdf2phones) {
  int32 phone_num = 0;
  for (auto &pdf: pdf2phones) {
    if(!pdf.empty()) {
      phone_num = std::max(phone_num, 1 + *pdf.rbegin());
    }
  }
  return phone_num;
}

/** ComputeLpps compute log posteriors for pure-phones by sum the posterior
    of the states belonging to those triphones whose current phone is the canonical
    phone:

    p(p|o_t) = \sum_{s \in p} p(s|o_t),

    where s is the senone label, {s|s \in p} is the states belonging to those
    riphones whose current phone is the canonical phone p.

 */
void ComputeLpps(const Matrix<BaseFloat> &prob,
                 const std::vector<std::set<int32> > &pdf2phones,
                 Matrix<BaseFloat> *lpps) {
  int32 mono_num = PhoneNum(pdf2phones);
  lpps->Resize(prob.NumRows(), mono_num, kSetZero);
  for (int32 i = 0; i < prob.NumCols(); i++) {
    SubMatrix<float> src(prob, 0, prob.NumRows(), i, 1);
    for (int32 ph : pdf2phones.at(i)) {
      SubMatrix<float> dst(*lpps, 0, prob.NumRows(), ph, 1);
      dst.AddMat(1, src);
    }
  }
  lpps->ApplyLog();
}

}  // namespace kaldi

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Compute Goodness Of Pronunciation (GOP) from a matrix of "
        "probabilities (e.g. from nnet3-compute).\n"
        "Usage:  compute-gop [options] <model> <alignments-rspecifier> "
        "<prob-matrix-rspecifier> <gop-wspecifier> "
        "[<phone-feature-wspecifier>]\n"
        "e.g.:\n"
        " nnet3-compute [args] | compute-gop 1.mdl ark:ali-phone.1 ark:-"
        " ark:gop.1 ark:phone-feat.1\n";

    ParseOptions po(usage);

    bool log_applied = true;
    std::string phone_map_rxfilename;
    std::string skip_phones_string = "0";

    po.Register("log-applied", &log_applied,
        "If true, assume the input probabilities have been applied log.");
    po.Register("phone-map", &phone_map_rxfilename,
                "File name containing old->new phone mapping (each line is: "
                "old-integer-id new-integer-id)");
    po.Register("skip_phones_string", &skip_phones_string,
                "Do not write features and gops for those phones");

    po.Read(argc, argv);

    if (po.NumArgs() != 4 && po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
                alignments_rspecifier = po.GetArg(2),
                prob_rspecifier = po.GetArg(3),
                gop_wspecifier = po.GetArg(4),
                feat_wspecifier = po.GetArg(5);

    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
    }
    std::vector<std::set<int32> > pdf2phones;
    GetPdfToPhonesMap(trans_model, &pdf2phones);
    int32 phone_num = trans_model.NumPhones();

    std::vector<int32> phone_map;
    if (phone_map_rxfilename != "") {
      // Map phone IDs
      ReadPhoneMap(phone_map_rxfilename, &phone_map);
      std::vector<std::set<int32> > pdf2phones_old(pdf2phones);
      for (int32 i = 0; i < pdf2phones_old.size(); i++) {
        pdf2phones[i].clear();
        for (int32 ph : pdf2phones_old.at(i)) {
          pdf2phones[i].insert(phone_map[ph]);
        }
      }
      phone_num = PhoneNum(pdf2phones);
    }

    std::set<int32> skip_phones;
    if (skip_phones_string != "") {
      std::vector<int32> skip_phones_vec;
      SplitStringToIntegers(skip_phones_string, ":", false, &skip_phones_vec);
      for (int32 ph: skip_phones_vec) {
        skip_phones.insert(ph);
      }
    }

    RandomAccessInt32VectorReader alignment_reader(alignments_rspecifier);
    SequentialBaseFloatMatrixReader prob_reader(prob_rspecifier);
    PosteriorWriter gop_writer(gop_wspecifier);
    BaseFloatVectorWriter feat_writer(feat_wspecifier);

    int32 num_done = 0;
    for (; !prob_reader.Done(); prob_reader.Next()) {
      std::string key = prob_reader.Key();
      if (!alignment_reader.HasKey(key)) {
        KALDI_WARN << "No alignment for utterance " << key;
        continue;
      }
      auto alignment = alignment_reader.Value(key);
      Matrix<BaseFloat> &probs = prob_reader.Value();
      if (log_applied) probs.ApplyExp();

      Matrix<BaseFloat> lpps;
      ComputeLpps(probs, pdf2phones, &lpps);

      int32 frame_num = alignment.size();
      if (alignment.size() != probs.NumRows()) {
        KALDI_WARN << "The frame numbers of alignment and prob are not equal.";
        if (frame_num > probs.NumRows()) frame_num = probs.NumRows();
      }

      KALDI_ASSERT(frame_num > 0);
      int32 cur_phone_id = alignment[0];
      int32 duration = 0;
      Vector<BaseFloat> phone_level_feat(1 + phone_num * 2);  // [phone LPPs LPRs]
      SubVector<BaseFloat> lpp_part(phone_level_feat, 1, phone_num);
      std::vector<Vector<BaseFloat> > phone_level_feat_stdvector;
      Posterior posterior_gop;
      for (int32 i = 0; i < frame_num; i++) {
        // Calculate LPP and LPR for each pure-phone
        Vector<BaseFloat> frame_level_lpp(phone_num);
        frame_level_lpp.CopyRowFromMat(lpps, i);

        // Ignore the phones in skip_phones
        for (auto &skip_ph: skip_phones) {
          frame_level_lpp(skip_ph) = -10;
        }

        // LPP(p)=\frac{1}{t_e-t_s+1} \sum_{t=t_s}^{t_e}\log p(p|o_t)
        lpp_part.AddVec(1, frame_level_lpp);
        duration++;

        int32 next_phone_id = (i < frame_num - 1) ? alignment[i + 1]: -1;
        if (next_phone_id != cur_phone_id) {
          int32 phone_id = phone_map.empty() ? cur_phone_id : phone_map[cur_phone_id];

          // The current phone's feature have been ready
          lpp_part.Scale(1.0 / duration);

          // LPR(p_j|p_i)=\log p(p_j|\mathbf o; t_s, t_e)-\log p(p_i|\mathbf o; t_s, t_e)
          for (int k = 0; k < phone_num; k++)
            phone_level_feat(1 + phone_num + k) = lpp_part(phone_id) - lpp_part(k);

          // Compute GOP from LPP
          // GOP(p)=\log \frac{LPP(p)}{\max_{q\in Q} LPP(q)}
          BaseFloat gop = lpp_part(phone_id) - lpp_part.Max();

          if (skip_phones.find(phone_id) == skip_phones.end()) {
            phone_level_feat(0) = phone_id;
            phone_level_feat_stdvector.push_back(phone_level_feat);
            std::vector<std::pair<int32, BaseFloat> > posterior_item;
            posterior_item.push_back(std::make_pair(phone_id, gop));
            posterior_gop.push_back(posterior_item);
          }

          // Reset
          phone_level_feat.Set(0);
          duration = 0;
        }
        cur_phone_id = next_phone_id;
      }

      // Write GOPs and the GOP-based features
      int32 example_id = 0;
      for (auto &feat: phone_level_feat_stdvector) {
        std::string cur_key = key + "." + std::to_string(example_id);
        feat_writer.Write(cur_key, feat);
        example_id++;
      }
      gop_writer.Write(key, posterior_gop);
      num_done++;
    }

    KALDI_LOG << "Processed " << num_done << " prob matrices.";
    return (num_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
