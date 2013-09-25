// sgmmbin/sgmm-acc-fmllrbasis-ali.cc

// Copyright 2009-2011  Saarland University
// Author:  Arnab Ghoshal

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

#include <vector>

#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "sgmm/am-sgmm.h"
#include "sgmm/fmllr-sgmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Accumulate stats for FMLLR bases training.\n"
        "Usage: sgmm-acc-fmllrbasis-ali [options] <model-in> <feature-rspecifier> "
        "<alignments-rspecifier> <spk2utt-rspecifier> <stats-out>\n"
        "e.g.: sgmm-acc-fmllrbasis-ali 1.mdl scp:train.scp ark:1.ali 1.acc\n";

    ParseOptions po(usage);
    bool binary_write = true;
    std::string gselect_rspecifier, spkvecs_rspecifier, silphones_str;
    BaseFloat sil_weight = 0.0;
    kaldi::SgmmGselectConfig sgmm_opts;
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("gselect", &gselect_rspecifier,
                "Precomputed Gaussian indices (rspecifier)");
    po.Register("spk-vecs", &spkvecs_rspecifier,
                "Speaker vectors to use during aligment (rspecifier)");
    po.Register("sil-phone-list", &silphones_str,
                "Colon-separated list of phones (to weigh differently)");
    po.Register("sil-weight", &sil_weight, "Weight for \"silence\" phones.");
    sgmm_opts.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignments_rspecifier = po.GetArg(3),
        spk2utt_rspecifier = po.GetArg(4),
        accs_wxfilename = po.GetArg(5);

    typedef kaldi::int32 int32;

    AmSgmm am_sgmm;
    TransitionModel trans_model;
    SgmmFmllrGlobalParams fmllr_globals;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
      fmllr_globals.Read(ki.Stream(), binary);
    }

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);

    RandomAccessBaseFloatVectorReader spkvecs_reader(spkvecs_rspecifier);

    std::vector<int32> silence_phones;
    if (!SplitStringToIntegers(silphones_str, ":", false, &silence_phones)) {
      KALDI_ERR << "Silence-phones string has wrong format "
                << silphones_str;
    }
    ConstIntegerSet<int32> silence_set(silence_phones);  // faster lookup.


    kaldi::SgmmPerFrameDerivedVars per_frame_vars;
    SpMatrix<double> fmllr_grad_scatter;
    int32 dim = am_sgmm.FeatureDim();
    fmllr_grad_scatter.Resize(dim * (dim + 1), kSetZero);
    FmllrSgmmAccs spk_stats;
    spk_stats.Init(dim, am_sgmm.NumGauss());

    double tot_like = 0.0, tot_t = 0.0;
    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0;

    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      spk_stats.SetZero();
      string spk = spk2utt_reader.Key();
      const std::vector<string> &uttlist = spk2utt_reader.Value();

      SgmmPerSpkDerivedVars spk_vars;
      if (spkvecs_reader.IsOpen()) {
        if (spkvecs_reader.HasKey(spk)) {
          spk_vars.v_s = spkvecs_reader.Value(spk);
          am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
        } else {
          KALDI_WARN << "Cannot find speaker vector for " << spk;
          num_other_error++;
          continue;
        }
      }  // else spk_vars is "empty"

      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!alignments_reader.HasKey(utt)) {
          num_no_alignment++;
          continue;
        }
        const std::vector<int32> &alignment = alignments_reader.Value(utt);

        if (!feature_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find features for utterance " << utt;
          num_other_error++;
          continue;
        }
        const Matrix<BaseFloat> &feats = feature_reader.Value(utt);

        if (alignment.size() != feats.NumRows()) {
          KALDI_WARN << "Alignments has wrong size "<< (alignment.size()) <<
              " vs. "<< (feats.NumRows());
          num_other_error++;
          continue;
        }

        bool have_gselect = false;
        if (gselect_reader.IsOpen()) {
          if (gselect_reader.HasKey(utt)) {
            have_gselect = (gselect_reader.Value(utt).size() == feats.NumRows());
            if (!have_gselect)
              KALDI_WARN << "Gaussian-selection info available for utterance "
                         << utt << " has wrong size.";
          } else {
            KALDI_WARN << "No Gaussian-selection info available for utterance "
                       << utt;
          }
        }

        const std::vector<std::vector<int32> > *gselect =
            (have_gselect ? &gselect_reader.Value(utt) : NULL);
        double file_like = 0.0, file_t = 0.0;


        for (size_t i = 0; i < alignment.size(); i++) {
          int32 tid = alignment[i];  // transition identifier.
          int32 pdf_id = trans_model.TransitionIdToPdf(tid),
              phone = trans_model.TransitionIdToPhone(tid);
          BaseFloat weight = 1.0;
          if (silence_set.count(phone) != 0) {  // is a silence.
            if (sil_weight > 0.0)
              weight = sil_weight;
            else
              continue;
          }

          std::vector<int32> this_gselect;
          if (gselect != NULL)
            this_gselect = (*gselect)[i];
          else
            am_sgmm.GaussianSelection(sgmm_opts, feats.Row(i), &this_gselect);
          am_sgmm.ComputePerFrameVars(feats.Row(i), this_gselect, spk_vars, 0.0,
                                      &per_frame_vars);
          file_like +=
              spk_stats.Accumulate(am_sgmm, spk_vars, feats.Row(i),
                                   per_frame_vars, pdf_id, weight);
          file_t += weight;
        }  // end looping over all the frames in the utterance
        KALDI_VLOG(1) << "Average likelihood for utterance " << utt << " is "
                      << (file_like/file_t) << " over " << file_t << " frames";
        tot_like += file_like;
        tot_t += file_t;
        num_done++;
        if (num_done % 20 == 0)
          KALDI_VLOG(1) << "After " << num_done << " utterances: Average "
                        << "likelihood per frame = " << (tot_like/tot_t)
                        << ", over " << tot_t << " frames";
      }  // end looping over all utterance for a given speaker
      spk_stats.AccumulateForFmllrSubspace(am_sgmm, fmllr_globals, &fmllr_grad_scatter);
    }  // end looping over all speakers

    KALDI_LOG << "Done " << num_done << " files, " << num_no_alignment
              << " with no alignments, " << num_other_error
              << " with other errors.";

    KALDI_LOG << "Overall likelihood per frame frame = " << (tot_like/tot_t)
              << " over " << tot_t << " frames.";

    {
      Output ko(accs_wxfilename, binary_write);
      fmllr_grad_scatter.Write(ko.Stream(), binary_write);
      KALDI_LOG << "Written accs to: " << accs_wxfilename;
    }
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


