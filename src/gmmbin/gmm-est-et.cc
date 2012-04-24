// gmmbin/gmm-est-et.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include <string>
using std::string;
#include <vector>
using std::vector;

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "transform/exponential-transform.h"

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Compute exponential transforms (which are a special case of fMLLR transforms)\n"
        " per-utterance (default) or per-speaker for \n"
        " the supplied set of speakers (spk2utt option).\n"
        "Usage: gmm-est-et [options] <model> <exponential-transform> <feature-rspecifier> "
        "<gpost-rspecifier> <transforms-wspecifier> [<warp-factors-wspecifier>]\n";

    ParseOptions po(usage);
    string spk2utt_rspecifier;
    bool binary = true;
    std::string normalize_type = "";
    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to "
                "utterance-list map");
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("normalize-type", &normalize_type, "Change normalization type: \"\"|\"offset\"|\"diag\"|\"none\"");
    po.Read(argc, argv);

    if (po.NumArgs() < 5 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    string model_rxfilename = po.GetArg(1),
        et_rxfilename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        gpost_rspecifier = po.GetArg(4),
        xforms_wspecifier = po.GetArg(5),
        warps_wspecifier = po.GetOptArg(6);

    RandomAccessGauPostReader gpost_reader(gpost_rspecifier);
    BaseFloatMatrixWriter xform_writer(xforms_wspecifier);
    BaseFloatWriter warps_writer(warps_wspecifier);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    ExponentialTransform et;
    ReadKaldiObject(et_rxfilename, &et);

    if (normalize_type != "") {
      EtNormalizeType nt = kEtNormalizeNone;
      if (normalize_type == "offset") nt = kEtNormalizeOffset;
      else if (normalize_type == "diag") nt = kEtNormalizeDiag;
      else if (normalize_type == "none") nt = kEtNormalizeNone;
      // "none" unlikely, since pointless: only allowed if already == none.
      else KALDI_ERR << "Invalid normalize-type option: " << normalize_type;
      // The next statement may fail if you tried to reduce
      // the amount of normalization.
      et.SetNormalizeType(nt);
    }

    int32 dim = et.Dim();
    double tot_objf_impr = 0.0,
        tot_count = 0.0;

    int32 num_done = 0, num_no_gpost = 0, num_other_error = 0;

    if (spk2utt_rspecifier != "") {  // per-speaker adaptation
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        string spk = spk2utt_reader.Key();
        FmllrDiagGmmAccs accs(dim);

        const vector<string> &uttlist = spk2utt_reader.Value();
        for (vector<string>::const_iterator utt_itr = uttlist.begin(),
            itr_end = uttlist.end(); utt_itr != itr_end; ++utt_itr) {
          if (!feature_reader.HasKey(*utt_itr)) {
            KALDI_WARN << "Did not find features for utterance " << *utt_itr;
            continue;
          }
          if (!gpost_reader.HasKey(*utt_itr)) {
            KALDI_WARN << "Did not find Gaussian posteriors for utterance "
                       << *utt_itr;
            num_no_gpost++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(*utt_itr);

          const GauPost &gpost = gpost_reader.Value(*utt_itr);
          if (static_cast<int32>(gpost.size()) != feats.NumRows()) {
            KALDI_WARN << "gpost has wrong size " << (gpost.size())
                       << " vs. " << (feats.NumRows());
            num_other_error++;
            continue;
          }

          for (size_t i = 0; i < gpost.size(); i++) {
            const SubVector<BaseFloat> feat(feats, i);
            for (size_t j = 0; j < gpost[i].size(); j++) {
              int32 pdf_id = trans_model.TransitionIdToPdf(gpost[i][j].first);
              const Vector<BaseFloat> &posteriors(gpost[i][j].second);
              const DiagGmm &gmm = am_gmm.GetPdf(pdf_id);
              KALDI_ASSERT(gmm.NumGauss() == posteriors.Dim());
              accs.AccumulateFromPosteriors(gmm, feat, posteriors);
            }
          }
          num_done++;
        }  // end looping over all utterances of the current speaker

        Matrix<BaseFloat> xform(dim, dim+1);
        BaseFloat objf_impr, spk_count, t;  // t is the "t" variable in the
        // exponential transform, not time.
        et.ComputeTransform(accs, &xform, &t, NULL, &objf_impr, &spk_count);
        tot_objf_impr += objf_impr;
        tot_count += spk_count;
        KALDI_LOG << "Objf impr for speaker " << spk << " is " << (objf_impr/spk_count)
                  << " over " << spk_count << " frames.";
        xform_writer.Write(spk, xform);

        if (warps_wspecifier != "") warps_writer.Write(spk, t);
      }  // end looping over speakers
    } else {  // per-utterance adaptation
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        string utt = feature_reader.Key();
        FmllrDiagGmmAccs accs(dim);

        if (!gpost_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find aligned transcription for utterance "
                     << utt;
          num_no_gpost++;
          continue;
        }
        const Matrix<BaseFloat> &feats = feature_reader.Value();
        const GauPost &gpost = gpost_reader.Value(utt);

        if (static_cast<int32>(gpost.size()) != feats.NumRows()) {
          KALDI_WARN << "Gpost has wrong size " << (gpost.size())
              << " vs. " << (feats.NumRows());
          num_other_error++;
          continue;
        }

        num_done++;
        for (size_t i = 0; i < gpost.size(); i++) {
          const SubVector<BaseFloat> feat(feats, i);
          for (size_t j = 0; j < gpost[i].size(); j++) {
            int32 pdf_id = trans_model.TransitionIdToPdf(gpost[i][j].first);
            const Vector<BaseFloat> &posteriors(gpost[i][j].second);
            const DiagGmm &gmm = am_gmm.GetPdf(pdf_id);
            KALDI_ASSERT(gmm.NumGauss() == posteriors.Dim());
            accs.AccumulateFromPosteriors(gmm, feat, posteriors);
          }
        }
        num_done++;
        Matrix<BaseFloat> xform(dim, dim+1);
        BaseFloat objf_impr, utt_count, t;  // t is the "t" variable in the
        // exponential transform, not time.
        et.ComputeTransform(accs, &xform, &t, NULL, &objf_impr, &utt_count);
        tot_objf_impr += objf_impr;
        tot_count += utt_count;
        KALDI_LOG << "Objf impr for utterance " << utt << " is " << (objf_impr/utt_count)
                  << " over " << utt_count << " frames.";
        xform_writer.Write(utt, xform);
        if (warps_wspecifier != "") warps_writer.Write(utt, t);
      }
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_gpost
              << " with no posteriors, " << num_other_error
              << " with other errors.";
    KALDI_LOG << "Overall objf impr per frame = "
              << (tot_objf_impr / tot_count) << " over " << tot_count
              << " frames.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

