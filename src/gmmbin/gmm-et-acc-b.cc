// gmmbin/gmm-et-acc-b.cc

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

namespace kaldi {
static void ProcessUtterance(const ExponentialTransform &et,
                             const GauPost &gpost,
                             const Matrix<BaseFloat> &xform,
                             const Matrix<BaseFloat> &feats,  // un-transformed feats.
                             const TransitionModel &trans_model,
                             const AmDiagGmm &am_gmm,
                             BaseFloat t,
                             ExponentialTransformAccsB *accs_b) {
  // First work out Ds.
  int32 dim = et.Dim();
  Matrix<BaseFloat> Ds(dim, dim+1);

  et.ComputeDs(xform, t, &Ds);

  for (size_t i = 0; i < gpost.size(); i++) {
    SubVector<BaseFloat> feat(feats, i);
    Vector<BaseFloat> t_data(feat);  // transformed feature.
    ApplyFmllrTransform(xform, &t_data);

    for (size_t j = 0; j < gpost[i].size(); j++) {
      int32 pdf_id = trans_model.TransitionIdToPdf(gpost[i][j].first);
      const DiagGmm  &gmm = am_gmm.GetPdf(pdf_id);
      const Vector<BaseFloat> &posteriors (gpost[i][j].second);
      accs_b->AccumulateFromPosteriors(gmm, t_data, posteriors, Ds);
    }
  }
}

} // end namespace kaldi

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Accumulate statistics for estimating the A matrix of exponential transform, \n"
        " per-utterance (default) or per-speaker for \n"
        " the supplied set of speakers (spk2utt option).\n"
        "Note: the align-model is needed to get GMM posteriors; it's in the unadapted space.\n"
        "Usage: gmm-et-acc-b [options] <align-model> <model> <exponential-transform> <feature-rspecifier> "
        "<posteriors-rspecifier> <transform-rspecifier> <warp-rspecifier> <accs-filename>\n";

    ParseOptions po(usage);
    string spk2utt_rspecifier;
    bool binary = false;
    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to "
                "utterance-list map");
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 7) {
      po.PrintUsage();
      exit(1);
    }

    string model_rxfilename = po.GetArg(1),
        et_rxfilename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        gpost_rspecifier = po.GetArg(4),
        transform_rspecifier = po.GetArg(5),
        warps_rspecifier = po.GetArg(6),
        accs_wxfilename = po.GetArg(7);


    RandomAccessGauPostReader gpost_reader(gpost_rspecifier);
    RandomAccessBaseFloatMatrixReader transform_reader(transform_rspecifier);
    RandomAccessBaseFloatReader warps_reader(warps_rspecifier);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input is(model_rxfilename, &binary);
      trans_model.Read(is.Stream(), binary);
      am_gmm.Read(is.Stream(), binary);
    }

    ExponentialTransform et;
    {
      bool binary;
      Input ki(et_rxfilename, &binary);
      et.Read(ki.Stream(), binary);
    }

    int32 dim = et.Dim();

    ExponentialTransformAccsB accs_b(dim);

    int32 num_done = 0, num_no_gpost = 0, num_other_error = 0;
    if (spk2utt_rspecifier != "") {  // per-speaker adaptation
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        string spk = spk2utt_reader.Key();
        if (!transform_reader.HasKey(spk)) {
          KALDI_WARN << "Could not read transform for speaker " << spk;
          num_other_error++;
        }
        if (!warps_reader.HasKey(spk)) {
          KALDI_WARN << "Could not read warp factor for speaker " << spk;
          num_other_error++;
          continue;
        }
        const Matrix<BaseFloat> &xform(transform_reader.Value(spk));
        BaseFloat t = warps_reader.Value(spk);

        const vector<string> &uttlist = spk2utt_reader.Value();
        for (vector<string>::const_iterator utt_itr = uttlist.begin(),
                 itr_end = uttlist.end(); utt_itr != itr_end; ++utt_itr) {
          if (!feature_reader.HasKey(*utt_itr)) {
            KALDI_WARN << "Did not find features for utterance " << *utt_itr;
            continue;
          }
          if (!gpost_reader.HasKey(*utt_itr)) {
            KALDI_WARN << "Did not find gpost for utterance "
                       << *utt_itr;
            num_no_gpost++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(*utt_itr);

          const GauPost &gpost = gpost_reader.Value(*utt_itr);

          if (static_cast<int32>(gpost.size()) != feats.NumRows()) {
            KALDI_WARN << "gpost has wrong size " << gpost.size()
                       << " vs. " << feats.NumRows();
            num_other_error++;
            continue;
          }

          ProcessUtterance(et, gpost, xform, feats, trans_model,
                           am_gmm, t, &accs_b);
          num_done++;
          if (num_done % 50 == 0)
            KALDI_VLOG(1) << "Done " << num_done << " utterances.";
        }  // end looping over all utterances of the current speaker
      }  // end looping over speakers
    } else {  // per-utterance adaptation
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        string utt = feature_reader.Key();
        FmllrDiagGmmAccs accs(dim);

        if (!transform_reader.HasKey(utt)) {
          KALDI_WARN << "Could not read transform for speaker " << utt;
          num_other_error++;
        }
        if (!warps_reader.HasKey(utt)) {
          KALDI_WARN << "Could not read warp factor for speaker " << utt;
          num_other_error++;
          continue;
        }
        if (!gpost_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find gpost for utterance "
                     << utt;
          num_no_gpost++;
          continue;
        }
        const Matrix<BaseFloat> &feats = feature_reader.Value();
        const GauPost &gpost = gpost_reader.Value(utt);
        const Matrix<BaseFloat> &xform(transform_reader.Value(utt));
        BaseFloat t = warps_reader.Value(utt);

        if (static_cast<int32>(gpost.size()) != feats.NumRows()) {
          KALDI_WARN << "gpost has wrong size " << gpost.size()
                     << " vs. " << feats.NumRows();
          num_other_error++;
          continue;
        }

        ProcessUtterance(et, gpost, xform, feats, trans_model,
                         am_gmm, t, &accs_b);
        num_done++;

        if (num_done % 50 == 0)
          KALDI_LOG << "Done " << num_done << " utterances";
      }
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_gpost
              << " with no gposts, " << num_other_error << " with other errors.";

    Output ko(accs_wxfilename, binary);
    accs_b.Write(ko.Stream(), binary);
    KALDI_LOG << "Written accs.";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

