// gmmbin/fmpe-gmm-acc-stats-gpost.cc

// Copyright 2009-2011  Yanmin Qian

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
#include "gmm/diag-gmm.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "gmm/fmpe-am-diag-gmm.h"




int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Accumulate positive and negative stats for Fmpe training (reading in gaussian-level posteriors).\n"
        "Note: not yet tested.\n"
        "Usage:  fmpe-gmm-acc-stats-gpost [options] <model-in> <model-diffs-in> <gmms-model-in> <feature-rspecifier> <gposteriors-ebw-rspecifier> <gposteriors-mle-rspecifier> <stats-out>\n"
        "e.g.: \n"
        " fmpe-gmm-acc-stats-gpost 1.mdl 1.model.diffs 1.gmm scp:train.scp ark:1.ebw.gpost ark:1.mle.gpost 1.fmpe.acc\n";

    typedef kaldi::int32 int32;

    bool binary = false;
    FmpeConfig fmpe_opts;
    int32 gmm_cluster_centers_nbest = 25;
    int32 gmm_gaussian_nbest = 2;
    double lat_prob_scale = 0.083;
    double E = 10.0;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("gmm-cluster-centers-nbest", &gmm_cluster_centers_nbest,
        "Number of highest-scoring of the best cluster centers.");
    po.Register("gmm-gaussian-nbest", &gmm_gaussian_nbest, "Number of"
        " of highest-scoring of the best gaussians.");
    po.Register("lat-prob-scale", &lat_prob_scale,
        "The lattice probability scale, very important.");
    po.Register("E", &E, "The constant that contrals the overall learning rate.");

    fmpe_opts.Register(&po);

    po.Read(argc, argv);


    if (po.NumArgs() != 7) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        model_diffs_filename = po.GetArg(2),
        gmms_model_filename = po.GetArg(3),
        feature_rspecifier = po.GetArg(4),
        gposteriors_ebw_rspecifier = po.GetArg(5),
        gposteriors_mle_rspecifier = po.GetArg(6),
        accs_wxfilename = po.GetArg(7);

    using namespace kaldi;

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    FmpeAccs fmpe_accs(fmpe_opts);
    fmpe_accs.Init(am_gmm, true);
    {
      bool binary;
      Input ki(model_diffs_filename, &binary);
      fmpe_accs.ReadModelDiffs(ki.Stream(), binary);
    }

    kaldi::DiagGmm gmm;
    kaldi::DiagGmm gmm_clusters;
    std::vector<int32> gaussian_cluster_center_map;
    {
      bool binary;
      Input ki(gmms_model_filename, &binary);
      gmm.Read(ki.Stream(), binary);
      gmm_clusters.Read(ki.Stream(), binary);
      ReadIntegerVector(ki.Stream(), binary, &gaussian_cluster_center_map);
    }

    fmpe_accs.InitializeGMMs(gmm, gmm_clusters, gaussian_cluster_center_map);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessGauPostReader gposteriors_ebw_reader(gposteriors_ebw_rspecifier);
    RandomAccessGauPostReader gposteriors_mle_reader(gposteriors_mle_rspecifier);

    int32 num_done = 0, num_no_posterior = 0, num_other_error = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if ((!gposteriors_ebw_reader.HasKey(key)) &&
		  (!gposteriors_mle_reader.HasKey(key))) {
        num_no_posterior++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const GauPost &gpost_ebw = gposteriors_ebw_reader.Value(key);
        const GauPost &gpost_mle = gposteriors_ebw_reader.Value(key);

        if ((static_cast<int32>(gpost_ebw.size()) != mat.NumRows()) &&
			(static_cast<int32>(gpost_mle.size()) != mat.NumRows())) {
          KALDI_WARN << "Gaussian Posterior vector has wrong size : gpost-ebw. " <<
			  (gpost_ebw.size()) << "gpost-mle. " << (gpost_mle.size()) << " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        num_done++;

        std::vector<std::vector<std::pair<int32, Vector<double> > > > whole_file_offset;
        std::vector<std::pair<int32, std::vector<std::pair<int32, Vector<double> > > > > ht;

        fmpe_accs.ComputeWholeFileOffsetFeature(mat, &whole_file_offset);

        for (size_t i = 0; i < mat.NumRows(); i++) {
          fmpe_accs.ComputeHighDimemsionFeature(whole_file_offset, i, &ht);
          Vector<double> direct_diff(mat.NumCols()), indirect_diff(mat.NumCols());
		  /// compute the direct differentials
          for (size_t j = 0; j < gpost_ebw[i].size(); j++) {
            int32 tid = gpost_ebw[i][j].first,  // transition identifier.
                pdf_id = trans_model.TransitionIdToPdf(tid);
            fmpe_accs.AccumulateDirectDiffFromPosteriors(am_gmm.GetPdf(pdf_id),
														 mat.Row(i),
														 gpost_ebw[i][j].second,
														 &direct_diff);
          }
		  /// compute the indirect differentials
          for (size_t j = 0; j < gpost_mle[i].size(); j++) {
            int32 tid = gpost_mle[i][j].first,  // transition identifier.
                pdf_id = trans_model.TransitionIdToPdf(tid);
            fmpe_accs.AccumulateInDirectDiffFromPosteriors(am_gmm.GetPdf(pdf_id),
														   fmpe_accs.GetAccsModelDiff(pdf_id),
														   mat.Row(i),
														   gpost_mle[i][j].second,
														   &indirect_diff);
          }
          fmpe_accs.AccumulateFromDifferential(direct_diff, indirect_diff, ht);
          ht.clear();
        }
        if (num_done % 50 == 0) {
          KALDI_LOG << "Processed " << num_done << " utterances.";
        }
      }
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_posterior
              << " with no posteriors, " << num_other_error
              << " with other errors.";

    {
      Output ko(accs_wxfilename, binary);
      fmpe_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


