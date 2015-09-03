// ivector/ivector-extractor-test.cc

// Copyright 2013  Daniel Povey

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

#include "gmm/model-test-common.h"
#include "gmm/full-gmm-normal.h"
#include "ivector/ivector-extractor.h"
#include "util/kaldi-io.h"


namespace kaldi {

void TestIvectorExtractorIO(const IvectorExtractor &extractor) {
  std::ostringstream ostr;
  bool binary = (Rand() % 2 == 0);
  extractor.Write(ostr, binary);
  std::istringstream istr(ostr.str());
  IvectorExtractor extractor2;
  extractor2.Read(istr, binary);
  std::ostringstream ostr2;
  extractor2.Write(ostr2, binary);
  KALDI_ASSERT(ostr.str() == ostr2.str());
}
void TestIvectorExtractorStatsIO(IvectorExtractorStats &stats) {
  std::ostringstream ostr;
  bool binary = (Rand() % 2 == 0);
  stats.Write(ostr, binary);
  std::istringstream istr(ostr.str());
  IvectorExtractorStats stats2;
  stats2.Read(istr, binary);
  std::ostringstream ostr2;
  stats2.Write(ostr2, binary);
  
  if (binary) {
    // this was failing in text mode, due to differences like
    // 8.2244e+06 vs  8.22440e+06
    KALDI_ASSERT(ostr.str() == ostr2.str());
  }
  
  { // Test I/O of IvectorExtractorStats and that it works identically with the "add"
    // mechanism.  We only test this with binary == true; otherwise it's not
    // identical due to limited precision.
    std::ostringstream ostr;
    bool binary = true;
    stats.Write(ostr, binary);
    IvectorExtractorStats stats2;
    {
      std::istringstream istr(ostr.str());
      stats2.Read(istr, binary);
    }
    {
      std::istringstream istr(ostr.str());
      stats2.Read(istr, binary, true); // add to existing.
    }
    IvectorExtractorStats stats3(stats);
    stats3.Add(stats);
    
    std::ostringstream ostr2;
    stats2.Write(ostr2, false);

    std::ostringstream ostr3;
    stats3.Write(ostr3, false);

    //if (binary) {
    //  KALDI_ASSERT(ostr2.str() == ostr3.str());
    //}
  }
}

void TestIvectorExtraction(const IvectorExtractor &extractor,
                           const MatrixBase<BaseFloat> &feats,
                           const FullGmm &fgmm) {
  if (extractor.IvectorDependentWeights())
    return;  // Nothing to do as online iVector estimator does not work in this
             // case.
  int32 num_frames = feats.NumRows(),
      feat_dim = feats.NumCols(),
      num_gauss = extractor.NumGauss(),
      ivector_dim = extractor.IvectorDim();
  Posterior post(num_frames);

  double tot_log_like = 0.0;
  for (int32 t = 0; t < num_frames; t++) {
    SubVector<BaseFloat> frame(feats, t);
    Vector<BaseFloat> posterior(fgmm.NumGauss(), kUndefined);
    tot_log_like += fgmm.ComponentPosteriors(frame, &posterior);
    for (int32 i = 0; i < posterior.Dim(); i++)
      post[t].push_back(std::make_pair(i, posterior(i)));
  }
    
  // The zeroth and 1st-order stats are in "utt_stats".
  IvectorExtractorUtteranceStats utt_stats(num_gauss, feat_dim,
                                           false);
  utt_stats.AccStats(feats, post);

  OnlineIvectorEstimationStats online_stats(extractor.IvectorDim(),
                                            extractor.PriorOffset(),
                                            0.0);
  
  for (int32 t = 0; t < num_frames; t++) {
    online_stats.AccStats(extractor, feats.Row(t), post[t]);
  }
  
  Vector<double> ivector1(ivector_dim), ivector2(ivector_dim);

  extractor.GetIvectorDistribution(utt_stats, &ivector1, NULL);

  int32 num_cg_iters = -1;  // for testing purposes, compute it exactly.
  online_stats.GetIvector(num_cg_iters, &ivector2);

  KALDI_LOG << "ivector1 = " << ivector1;
  KALDI_LOG << "ivector2 = " << ivector2;

  // objf change vs. default iVector.  note, here I'm using objf
  // and auxf pretty much interchangeably :-(
  double objf_change2 = online_stats.ObjfChange(ivector2) *
      utt_stats.NumFrames();

  Vector<double> ivector_baseline(ivector_dim);
  ivector_baseline(0) = extractor.PriorOffset();
  double objf_change1 = extractor.GetAuxf(utt_stats, ivector1) -
      extractor.GetAuxf(utt_stats, ivector_baseline);
  KALDI_LOG << "objf_change1 = " << objf_change1
            << ", objf_change2 = " << objf_change2;
  
  KALDI_ASSERT(ivector1.ApproxEqual(ivector2));
}


void UnitTestIvectorExtractor() {
  FullGmm fgmm;
  int32 dim = 5 + Rand() % 5, num_comp = 1 + Rand() % 5;
  KALDI_LOG << "Num Gauss = " << num_comp;
  unittest::InitRandFullGmm(dim, num_comp, &fgmm);
  FullGmmNormal fgmm_normal(fgmm);

  IvectorExtractorOptions ivector_opts;
  ivector_opts.ivector_dim = dim + 5;
  ivector_opts.use_weights = (Rand() % 2 == 0);
  KALDI_LOG << "Feature dim is " << dim
            << ", ivector dim is " << ivector_opts.ivector_dim;
  IvectorExtractor extractor(ivector_opts, fgmm);
  TestIvectorExtractorIO(extractor);

  IvectorExtractorStatsOptions stats_opts;
  if (Rand() % 2 == 0) stats_opts.update_variances = false;
  stats_opts.num_samples_for_weights = 100; // Improve accuracy
  // of estimation, since we do it with relatively few utterances,
  // and we're testing the convergence.

  int32 num_utts = 1 + Rand() % 5;
  std::vector<Matrix<BaseFloat> > all_feats(num_utts);
  for (int32 utt = 0; utt < num_utts; utt++) {
    int32 num_frames = 100 + Rand() % 200;
    if (Rand() % 2 == 0) num_frames *= 10;
    if (Rand() % 2 == 0) num_frames /= 1.0;
    Matrix<BaseFloat> feats(num_frames, dim);
    fgmm_normal.Rand(&feats);
    feats.Swap(&all_feats[utt]);
  }

  int32 num_iters = 4;
  double last_auxf_impr = 0.0, last_auxf = 0.0;
  for (int32 iter = 0; iter < num_iters; iter++) {
    IvectorExtractorStats stats(extractor, stats_opts);
      
    for (int32 utt = 0; utt < num_utts; utt++) {
      Matrix<BaseFloat> &feats = all_feats[utt];
      stats.AccStatsForUtterance(extractor, feats, fgmm);
      TestIvectorExtraction(extractor, feats, fgmm);
    }
    TestIvectorExtractorStatsIO(stats);
    
    IvectorExtractorEstimationOptions estimation_opts;
    estimation_opts.gaussian_min_count = dim + 5;
    double auxf = stats.AuxfPerFrame(),
        auxf_impr = stats.Update(estimation_opts, &extractor);

    KALDI_LOG << "Iter " << iter << ", auxf per frame was " << auxf
              << ", improvement in this update "
              << "phase was " << auxf_impr;
    if (iter > 0) {
      double auxf_change = auxf - last_auxf;
      KALDI_LOG << "Predicted auxf change from last update phase was "
                << last_auxf_impr << " versus observed change "
                << auxf_change;
      double wiggle_room = (ivector_opts.use_weights ? 5.0e-05 : 1.0e-08);
      // The weight update is (a) not exact, and (b) relies on sampling, [two
      // separate issues], so it might not always improve.  But with
      // a large number of "weight samples", it's OK.
      KALDI_ASSERT(auxf_change >= last_auxf_impr - wiggle_room);
    }
    last_auxf_impr = auxf_impr;
    last_auxf = auxf;
  }
  std::cout << "********************************************************************************************\n";
}

}

int main() {
  using namespace kaldi;
  SetVerboseLevel(5);
  for (int i = 0; i < 10; i++)
    UnitTestIvectorExtractor();
  std::cout << "Test OK.\n";
  return 0;
}
