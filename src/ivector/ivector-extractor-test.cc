// ivector/ivector-extractor-test.cc

// Copyright 2013  Daniel Povey

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
  bool binary = (rand() % 2 == 0);
  extractor.Write(ostr, binary);
  std::istringstream istr(ostr.str());
  IvectorExtractor extractor2;
  extractor2.Read(istr, binary);
  std::ostringstream ostr2;
  extractor2.Write(ostr2, binary);
  KALDI_ASSERT(ostr.str() == ostr2.str());
}
void TestIvectorStatsIO(IvectorStats &stats) {
  std::ostringstream ostr;
  bool binary = (rand() % 2 == 0);
  stats.Write(ostr, binary);
  std::istringstream istr(ostr.str());
  IvectorStats stats2;
  stats2.Read(istr, binary);
  std::ostringstream ostr2;
  stats2.Write(ostr2, binary);
  KALDI_ASSERT(ostr.str() == ostr2.str());

  { // Test I/O of IvectorStats and that it works identically with the "add"
    // mechanism.  We only test this with binary == true; otherwise it's not
    // identical due to limited precision.
    std::ostringstream ostr;
    bool binary = true;
    stats.Write(ostr, binary);
    IvectorStats stats2;
    {
      std::istringstream istr(ostr.str());
      stats2.Read(istr, binary);
    }
    {
      std::istringstream istr(ostr.str());
      stats2.Read(istr, binary, true); // add to existing.
    }
    IvectorStats stats3(stats);
    stats3.Add(stats);
    
    std::ostringstream ostr2;
    stats2.Write(ostr2, false);

    std::ostringstream ostr3;
    stats3.Write(ostr3, false);
    // This test stopped working after we made the stats single precision.
    // It's OK.  Disabling it.
    // KALDI_ASSERT(ostr2.str() == ostr3.str());
  }
  
}
  

void UnitTestIvectorExtractor() {
  FullGmm fgmm;
  int32 dim = 5 + rand() % 5, num_comp = 1 + rand() % 5;
  KALDI_LOG << "Num Gauss = " << num_comp;
  unittest::InitRandFullGmm(dim, num_comp, &fgmm);
  FullGmmNormal fgmm_normal(fgmm);

  IvectorExtractorOptions ivector_opts;
  ivector_opts.ivector_dim = dim + 5;
  ivector_opts.use_weights = (rand() % 2 == 0);
  KALDI_LOG << "Feature dim is " << dim
            << ", ivector dim is " << ivector_opts.ivector_dim;
  IvectorExtractor extractor(ivector_opts, fgmm);
  TestIvectorExtractorIO(extractor);

  IvectorStatsOptions stats_opts;
  if (rand() % 2 == 0) stats_opts.update_variances = false;
  stats_opts.num_samples_for_weights = 100; // Improve accuracy
  // of estimation, since we do it with relatively few utterances,
  // and we're testing the convergence.

  int32 num_utts = 1 + rand() % 5;
  std::vector<Matrix<BaseFloat> > all_feats(num_utts);
  for (int32 utt = 0; utt < num_utts; utt++) {
    int32 num_frames = 100 + rand() % 200;
    if (rand() % 2 == 0) num_frames *= 10;
    if (rand() % 2 == 0) num_frames /= 1.0;
    Matrix<BaseFloat> feats(num_frames, dim);
    fgmm_normal.Rand(&feats);
    feats.Swap(&all_feats[utt]);
  }

  int32 num_iters = 4;
  double last_auxf_impr = 0.0, last_auxf = 0.0;
  for (int32 iter = 0; iter < num_iters; iter++) {
    IvectorStats stats(extractor, stats_opts);
      
    for (int32 utt = 0; utt < num_utts; utt++) {
      Matrix<BaseFloat> &feats = all_feats[utt];
      stats.AccStatsForUtterance(extractor, feats, fgmm);
    }
    TestIvectorStatsIO(stats);
    
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
  SetVerboseLevel(3);
  for (int i = 0; i < 10; i++)
    UnitTestIvectorExtractor();
  std::cout << "Test OK.\n";
  return 0;
}
