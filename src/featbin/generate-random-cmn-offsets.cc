// featbin/generate-random-cmn-offsets.cc

// Copyright 2016 Pegah Ghahremani

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
#include "matrix/kaldi-matrix.h"
#include "transform/cmvn.h"

namespace kaldi {
/*
  This function generates random offset for each speaker using zero mean and covariance sigma^2.
  as y = sigma * x  , x = N(1, 0) => E(y) = 0, Var(y) = sigma^2.
  Rows of random_offsets are random offsets and number of rows is equal to number of
  random offsets generated.
*/
void GenerateRandomOffset(const Matrix<double> &sigma,
                          MatrixBase<double> *random_offsets) {
  KALDI_ASSERT(sigma.NumCols() == sigma.NumRows() &&
               sigma.NumRows() == random_offsets->NumCols());
  Matrix<double> rand_val(random_offsets->NumRows(), random_offsets->NumCols());
  rand_val.SetRandn();
  // y = sigma * x
  random_offsets->AddMatMat(1.0, rand_val, kNoTrans, sigma, kNoTrans, 0.0);
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Generates random cmn offset using speakers' statistics. \n"
        " It first computes between-speaker's covariance for the feature-space CMN offsets, \n"
        "by taking the centered covariance of the speaker or pseudo-speaker means \n"
        " using speakers in cmvn.scp and use this times some constant to \n"
        " generate random CMN offsets for speakers in spk2utt list to be applied to the features.\n"
        "The output matrices are of dimension --num-cmn-offsets by [feat-dim]."
        "Usage: generate-random-cmn-offsets [options] <spk2utt> <cmvn-archive-in> <cmn-offset-archive-out>\n"
        "e.g.: generate-random-cmn-offsets --srand=0 --num-cmn-offsets=5 --cmn-offset-scale=0.5 \n"
        " ark:data/train/spk2utt scp:data/train/cmvn.scp \n"
        " ark,scp:cmn_offset_per_spk.ark,cmn_offset_per_spk.scp \n";
    ParseOptions po(usage);

    int32 num_cmn_offsets = 4,
      srand_seed = 0;
    double cmn_offset_scale = 0.5;
    po.Register("num-cmn-offsets", &num_cmn_offsets, "number of random cmn offsets generated for each spk");
    po.Register("srand", &srand_seed, "Seed for random number generator used to generate random offset per spk.");
    po.Register("cmn-offset-scale", &cmn_offset_scale, "offset scale used to scale the covariance matrix.");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string spk2utt_rspecifier = po.GetArg(1),
    cmvn_rspecifier = po.GetArg(2),
      cmn_offset_wspecifier = po.GetArg(3);

    srand(srand_seed);

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    SequentialDoubleMatrixReader cmvn_reader(cmvn_rspecifier);
    DoubleMatrixWriter cmn_offset_writer(cmn_offset_wspecifier);

    int32 num_spks = 0,
      feat_dim = -1;
    // spk_cov is the centered covariance of the speaker means
    // as sum_{i=0}^n (m_i - g)^T (m_i - g), where m_i is spk mean for
    // speaker i and g is the gloabl mean.
    // add sum_{i=0}^n m_i^T * m_i
    Matrix<double> spk_cov;
    Vector<double> spk_mean_sum;
    for (; !cmvn_reader.Done(); cmvn_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      const Matrix<double> &cmvn_stats = cmvn_reader.Value();
      if (feat_dim < 0) {
        feat_dim = cmvn_stats.NumCols() - 1;
        spk_cov.Resize(feat_dim, feat_dim);
        spk_mean_sum.Resize(feat_dim);
      }
      Vector<double> spk_mean(feat_dim);
      num_spks++;
      spk_mean = cmvn_stats.Row(0).Range(0, feat_dim);
      double count = cmvn_stats(0, feat_dim);
      spk_mean.Scale(1.0/count);
      spk_mean_sum.AddVec(1.0, spk_mean);
      spk_cov.AddVecVec(1.0, spk_mean, spk_mean);
    }
    // spk_means_mean = 1/num_spks (sum_{i=0}^num_spks m_i)
    Vector<double> spk_means_mean(spk_mean_sum);
    spk_means_mean.Scale(1.0 / num_spks);
    // add term -2 * (sum_{i=0}^n m_i^T) g
    spk_cov.AddVecVec(-2.0, spk_mean_sum, spk_means_mean);
    // add term num_spks * gT * g
    spk_cov.AddVecVec(num_spks, spk_means_mean, spk_means_mean);
    spk_cov.Scale(1.0 / num_spks);
    // compute sigma as spk_cov^0.5 for matrix w.r.t its svd decomposition.
    Matrix<double> sigma(spk_cov);
    sigma.Power(0.5);

    // Generate random cmn offset for each speaker, and copy same cmn offsets
    // for all spk's utterances.
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      Matrix<double> spk_offsets(num_cmn_offsets, feat_dim);
      std::string spk = spk2utt_reader.Key();
      GenerateRandomOffset(sigma, &spk_offsets);
      spk_offsets.Scale(cmn_offset_scale);
      // the 1st row of offset is 0.
      spk_offsets.Row(0).Set(0.0);
      cmn_offset_writer.Write(spk, spk_offsets);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
