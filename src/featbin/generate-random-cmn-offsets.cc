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
  This function generates random offset for each spk w.r.t speaker mean and variance sigma^2.
  as y = sigma * x + mean , x = N(1, 0) => E(y) = mean, Var(y) = sigma^2.
  Rows of random_offsets are random offsets and number of rows is equal to number of
  random offsets generated.
*/
void GenerateRandomOffset(const Vector<double> &mean,
                          const Matrix<double> &sigma,
                          MatrixBase<double> *random_offsets) {
  KALDI_ASSERT(mean.Dim() == sigma.NumRows());
  KALDI_ASSERT(sigma.NumCols() == sigma.NumRows() &&
               sigma.NumRows() == random_offsets->NumCols());
  Matrix<double> rand_val(random_offsets->NumRows(), random_offsets->NumCols());
  rand_val.SetRandn();
  // y = sigma * x + mean
  random_offsets->AddMatMat(1.0, rand_val, kNoTrans, sigma, kNoTrans, 0.0);
  random_offsets->AddVecToRows(1.0, mean);
}

/*
 The factor 1.0 / sqrt(1.0 + cmn_offset_scale * cmn_offset_scale)
 ensures that by adding these offsets, we don't change the total covariance of
 the speaker means after adding the offsets.
 Note that if we replace the sqrt factor with 1.0 below, it reduces the case where
 preserve_total_covariance is false.
 offset_from_global_mean = (o * cmn_offset_scale + (m-g)) / sqrt(cmn_offset_scal^2+1)
 offset = offset_from_global_mean - (m-g), where m is speaker mean and g is global mean.
*/
void PreserveTotalCovariance(double cmn_offset_scale,
                             const Vector<double> &centered_mean,
                             Matrix<double> *random_offsets) {
  double scale1 = cmn_offset_scale / pow(1 + cmn_offset_scale * cmn_offset_scale , 0.5),
    scale2 = (1.0 / pow(1.0 + cmn_offset_scale * cmn_offset_scale , 0.5) - 1.0);
  random_offsets->Scale(scale1);
  random_offsets->AddVecToRows(scale2, centered_mean);
}

}
int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Generates random cmn offset using speakers' stats. \n"
        " It first computes covariance for the feature-space CMN offsets, \n"
        "by taking the centered covariance of the speaker or pseudo-speaker means \n"
        " and use this times some constant to generate random CMN offsets to be applied \n"
        " to the features. \n"
        "Usage: generate-random-cmn-offsets [options] <cmn-archive-in> <cmn-offset-archive-out>\n"
        "e.g.: generate-random-cmn-offsets --srand=0 --num-cmn-offsets=5 --cmn-offset-scale=0.5 \n"
        " ark:data/train/spk2utt scp:data/train/feats.scp ark,scp:cmn_offset_per_spk.ark,cmn_offset_per_spk.scp \n";
    ParseOptions po(usage);
    
    bool  preserve_total_covariance = false, zero_mean = false;

    int32 num_cmn_offsets = 5, 
      srand_seed = 0;
    double cmn_offset_scale = 1.0;

    std::string weights_rspecifier;

    po.Register("num-cmn-offsets", &num_cmn_offsets, "number of random cmn offsets generated for each spk");
    po.Register("srand", &srand_seed, "Seed for random number generator used to generate random offset per spk.");
    po.Register("cmn-offset-scale", &cmn_offset_scale, "offset scale used to scale the covariance matrix.");
    po.Register("preserve-total-covariance", &preserve_total_covariance, "If true, the total covariance for random"
                " offset of spks are preserved.");
    po.Register("weights", &weights_rspecifier, "rspecifier for a vector of floats "
                "for each utterance, that's a per-frame weight.");
    po.Register("zero-mean-offsets", &zero_mean, 
                "If true, the random offsets generated using zero mean distribution."); 
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

   
    std::string spk2utt_rspecifier = po.GetArg(1),
      feat_rspecifier = po.GetArg(2),
      cmn_wspecifier = po.GetArg(3);
    DoubleMatrixWriter cmn_writer(cmn_wspecifier); 
   
    srand(srand_seed);
    int32 num_done = 0, num_err = 0, feat_dim = 0;

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessBaseFloatMatrixReader feat_reader(feat_rspecifier); 

    RandomAccessBaseFloatVectorReader weights_reader(weights_rspecifier);

    Matrix<double> global_stats,
      spk_cov;
    bool is_global_init = false;
    int32 num_spks = 0;
    Vector<double> spk_mean_sum;
    std::map<std::string, Vector<double> > spk_means;
    std::map<std::string, std::vector<std::string> > spk_uttlist;
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key(); 
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();
      spk_uttlist[spk] = uttlist;
      bool is_init = false;
      Matrix<double> stats;
      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!feat_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find features for utterance " << utt;
            num_err++;
            continue;
        }
        const Matrix<BaseFloat> &feats = feat_reader.Value(utt);
        // Initialize local stat to accumulate stats per speaker
        if (!is_init) {
          InitCmvnStats(feats.NumCols(), &stats);
          feat_dim = feats.NumCols();
          is_init = true;
        }

       
        if (!AccCmvnStatsWrapper(utt, feats, &weights_reader, &stats)) 
          num_err++;
        else
          num_done++;
      }
      Vector<double> spk_mean(stats.Row(0).Range(0, feat_dim));
      double count = stats(0, feat_dim);
      spk_mean.Scale(1.0/count);
      spk_means[spk] = spk_mean;
      // Initialize global stats to accumulate stats for all data.
      if (!is_global_init) {
        InitCmvnStats(feat_dim, &global_stats);
        spk_cov.Resize(feat_dim, feat_dim);
        spk_mean_sum.Resize(feat_dim);
        is_global_init = true;
      }
      
      // spk_cov is the centered covariance of the speaker means
      // as sum_{i=0}^n (m_i - g)^T (m_i - g), where m_i is spk mean for 
      // speaker i and g is the gloabl mean.
      // add sum_{i=0}^n m_i^T * m_i
      num_spks++;
      spk_cov.AddVecVec(1.0, spk_mean, spk_mean);
      spk_mean_sum.AddVec(1.0, spk_mean);
      global_stats.AddMat(1.0, stats);
    }
    // global_mean (g) is the global mean for all frames.
    Vector<double> global_mean(global_stats.Row(0).Range(0, feat_dim));

    double global_count = global_stats(0, feat_dim);
    global_mean.Scale(1.0 / global_count);

    Vector<double> global_mean2(spk_mean_sum); 
    global_mean2.Scale(1.0 / num_spks);
    // add term -2 * (sum_{i=0}^n m_i^T) g
    spk_cov.AddVecVec(-2.0, spk_mean_sum, global_mean2);
    
    // add term gT * g
    spk_cov.AddVecVec(1.0, global_mean2, global_mean2);
    spk_cov.Scale(1.0 / num_spks);
    // compute sigma as spk_cov^0.5 for matrix w.r.t its svd decomposition.
    Matrix<double> sigma(spk_cov);
    sigma.Power(0.5);
    // Generate random cmn offset for each speaker, and copy same cmn offsets
    // for all spk's utterances.
    for (std::map<std::string, Vector<double>>::const_iterator 
         iter = spk_means.begin(); iter != spk_means.end(); ++iter) {
      Matrix<double> spk_offsets(num_cmn_offsets, feat_dim); // the 1st row of offset is 0.
      std::string spk_name = iter->first;
      // centered_spk_mean for spk i is cenetered mean defined as m_i - g
      Vector<double> centered_spk_mean(spk_means[spk_name]);
      centered_spk_mean.AddVec(-1.0, global_mean2);
      
      Vector<double> offset_mean(feat_dim);
      if (!zero_mean)
        offset_mean.AddVec(-1.0, centered_spk_mean);

      GenerateRandomOffset(offset_mean, sigma, &spk_offsets);
      if (!preserve_total_covariance) 
        spk_offsets.Scale(cmn_offset_scale);
      else
        PreserveTotalCovariance(cmn_offset_scale, centered_spk_mean, &spk_offsets);

      spk_offsets.Row(0).Set(0.0);
      std::vector<std::string> uttlist = spk_uttlist[spk_name];
      // cmn_offset indexed by itterance. This makes it easier to copy across data dir.
      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        cmn_writer.Write(utt, spk_offsets);
      }
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
