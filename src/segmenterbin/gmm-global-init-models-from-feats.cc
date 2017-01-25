// gmmbin/gmm-global-init-models-from-feats.cc

// Copyright 2013   Johns Hopkins University (author: Daniel Povey)
//           2016   Vimal Manohar

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
#include "gmm/model-common.h"
#include "gmm/full-gmm.h"
#include "gmm/diag-gmm.h"
#include "gmm/mle-full-gmm.h"

namespace kaldi {

// We initialize the GMM parameters by setting the variance to the global
// variance of the features, and the means to distinct randomly chosen frames.
void InitGmmFromRandomFrames(const MatrixBase<BaseFloat> &feats, DiagGmm *gmm) {
  int32 num_gauss = gmm->NumGauss(), num_frames = feats.NumRows(),
      dim = feats.NumCols();
  KALDI_ASSERT(num_frames >= 10 * num_gauss && "Too few frames to train on");
  Vector<double> mean(dim), var(dim);
  for (int32 i = 0; i < num_frames; i++) {
    mean.AddVec(1.0 / num_frames, feats.Row(i));
    var.AddVec2(1.0 / num_frames, feats.Row(i));
  }
  var.AddVec2(-1.0, mean);
  if (var.Max() <= 0.0)
    KALDI_ERR << "Features do not have positive variance " << var;
  
  DiagGmmNormal gmm_normal(*gmm);

  std::set<int32> used_frames;
  for (int32 g = 0; g < num_gauss; g++) {
    int32 random_frame = RandInt(0, num_frames - 1);
    while (used_frames.count(random_frame) != 0)
      random_frame = RandInt(0, num_frames - 1);
    used_frames.insert(random_frame);
    gmm_normal.weights_(g) = 1.0 / num_gauss;
    gmm_normal.means_.Row(g).CopyFromVec(feats.Row(random_frame));
    gmm_normal.vars_.Row(g).CopyFromVec(var);
  }
  gmm->CopyFromNormal(gmm_normal);
  gmm->ComputeGconsts();
}

void MleDiagGmmSharedVarsUpdate(const MleDiagGmmOptions &config,
                                const AccumDiagGmm &diag_gmm_acc,
                                GmmFlagsType flags,
                                DiagGmm *gmm,
                                BaseFloat *obj_change_out,
                                BaseFloat *count_out,
                                int32 *floored_elements_out,
                                int32 *floored_gaussians_out,
                                int32 *removed_gaussians_out) {
  KALDI_ASSERT(gmm != NULL);

  if (flags & ~diag_gmm_acc.Flags())
    KALDI_ERR << "Flags in argument do not match the active accumulators";

  KALDI_ASSERT(diag_gmm_acc.NumGauss() == gmm->NumGauss() &&
               diag_gmm_acc.Dim() == gmm->Dim());

  int32 num_gauss = gmm->NumGauss();
  double occ_sum = diag_gmm_acc.occupancy().Sum();

  int32 elements_floored = 0, gauss_floored = 0;
  
  // remember old objective value
  gmm->ComputeGconsts();
  BaseFloat obj_old = MlObjective(*gmm, diag_gmm_acc);

  // First get the gmm in "normal" representation (not the exponential-model
  // form).
  DiagGmmNormal ngmm(*gmm);

  Vector<double> shared_var(gmm->Dim());

  std::vector<int32> to_remove;
  for (int32 i = 0; i < num_gauss; i++) {
    double occ = diag_gmm_acc.occupancy()(i);
    double prob;
    if (occ_sum > 0.0)
      prob = occ / occ_sum;
    else
      prob = 1.0 / num_gauss;

    if (occ > static_cast<double>(config.min_gaussian_occupancy)
        && prob > static_cast<double>(config.min_gaussian_weight)) {
      
      ngmm.weights_(i) = prob;
      
      // copy old mean for later normalizations
      Vector<double> old_mean(ngmm.means_.Row(i));
      
      // update mean, then variance, as far as there are accumulators 
      if (diag_gmm_acc.Flags() & (kGmmMeans|kGmmVariances)) {
        Vector<double> mean(diag_gmm_acc.mean_accumulator().Row(i));
        mean.Scale(1.0 / occ);
        // transfer to estimate
        ngmm.means_.CopyRowFromVec(mean, i);
      }
      
      if (diag_gmm_acc.Flags() & kGmmVariances) {
        KALDI_ASSERT(diag_gmm_acc.Flags() & kGmmMeans);
        Vector<double> var(diag_gmm_acc.variance_accumulator().Row(i));
        var.Scale(1.0 / occ);
        var.AddVec2(-1.0, ngmm.means_.Row(i));  // subtract squared means.
        
        // if we intend to only update the variances, we need to compensate by 
        // adding the difference between the new and old mean
        if (!(flags & kGmmMeans)) {
          old_mean.AddVec(-1.0, ngmm.means_.Row(i));
          var.AddVec2(1.0, old_mean);
        }
        shared_var.AddVec(occ, var);
      }
    } else {  // Insufficient occupancy.
      if (config.remove_low_count_gaussians &&
          static_cast<int32>(to_remove.size()) < num_gauss-1) {
        // remove the component, unless it is the last one.
        KALDI_WARN << "Too little data - removing Gaussian (weight "
                   << std::fixed << prob
                   << ", occupation count " << std::fixed << diag_gmm_acc.occupancy()(i)
                   << ", vector size " << gmm->Dim() << ")";
        to_remove.push_back(i);
      } else {
        KALDI_WARN << "Gaussian has too little data but not removing it because"
                   << (config.remove_low_count_gaussians ?
                       " it is the last Gaussian: i = "
                       : " remove-low-count-gaussians == false: g = ") << i
                   << ", occ = " << diag_gmm_acc.occupancy()(i) << ", weight = " << prob;
        ngmm.weights_(i) =
            std::max(prob, static_cast<double>(config.min_gaussian_weight));
      }
    }
  }
        
  if (diag_gmm_acc.Flags() & kGmmVariances) {
    int32 floored;
    if (config.variance_floor_vector.Dim() != 0) {
      floored = shared_var.ApplyFloor(config.variance_floor_vector);
    } else {
      floored = shared_var.ApplyFloor(config.min_variance);
    }
    if (floored != 0) {
      elements_floored += floored;
      gauss_floored++;
    }

    shared_var.Scale(1.0 / occ_sum);
    for (int32 i = 0; i < num_gauss; i++) {
      ngmm.vars_.CopyRowFromVec(shared_var, i);
    }
  }
  
  // copy to natural representation according to flags
  ngmm.CopyToDiagGmm(gmm, flags);

  gmm->ComputeGconsts();  // or MlObjective will fail.
  BaseFloat obj_new = MlObjective(*gmm, diag_gmm_acc);
  
  if (obj_change_out) 
    *obj_change_out = (obj_new - obj_old);
  if (count_out) *count_out = occ_sum;
  if (floored_elements_out) *floored_elements_out = elements_floored;
  if (floored_gaussians_out) *floored_gaussians_out = gauss_floored;
  
  if (to_remove.size() > 0) {
    gmm->RemoveComponents(to_remove, true /*renormalize weights*/);
    gmm->ComputeGconsts();
  }
  if (removed_gaussians_out != NULL) *removed_gaussians_out = to_remove.size();

  if (gauss_floored > 0)
    KALDI_VLOG(2) << gauss_floored << " variances floored in " << gauss_floored
                  << " Gaussians.";
}


void TrainOneIter(const MatrixBase<BaseFloat> &feats,
                  const MleDiagGmmOptions &gmm_opts,
                  int32 iter,
                  int32 num_threads,
                  bool share_covars,
                  DiagGmm *gmm) {
  AccumDiagGmm gmm_acc(*gmm, kGmmAll);

  Vector<BaseFloat> frame_weights(feats.NumRows(), kUndefined);
  frame_weights.Set(1.0);

  double tot_like;
  tot_like = gmm_acc.AccumulateFromDiagMultiThreaded(*gmm, feats, frame_weights,
                                                     num_threads);

  KALDI_LOG << "Likelihood per frame on iteration " << iter
            << " was " << (tot_like / feats.NumRows()) << " over "
            << feats.NumRows() << " frames.";
  
  BaseFloat objf_change, count;
  MleDiagGmmUpdate(gmm_opts, gmm_acc, kGmmAll, gmm, &objf_change, &count);

  KALDI_LOG << "Objective-function change on iteration " << iter << " was "
            << (objf_change / count) << " over " << count << " frames.";
}

void TrainGmm(const MatrixBase<BaseFloat> &feats, 
              const MleDiagGmmOptions &gmm_opts,
              int32 num_gauss, int32 num_gauss_init, int32 num_iters,
              int32 num_threads, bool share_covars, DiagGmm *gmm) {
  KALDI_LOG << "Initializing GMM means from random frames to "
            << num_gauss_init << " Gaussians.";
  InitGmmFromRandomFrames(feats, gmm);

  // we'll increase the #Gaussians by splitting,
  // till halfway through training.
  int32 cur_num_gauss = num_gauss_init,
      gauss_inc = (num_gauss - num_gauss_init) / (num_iters / 2);
      
  for (int32 iter = 0; iter < num_iters; iter++) {
    TrainOneIter(feats, gmm_opts, iter, num_threads, share_covars, gmm);

    int32 next_num_gauss = std::min(num_gauss, cur_num_gauss + gauss_inc);
    if (next_num_gauss > gmm->NumGauss()) {
      KALDI_LOG << "Splitting to " << next_num_gauss << " Gaussians.";
      gmm->Split(next_num_gauss, 0.1);
      cur_num_gauss = next_num_gauss;
    }
  }
}

} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "This program initializes a single diagonal GMM and does multiple iterations of\n"
        "training from features stored in memory.\n"
        "Usage:  gmm-global-init-from-feats [options] <feature-rspecifier> <models-wspecifier>\n"
        "e.g.: gmm-global-init-from-feats scp:train.scp ark:1.ark\n";

    ParseOptions po(usage);
    MleDiagGmmOptions gmm_opts;
    
    bool binary = true;
    int32 num_gauss = 100;
    int32 num_gauss_init = 0;
    int32 max_gauss = 0;
    int32 min_gauss = 0;
    int32 num_iters = 50;
    int32 num_frames = 200000;
    int32 srand_seed = 0;
    int32 num_threads = 4;
    BaseFloat num_gauss_fraction = -1;
    bool share_covars = false;
    std::string spk2utt_rspecifier;
    
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("num-gauss", &num_gauss, "Number of Gaussians in the model");
    po.Register("num-gauss-init", &num_gauss_init, "Number of Gaussians in "
                "the model initially (if nonzero and less than num_gauss, "
                "we'll do mixture splitting)");
    po.Register("num-iters", &num_iters, "Number of iterations of training");
    po.Register("num-frames", &num_frames, "Number of feature vectors to store in "
                "memory and train on (randomly chosen from the input features)");
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    po.Register("num-threads", &num_threads, "Number of threads used for "
                "statistics accumulation");
    po.Register("spk2utt-rspecifier", &spk2utt_rspecifier, 
                "If specified, estimates models per-speaker");
    po.Register("num-gauss-fraction", &num_gauss_fraction, 
                "If specified, chooses the number of gaussians to be "
                "num-gauss-fraction * min(num-frames-available, num-frames). "
                "This number is expected to be in the range(0, 0.1).");
    po.Register("max-gauss", &max_gauss, "Maximum number of Gaussians allowed "
                "in the model. Applicable when num_gauss_fraction is specified.");
    po.Register("min-gauss", &min_gauss, "Minimum number of Gaussians allowed "
                "in the model. Applicable when num_gauss_fraction is specified.");
    po.Register("share-covars", &share_covars, "If true, then the variances "
                "of the Gaussian components are tied.");
                
    gmm_opts.Register(&po);

    po.Read(argc, argv);

    srand(srand_seed);    
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    if (num_gauss_fraction != -1) {
      KALDI_ASSERT(num_gauss_fraction > 0 && num_gauss_fraction < 0.1);
    }

    KALDI_ASSERT(max_gauss >= 0 && min_gauss >= 0 && max_gauss >= min_gauss);

    std::string feature_rspecifier = po.GetArg(1),
        model_wspecifier = po.GetArg(2);
    
    DiagGmmWriter gmm_writer(model_wspecifier);

    KALDI_ASSERT(num_frames > 0);
  
    if (spk2utt_rspecifier.empty()) {
      KALDI_LOG << "Reading features (will keep " << num_frames << " frames "
                << "per utterance.)";
    } else {
      KALDI_LOG << "Reading features (will keep " << num_frames << " frames "
                << "per speaker.)";
    }
    
    int32 dim = 0;

    if (spk2utt_rspecifier.empty()) {
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        const Matrix<BaseFloat> &this_feats = feature_reader.Value();
        if (dim == 0) {
          dim = this_feats.NumCols();
        } else if (this_feats.NumCols() != dim) {
          KALDI_ERR << "Features have inconsistent dims "
            << this_feats.NumCols() << " vs. " << dim
            << " (current utt is) " << feature_reader.Key();
        }

        Matrix<BaseFloat> feats(num_frames, dim);
        int64 num_read = 0; 

        for (int32 t = 0; t < this_feats.NumRows(); t++) {
          num_read++;
          if (num_read <= num_frames) {
            feats.Row(num_read - 1).CopyFromVec(this_feats.Row(t));
          } else {
            BaseFloat keep_prob = num_frames / static_cast<BaseFloat>(num_read);
            if (WithProb(keep_prob)) { // With probability "keep_prob"
              feats.Row(RandInt(0, num_frames - 1)).CopyFromVec(this_feats.Row(t));
            }
          }
        }

        KALDI_ASSERT(num_read > 0);
        
        if (num_read < num_frames) {
          KALDI_WARN << "For utterance " << feature_reader.Key() << ", "
                     << "number of frames read " << num_read << " was less than "
                     << "target number " << num_frames << ", using all we read.";
          feats.Resize(num_read, dim, kCopyData);
        } else {
          BaseFloat percent = num_frames * 100.0 / num_read;
          KALDI_LOG << "For utterance " << feature_reader.Key() << ", "
                    << "kept " << num_frames << " out of " << num_read
                    << " input frames = " << percent << "%.";
        }

        int32 this_num_gauss_init = num_gauss_init;
        int32 this_num_gauss = num_gauss;

        if (num_gauss_fraction != -1) {
          this_num_gauss = feats.NumRows() * num_gauss_fraction;
          if (this_num_gauss > max_gauss)
            this_num_gauss = max_gauss;
          if (this_num_gauss < min_gauss)
            this_num_gauss = min_gauss;
        }
        
        if (this_num_gauss_init <= 0 || this_num_gauss_init > this_num_gauss)
          this_num_gauss_init = this_num_gauss;

        DiagGmm gmm(this_num_gauss_init, dim);
        TrainGmm(feats, gmm_opts, this_num_gauss, this_num_gauss_init, 
                 num_iters, num_threads, share_covars, &gmm);

        gmm_writer.Write(feature_reader.Key(), gmm);
      }
      KALDI_LOG << "Done initializing GMMs.";
    } else {
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

      int32 num_err = 0;
      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        const std::vector<std::string> &uttlist = spk2utt_reader.Value();

        Matrix<BaseFloat> feats;
        int64 num_read = 0; 

        for (std::vector<std::string>::const_iterator it = uttlist.begin();
             it != uttlist.end(); ++it) {
          if (!feature_reader.HasKey(*it)) {
            KALDI_WARN << "Could not find features for utterance " << *it;
            num_err++;
          }
          
          const Matrix<BaseFloat> &this_feats = feature_reader.Value(*it);
          if (feats.NumCols() == 0) {
            dim = this_feats.NumCols();
            feats.Resize(num_frames, dim);
          } else if (this_feats.NumCols() != dim) {
            KALDI_ERR << "Features have inconsistent dims "
              << this_feats.NumCols() << " vs. " << dim
              << " (current utt is) " << *it; 
          }

          for (int32 t = 0; t < this_feats.NumRows(); t++) {
            num_read++;
            if (num_read <= num_frames) {
              feats.Row(num_read - 1).CopyFromVec(this_feats.Row(t));
            } else {
              BaseFloat keep_prob = num_frames / static_cast<BaseFloat>(num_read);
              if (WithProb(keep_prob)) { // With probability "keep_prob"
                feats.Row(RandInt(0, num_frames - 1)).CopyFromVec(this_feats.Row(t));
              }
            }
          }
        }
        
        KALDI_ASSERT(num_read > 0);

        if (num_read < num_frames) {
          KALDI_WARN << "For speaker " << spk2utt_reader.Key() << ", "
                     << "number of frames read " << num_read << " was less than "
                     << "target number " << num_frames << ", using all we read.";
          feats.Resize(num_read, dim, kCopyData);
        } else {
          BaseFloat percent = num_frames * 100.0 / num_read;
          KALDI_LOG << "For spekear " << spk2utt_reader.Key() << ", "
                    << "kept " << num_frames << " out of " << num_read
                    << " input frames = " << percent << "%.";
        }
      
        int32 this_num_gauss_init = num_gauss_init;
        int32 this_num_gauss = num_gauss;

        if (num_gauss_fraction != -1) {
          this_num_gauss = feats.NumRows() * num_gauss_fraction;
          if (this_num_gauss > max_gauss)
            this_num_gauss = max_gauss;
          if (this_num_gauss < min_gauss)
            this_num_gauss = min_gauss;
        }
        
        if (this_num_gauss_init <= 0 || this_num_gauss_init > this_num_gauss)
          this_num_gauss_init = this_num_gauss;

        DiagGmm gmm(this_num_gauss_init, dim);
        TrainGmm(feats, gmm_opts, this_num_gauss, this_num_gauss_init, 
                 num_iters, num_threads, share_covars, &gmm);

        gmm_writer.Write(spk2utt_reader.Key(), gmm);
      }

      KALDI_LOG << "Done initializing GMMs. Failed getting features for "
                << num_err << "utterances";
    }

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
