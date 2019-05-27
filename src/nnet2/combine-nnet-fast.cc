// nnet2/combine-nnet-fast.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet2/combine-nnet-fast.h"
#include "nnet2/nnet-update-parallel.h"
#include "util/kaldi-thread.h"

namespace kaldi {
namespace nnet2 {

/*
  This class is responsible for computing a Fisher matrix which is a kind of
  scatter of gradients on subsets; it's used for preconditioning the update in
  class FastNnetCombiner.  */
class FisherComputationClass: public MultiThreadable {
 public:
  FisherComputationClass(const Nnet &nnet,
                         const std::vector<Nnet> &nnets,
                         const std::vector<NnetExample> &egs,
                         int32 minibatch_size,
                         SpMatrix<double> *scatter):
      nnet_(nnet), nnets_(nnets), egs_(egs), minibatch_size_(minibatch_size),
      scatter_ptr_(scatter) { } // This initializer is only used to create a
  // temporary version of the object; the next initializer is used to
  // create the separate versions for the parallel jobs.

  FisherComputationClass(const FisherComputationClass &other):
      MultiThreadable(other),
      nnet_(other.nnet_), nnets_(other.nnets_), egs_(other.egs_),
      minibatch_size_(other.minibatch_size_), scatter_ptr_(other.scatter_ptr_) {
    scatter_.Resize(nnets_.size() * nnet_.NumUpdatableComponents());  }

  void operator () () {
    // b is the "minibatch id."
    int32 num_egs = static_cast<int32>(egs_.size());
    Nnet nnet_gradient(nnet_);
    for (int32 b = 0; b * minibatch_size_ < num_egs; b++) {
      if (b % num_threads_ != thread_id_)
        continue; // We're not responsible for this minibatch.
      int32 offset = b * minibatch_size_,
          length = std::min(minibatch_size_,
                       num_egs - offset);
      bool is_gradient = true;
      nnet_gradient.SetZero(is_gradient);
      std::vector<NnetExample> minibatch(egs_.begin() + offset,
                                                 egs_.begin() + offset + length);
      DoBackprop(nnet_, minibatch, &nnet_gradient);
      Vector<double> gradient(nnets_.size() * nnet_.NumUpdatableComponents());
      int32 i = 0;
      for (int32 n = 0; n < static_cast<int32>(nnets_.size()); n++) {
        for (int32 c = 0; c < nnet_.NumComponents(); c++) {
          const UpdatableComponent *uc = dynamic_cast<const UpdatableComponent*>(
              &(nnet_gradient.GetComponent(c))),
              *uc_other = dynamic_cast<const UpdatableComponent*>(
                  &(nnets_[n].GetComponent(c)));
          if (uc != NULL) {
            gradient(i) = uc->DotProduct(*uc_other);
            i++;
          }
        }
      }
      KALDI_ASSERT(i == gradient.Dim());
      scatter_.AddVec2(1.0, gradient);
    }
  }
  ~FisherComputationClass() {
    if (scatter_.NumRows() != 0) {
      if (scatter_ptr_->NumRows() == 0)
        scatter_ptr_->Resize(scatter_.NumRows());
      scatter_ptr_->AddSp(1.0, scatter_);
    }
  }

 private:
  const Nnet &nnet_; // point at which we compute the parameter gradients.
  const std::vector<Nnet> &nnets_; // The dot-product  of each of these with the parameter gradients,
  // are the actual gradients that go into "scatter".
  const std::vector<NnetExample> &egs_;
  int32 minibatch_size_; // equals config --fisher-minbatch-size e.g. 64 (smaller than
                         // regular minibatch size.)
  SpMatrix<double> *scatter_ptr_;
  SpMatrix<double> scatter_; // Local accumulation of the scatter.
};


class FastNnetCombiner {
 public:
  FastNnetCombiner(const NnetCombineFastConfig &combine_config,
                   const std::vector<NnetExample> &validation_set,
                   const std::vector<Nnet> &nnets_in,
                   Nnet *nnet_out):
      config_(combine_config), egs_(validation_set),
      nnets_(nnets_in), nnet_out_(nnet_out) {

    GetInitialParams();
    ComputePreconditioner();

    int32 dim = params_.Dim();
    KALDI_ASSERT(dim > 0);
    Vector<double> gradient(dim);

    double regularizer_objf, initial_regularizer_objf; // for diagnostics
    double objf, initial_objf;

    LbfgsOptions lbfgs_options;
    lbfgs_options.minimize = false; // We're maximizing.
    lbfgs_options.m = std::min(dim, config_.max_lbfgs_dim);
    lbfgs_options.first_step_impr = config_.initial_impr;

    OptimizeLbfgs<double> lbfgs(params_,
                                lbfgs_options);

    for (int32 i = 0; i < config_.num_lbfgs_iters; i++) {
      params_.CopyFromVec(lbfgs.GetProposedValue());
      objf = ComputeObjfAndGradient(&gradient, &regularizer_objf);
      // Note: there is debug printout in ComputeObjfAndGradient
      // (at verbose-level 2).
      if (i == 0) {
        initial_objf = objf;
        initial_regularizer_objf = regularizer_objf;
      }
      lbfgs.DoStep(objf, gradient);
    }
    params_ = lbfgs.GetValue(&objf);

    ComputeCurrentNnet(nnet_out_, true); // create the output neural net, and
                                         // print out the scaling factors.
    if (config_.regularizer != 0.0) {
      double initial_part = initial_objf - initial_regularizer_objf,
          part = objf - regularizer_objf;
      KALDI_LOG << "Combining nnets, objf/frame + regularizer changed from "
                << initial_part << " + " << initial_regularizer_objf
                << " = " << initial_objf << " to " << part << " + "
                << regularizer_objf << " = " << objf;
    } else {
      KALDI_LOG << "Combining nnets, objf per frame changed from "
                << initial_objf << " to " << objf;
    }
  }

 private:
  int32 GetInitialModel(
      const std::vector<NnetExample> &validation_set,
      const std::vector<Nnet> &nnets) const;

  void GetInitialParams();

  void ComputePreconditioner();

  // Computes and returns objective function per frame, including
  // regularizer term if applicable.  Also puts just the regularizer
  // term in *regularizer_objf.
  double ComputeObjfAndGradient(
      Vector<double> *gradient,
      double *regularizer_objf);

  void ComputeCurrentNnet(
      Nnet *dest, bool debug = false);

  static void CombineNnets(const Vector<double> &scale_params,
                           const std::vector<Nnet> &nnets,
                           Nnet *dest);


  // C_ is the cholesky of the smoothed Fisher matrix F.
  // Let F = C C^T.
  // Preconditioned gradient is \hat{g} = C^{-1} g
  // Note: preconditioned parameter is \hat{p} = C^T p,
  // so p = C^{-T} \hat{p}.
  TpMatrix<double> C_;
  TpMatrix<double> C_inv_;
  Vector<double> params_; // the parameters we're optimizing-- in the
                          // preconditioned space.  These are the same dimension
                          // as the number of nnets we're combining times the
                          // number of updatable layers.

  const NnetCombineFastConfig &config_;
  const std::vector<NnetExample> &egs_;
  const std::vector<Nnet> &nnets_;
  Nnet *nnet_out_;
};


// static
void FastNnetCombiner::CombineNnets(const Vector<double> &scale_params,
                                    const std::vector<Nnet> &nnets,
                                    Nnet *dest) {
  int32 num_nnets = nnets.size();
  KALDI_ASSERT(num_nnets >= 1);
  int32 num_uc = nnets[0].NumUpdatableComponents();
  KALDI_ASSERT(nnets[0].NumUpdatableComponents() >= 1);


  *dest = nnets[0];
  SubVector<double> scale_params0(scale_params, 0, num_uc);
  dest->ScaleComponents(Vector<BaseFloat>(scale_params0));
  for (int32 n = 1; n < num_nnets; n++) {
    SubVector<double> scale_params_n(scale_params, n * num_uc, num_uc);
    dest->AddNnet(Vector<BaseFloat>(scale_params_n), nnets[n]);
  }
}


void FastNnetCombiner::ComputePreconditioner() {
  SpMatrix<double> F; // Fisher matrix.
  Nnet nnet;
  ComputeCurrentNnet(&nnet); // will be at initial value of neural net.

  { // This block does the multi-threaded computation.
    // The next line just initializes an "example" object.
    FisherComputationClass fc(nnet, nnets_, egs_,
                              config_.fisher_minibatch_size,
                              &F);

    // Setting num_threads to zero if config_.num_threads == 1
    // is a signal to the MultiThreader class to run without creating
    // any extra threads in this case; it helps support GPUs.
    int32 num_threads = config_.num_threads == 1 ? 0 : config_.num_threads;
    // The work gets done in the initializer and destructor of
    // the class below.
    MultiThreader<FisherComputationClass> m(num_threads, fc);
  }

  // The scale of F is irrelevant but it might be quite
  // large at this point, so we just normalize it.
  KALDI_ASSERT(F.Trace() > 0);
  F.Scale(F.NumRows() / F.Trace()); // same scale as unit matrix.
  // Make zero diagonal elements of F non-zero.  Relates to updatable
  // components that have no effect, e.g. MixtureProbComponents that have
  // no real free parameters.
  KALDI_ASSERT(config_.fisher_floor > 0.0);
  for (int32 i = 0; i < F.NumRows(); i++)
    F(i, i) = std::max<BaseFloat>(F(i, i), config_.fisher_floor);
  // We next smooth the diagonal elements of F by a small amount.
  // This is mainly necessary in case the number of minibatches is
  // smaller than the dimension of F; we want to ensure F is full rank.
  for (int32 i = 0; i < F.NumRows(); i++)
    F(i, i) *= (1.0 + config_.alpha);

  C_.Resize(F.NumRows());
  C_.Cholesky(F);
  C_inv_ = C_;
  C_inv_.Invert();

  // Transform the params_ data-member to be in the preconditioned space.
  Vector<double> raw_params(params_);
  params_.AddTpVec(1.0, C_, kTrans, raw_params, 0.0);
}

// Note, we ignore the regularizer in selecting the best one.  It shouldn't
// really matter.
void FastNnetCombiner::GetInitialParams() {
  int32 initial_model = config_.initial_model,
      num_nnets = static_cast<int32>(nnets_.size());
  if (initial_model > num_nnets)
    initial_model = num_nnets;
  if (initial_model < 0)
    initial_model = GetInitialModel(egs_, nnets_);

  KALDI_ASSERT(initial_model >= 0 && initial_model <= num_nnets);
  int32 num_uc = nnets_[0].NumUpdatableComponents();

  Vector<double> raw_params(num_uc * num_nnets); // parameters in
                                                 // non-preconditioned space.
  if (initial_model < num_nnets) {
    KALDI_LOG << "Initializing with neural net with index " << initial_model;
    // At this point we're using the best of the individual neural nets.
    raw_params.Set(0.0);

    // Set the block of parameters corresponding to the "best" of the
    // source neural nets to
    SubVector<double> best_block(raw_params, num_uc * initial_model, num_uc);
    best_block.Set(1.0);
  } else { // initial_model == num_nnets
    KALDI_LOG << "Initializing with all neural nets averaged.";
    raw_params.Set(1.0 / num_nnets);
  }
  KALDI_ASSERT(C_.NumRows() == 0); // Assume this not set up yet.
  params_ = raw_params; // this is in non-preconditioned space.
}

/// Computes objf at point "params_".
double FastNnetCombiner::ComputeObjfAndGradient(
    Vector<double> *gradient,
    double *regularizer_objf_ptr) {
  Nnet nnet;
  ComputeCurrentNnet(&nnet); // compute it at the value "params_".

  Nnet nnet_gradient(nnet);
  bool is_gradient = true;
  nnet_gradient.SetZero(is_gradient);
  double tot_weight = 0.0;
  double objf = DoBackpropParallel(nnet, config_.minibatch_size, config_.num_threads,
                                   egs_, &tot_weight, &nnet_gradient) / egs_.size();

  // raw_gradient is gradient in non-preconditioned space.
  Vector<double> raw_gradient(params_.Dim());

  double regularizer_objf = 0.0; // sum of -0.5 * config_.regularizer * params-squared.
  int32 i = 0; // index into raw_gradient
  int32 num_nnets = nnets_.size();
  for (int32 n = 0; n < num_nnets; n++) {
    for (int32 j = 0; j < nnet.NumComponents(); j++) {
      const UpdatableComponent *uc =
          dynamic_cast<const UpdatableComponent*>(&(nnets_[n].GetComponent(j))),
          *uc_gradient =
          dynamic_cast<const UpdatableComponent*>(&(nnet_gradient.GetComponent(j))),
          *uc_params =
          dynamic_cast<const UpdatableComponent*>(&(nnet.GetComponent(j)));
      if (uc != NULL) {
        double gradient = uc->DotProduct(*uc_gradient) / tot_weight;
        // "gradient" is the derivative of the objective function w.r.t. this
        // element of the parameters (i.e. this weight, which gets applied to
        // the j'th component of the n'th source neural net).
        if (config_.regularizer != 0.0) {
          gradient -= config_.regularizer * uc->DotProduct(*uc_params);
          if (n == 0) // only add this once...
            regularizer_objf +=
                -0.5 * config_.regularizer * uc_params->DotProduct(*uc_params);
        }
        raw_gradient(i) = gradient;
        i++;
      }
    }
  }
  if (config_.regularizer != 0.0) {
    KALDI_VLOG(2) << "Objf is " << objf << " + regularizer " << regularizer_objf
                  << " = " << (objf + regularizer_objf) << ", raw gradient is "
                  << raw_gradient;
  } else {
    KALDI_VLOG(2) << "Objf is " << objf << ", raw gradient is " << raw_gradient;
  }
  KALDI_ASSERT(i == raw_gradient.Dim());
  // \hat{g} = C^{-1} g.
  gradient->AddTpVec(1.0, C_inv_, kNoTrans, raw_gradient, 0.0);
  *regularizer_objf_ptr = regularizer_objf;
  return objf + regularizer_objf;
}

void FastNnetCombiner::ComputeCurrentNnet(
    Nnet *dest, bool debug) {
  int32 num_nnets = nnets_.size();
  KALDI_ASSERT(num_nnets >= 1);
  KALDI_ASSERT(params_.Dim() == num_nnets * nnets_[0].NumUpdatableComponents());
  Vector<double> raw_params(params_.Dim()); // Weights in non-preconditioned space:
  // p = C^{-T} \hat{p}.  Here, raw_params is p, params_, is \hat{p}.

  if (C_inv_.NumRows() > 0)
    raw_params.AddTpVec(1.0, C_inv_, kTrans, params_, 0.0);
  else
    raw_params = params_; // C not set up yet: interpret params_ as raw parameters.

  if (debug) {
    Matrix<double> params_mat(num_nnets,
                              nnets_[0].NumUpdatableComponents());
    params_mat.CopyRowsFromVec(raw_params);
    KALDI_LOG << "Scale parameters are " << params_mat;
  }
  CombineNnets(raw_params, nnets_, dest);
}

/// Returns an integer saying which model to use:
/// either 0 ... num-models - 1 for the best individual model,
/// or (#models) for the average of all of them.
int32 FastNnetCombiner::GetInitialModel(
    const std::vector<NnetExample> &validation_set,
    const std::vector<Nnet> &nnets) const {
  int32 num_nnets = static_cast<int32>(nnets.size());
  KALDI_ASSERT(!nnets.empty());
  int32 best_n = -1;
  double best_objf = -std::numeric_limits<double>::infinity();
  Vector<double> objfs(nnets.size());
  for (int32 n = 0; n < num_nnets; n++) {
    double num_frames;
    double objf = ComputeNnetObjfParallel(nnets[n], config_.minibatch_size,
                                          config_.num_threads, validation_set,
                                          &num_frames);
    KALDI_ASSERT(num_frames != 0);
    objf /= num_frames;

    if (n == 0 || objf > best_objf) {
      best_objf = objf;
      best_n = n;
    }
    objfs(n) = objf;
  }
  KALDI_LOG << "Objective functions for the source neural nets are " << objfs;

  int32 num_uc = nnets[0].NumUpdatableComponents();

  if (num_nnets > 1) { // Now try a version where all the neural nets have the
                       // same weight.  Don't do this if num_nnets == 1 as
                       // it would be a waste of time (identical to n == 0).
    Vector<double> scale_params(num_uc * num_nnets);
    scale_params.Set(1.0 / num_nnets);
    Nnet average_nnet;
    CombineNnets(scale_params, nnets, &average_nnet);
    double num_frames;
    double objf = ComputeNnetObjfParallel(average_nnet, config_.minibatch_size,
                                          config_.num_threads, validation_set,
                                          &num_frames);
    objf /= num_frames;
    KALDI_LOG << "Objf with all neural nets averaged is " << objf;
    if (objf > best_objf) {
      return num_nnets;
    } else {
      return best_n;
    }
  } else {
    return best_n;
  }
}

void CombineNnetsFast(const NnetCombineFastConfig &combine_config,
                      const std::vector<NnetExample> &validation_set,
                      const std::vector<Nnet> &nnets_in,
                      Nnet *nnet_out) {
  // Everything happens in the initializer.
  FastNnetCombiner combiner(combine_config,
                            validation_set,
                            nnets_in,
                            nnet_out);
}


} // namespace nnet2
} // namespace kaldi
