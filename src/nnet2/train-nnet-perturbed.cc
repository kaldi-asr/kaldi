// nnet2/train-nnet-perturbed.cc

// Copyright 2012-2014   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet2/train-nnet-perturbed.h"
#include "nnet2/nnet-update.h"
#include "thread/kaldi-thread.h"

namespace kaldi {
namespace nnet2 {


class NnetPerturbedUpdater {
 public:
  // Note: in the case of training with SGD, "nnet" and "nnet_to_update" will be
  // identical.  They'd be different if we're accumulating the gradient for a
  // held-out set and don't want to update the model, but this shouldn't happen
  // for this "perturbed" update.  nnet_to_update may be NULL if you don't
  // want do do backprop, but this probably doesn't make sense.
  // num_layers_before_input is the number of layers to ignore before what
  // we consider to be the input (x) for purposes of this technique.  This will
  // likely equal 2: one for the feature-splicing layer (SpliceComponent) and
  // one for the preconditioning layer (FixedAffineComponent).  The within_class_covar
  // argument (within_class_covar)
  // 
  // within_class_covar is the within-class covariance matrix
  NnetPerturbedUpdater(const Nnet &nnet,
                       int32 num_layers_before_input,
                       const CuMatrix<BaseFloat> &within_class_covar,
                       Nnet *nnet_to_update);
  
  // This function does the entire forward and backward computation for this
  // minbatch.  Outputs to tot_objf_orig and tot_objf_perturbed the total
  // objective function (including any weighting factors) over this minibatch,
  // and the same after perturbing the data.
  void ComputeForMinibatch(const std::vector<NnetExample> &data,
                           BaseFloat D,
                           double *tot_objf_orig,
                           double *tot_objf_perturbed);
  
 protected:

  /// takes the input and formats as a single matrix, in forward_data_[0].
  void FormatInput(const std::vector<NnetExample> &data);

  /// Do the forward propagation for layers 0 ... num_layers_before_input_ - 1,
  /// typically the first two layers.  This will be called once per minibatch.
  void PropagateInitial() { Propagate(0, num_layers_before_input_); }


  /// Do the forward propagation for layers num_layers_before_input_
  /// ... num-layers-1, typically all but the first two layers.  This will be
  /// called twice per minibatch, once before and once after perturbing the
  /// inputs.
  void PropagateRemaining() { Propagate(num_layers_before_input_,
                                        nnet_.NumComponents()); }

  /// Internal Propagate function, does the forward computation for
  /// layers begin_layer ... end_layer - 1.
  void Propagate(int32 begin_layer, int32 end_layer);
  
  /// Computes objective function and derivative at output layer, but does not
  /// do the backprop [for that, see Backprop()].  This will be called twice per
  /// minibatch, once before and once after perturbing the inputs.
  void ComputeObjfAndDeriv(const std::vector<MatrixElement<BaseFloat> > &sv_labels,
                           CuMatrix<BaseFloat> *deriv,
                           BaseFloat *tot_objf,
                           BaseFloat *tot_weight) const;

  /// Computes supervision labels from data.
  void ComputeSupervisionLabels(const std::vector<NnetExample> &data,
                                std::vector<MatrixElement<BaseFloat> > *sv_labels);

  /// Backprop must be called after ComputeObjfAndDeriv (it will be called
  /// twice, the first time with a NULL nnet_to_update pointer).  It does the
  /// backpropagation (not including the first num_layers_before_input_ layers).
  /// "nnet_to_update" is updated, if non-NULL.  Note: "deriv" will contain, at
  /// input, the derivative w.r.t. the output layer (as computed by
  /// ComputeObjfAndDeriv), but will be used as a temporary variable by this
  /// function, and exit, will contain the derivative of the objective function
  /// w.r.t. the input of layer num_layers_before_input_.
  void Backprop(Nnet *nnet_to_update,
                CuMatrix<BaseFloat> *deriv) const;

  /// Perturb the input features (actually, the features at the input of layer
  /// num_layers_before_input_).  This modifies the value of
  /// forward_data_[num_layers_before_input_].  For the math, see \ref
  /// train-nnet-perturbed.h
  void PerturbInput(const CuMatrix<BaseFloat> &deriv_at_input,
                    BaseFloat D);                    
  
 private:
  
  const Nnet &nnet_;
  
  Nnet *nnet_to_update_;  
  int32 num_layers_before_input_;  // Number of layers before whichever layer we
                                   // regard as the input for purposes of this
                                   // method (normally 2, to include splicing
                                   // layer and preconditioning layer)
  std::vector<ChunkInfo> chunk_info_out_;
  const CuMatrix<BaseFloat> &within_class_covar_;
  
  int32 num_chunks_; // same as the minibatch size.
  
  std::vector<CuMatrix<BaseFloat> > forward_data_; // The forward data
  // for the outputs of each of the components.
};


NnetPerturbedUpdater::NnetPerturbedUpdater(const Nnet &nnet,
                                           int32 num_layers_before_input,
                                           const CuMatrix<BaseFloat> &within_class_covar,
                                           Nnet *nnet_to_update):
    nnet_(nnet),
    nnet_to_update_(nnet_to_update),
    num_layers_before_input_(num_layers_before_input),
    within_class_covar_(within_class_covar) {
  KALDI_ASSERT(num_layers_before_input_ >= 0 &&
               num_layers_before_input < nnet.NumComponents());
  for (int32 c = 0; c < num_layers_before_input_; c++) {
    const Component *comp = &(nnet.GetComponent(c));
    const UpdatableComponent *uc = dynamic_cast<const UpdatableComponent*>(comp);
    if (uc != NULL) {
      KALDI_ERR << "One of the pre-input layers is updatable.";
    }
  }
}    

void NnetPerturbedUpdater::PerturbInput(
    const CuMatrix<BaseFloat> &deriv_at_input,
    BaseFloat D) {
  // The code doesn't handle the case where there is further splicing after the
  // input.
  KALDI_ASSERT(num_chunks_ == deriv_at_input.NumRows());
  // For the math, see train-nnet-perturbed.h.
  // deriv_at_input is \nabla in the math.

  // "input" is the input features, currently unmodified, but we'll
  // modify them.
  CuMatrix<BaseFloat> &input(forward_data_[num_layers_before_input_]);
  KALDI_ASSERT(SameDim(input, deriv_at_input));
  // Each row of deriv_w will equal (W nabla_t)', where ' is transpose.
  CuMatrix<BaseFloat> deriv_w(input.NumRows(), input.NumCols());
  // note: for the second transpose-ness argument below we can choose either
  // kTrans or kNoTrans because the matrix is symmetric.  I'm guessing that
  // kTrans will be faster.
  deriv_w.AddMatMat(1.0, deriv_at_input, kNoTrans,
                    within_class_covar_, kTrans, 0.0);
  
  // k will be used to compute and store the gradient-scaling factor k_t.
  CuVector<BaseFloat> k(deriv_at_input.NumRows());
  // after the next call, each element of k will contain (\nabla_t^T W \nabla_t)
  // We want k_t = D / sqrt(\nabla_t^T W \nabla_t)
  // so we need to take this to the power -0.5.
  // We can't do this if it's zero, so we first floor to a very small value.
  k.AddDiagMatMat(1.0, deriv_w, kNoTrans, deriv_at_input, kTrans, 0.0);
  int32 num_floored = k.ApplyFloor(1.0e-20);
  if (num_floored > 0.0) {
    // Should only happen at the very start of training, 
    KALDI_WARN << num_floored << " gradients floored (derivative at input was "
               << "close to zero).. should only happen at start of training "
               << "or when adding a new layer.";
  }
  k.ApplyPow(-0.5);
  // now we have k_t = 1.0 / sqrt(\nabla_t^T W \nabla_t).
  // in the math, k_t contains an additional factor of D, but we'll
  // add this later.
  // Below, we will do  x'_t = x_t - k_t W \nabla_t
  // Here, each row of deriv_w contains the transpose of W \nabla_t.
  // The factor of D is because it was missing in k.
  input.AddDiagVecMat(-1.0 * D, k, deriv_w, kNoTrans, 1.0);
}

void NnetPerturbedUpdater::ComputeForMinibatch(
    const std::vector<NnetExample> &data,
    BaseFloat D,
    double *tot_objf_orig,
    double *tot_objf_perturbed) {

  FormatInput(data);
  PropagateInitial();
  PropagateRemaining();
  CuMatrix<BaseFloat> tmp_deriv;

  std::vector<MatrixElement<BaseFloat> > sv_labels;
  ComputeSupervisionLabels(data, &sv_labels);
  
  BaseFloat tot_objf, tot_weight;
  ComputeObjfAndDeriv(sv_labels, &tmp_deriv, &tot_objf, &tot_weight);

  KALDI_VLOG(4) << "Objective function (original) is " << (tot_objf/tot_weight)
                << " per sample, over " << tot_weight << " samples (weighted).";
  *tot_objf_orig = tot_objf;
  
  // only backprops till layer number num_layers_before_input_,
  // and derivative at that layer is in tmp_deriv.
  Backprop(NULL, &tmp_deriv);

  // perturb forward_data_[num_layers_before_input_].
  PerturbInput(tmp_deriv, D);
  
  // Now propagate forward again from that point.
  PropagateRemaining();

  ComputeObjfAndDeriv(sv_labels, &tmp_deriv, &tot_objf, &tot_weight);
  KALDI_VLOG(4) << "Objective function (perturbed) is " << (tot_objf/tot_weight)
                << " per sample, over " << tot_weight << " samples (weighted).";
  *tot_objf_perturbed = tot_objf;

  // The actual model updating would happen in the next call.
  if (nnet_to_update_ != NULL)
    Backprop(nnet_to_update_, &tmp_deriv);
}

void NnetPerturbedUpdater::Propagate(int32 begin_layer, int32 end_layer) {
  static int32 num_times_printed = 0;
  
  for (int32 c = begin_layer; c < end_layer; c++) {
    const Component &component = nnet_.GetComponent(c);
    const CuMatrix<BaseFloat> &input = forward_data_[c];
    CuMatrix<BaseFloat> &output = forward_data_[c+1];
    // Note: the Propagate function will automatically resize the
    // output.
    component.Propagate(chunk_info_out_[c], chunk_info_out_[c+1], input, &output);

    KALDI_VLOG(4) << "Propagating: sum at output of " << c << " is " << output.Sum();
    
    // If we won't need the output of the previous layer for
    // backprop, delete it to save memory.
    bool need_last_output =
        (c>0 && nnet_.GetComponent(c-1).BackpropNeedsOutput()) ||
        component.BackpropNeedsInput();
    if (g_kaldi_verbose_level >= 3 && num_times_printed < 100) {
      KALDI_VLOG(3) << "Stddev of data for component " << c
                    << " for this minibatch is "
                    << (TraceMatMat(forward_data_[c], forward_data_[c], kTrans) /
                        (forward_data_[c].NumRows() * forward_data_[c].NumCols()));
      num_times_printed++;
    }
    if (!need_last_output && c != num_layers_before_input_)
      forward_data_[c].Resize(0, 0); // We won't need this data.
  }
}

void NnetPerturbedUpdater::ComputeSupervisionLabels(
    const std::vector<NnetExample> &data,
    std::vector<MatrixElement<BaseFloat> > *sv_labels) {
  sv_labels->clear();
  sv_labels->reserve(num_chunks_); // We must have at least this many labels.
  for (int32 m = 0; m < num_chunks_; m++) {
    KALDI_ASSERT(data[m].labels.size() == 1 &&
                 "Training code does not currently support multi-frame egs");
    const std::vector<std::pair<int32,BaseFloat> > &labels = data[m].labels[0];
    for (size_t i = 0; i < labels.size(); i++) {
      MatrixElement<BaseFloat> elem = {m, labels[i].first, labels[i].second};
      sv_labels->push_back(elem);
    }
  }  
}

void NnetPerturbedUpdater::ComputeObjfAndDeriv(
    const std::vector<MatrixElement<BaseFloat> > &sv_labels,
    CuMatrix<BaseFloat> *deriv,
    BaseFloat *tot_objf,
    BaseFloat *tot_weight) const {
  int32 num_components = nnet_.NumComponents();  
  deriv->Resize(num_chunks_, nnet_.OutputDim()); // sets to zero.
  const CuMatrix<BaseFloat> &output(forward_data_[num_components]);
  KALDI_ASSERT(SameDim(output, *deriv));
  
  deriv->CompObjfAndDeriv(sv_labels, output, tot_objf, tot_weight);
}


void NnetPerturbedUpdater::Backprop(Nnet *nnet_to_update,
                                    CuMatrix<BaseFloat> *deriv) const {
  // We assume ComputeObjfAndDeriv has already been called.
  for (int32 c = nnet_.NumComponents() - 1; c >= num_layers_before_input_; c--) {
    const Component &component = nnet_.GetComponent(c);
    Component *component_to_update = (nnet_to_update == NULL ? NULL :
                                      &(nnet_to_update->GetComponent(c)));
    const CuMatrix<BaseFloat> &input = forward_data_[c],
        &output = forward_data_[c+1];
    CuMatrix<BaseFloat> input_deriv(input.NumRows(), input.NumCols());
    const CuMatrix<BaseFloat> &output_deriv(*deriv);
    component.Backprop(chunk_info_out_[c] , chunk_info_out_[c+1], input, output, output_deriv,
                       component_to_update, &input_deriv);
    input_deriv.Swap(deriv);
  }
}


void NnetPerturbedUpdater::FormatInput(const std::vector<NnetExample> &data) {
  KALDI_ASSERT(data.size() > 0);
  int32 num_splice = nnet_.LeftContext() + 1 + nnet_.RightContext();
  KALDI_ASSERT(data[0].input_frames.NumRows() >= num_splice);
  
  int32 feat_dim = data[0].input_frames.NumCols(),
         spk_dim = data[0].spk_info.Dim(),
         tot_dim = feat_dim + spk_dim; // we append these at the neural net
                                       // input... note, spk_dim might be 0.
  KALDI_ASSERT(tot_dim == nnet_.InputDim());
  KALDI_ASSERT(data[0].left_context >= nnet_.LeftContext());
  int32 ignore_frames = data[0].left_context - nnet_.LeftContext(); // If
  // the NnetExample has more left-context than we need, ignore some.
  // this may happen in settings where we increase the amount of context during
  // training, e.g. by adding layers that require more context.
  num_chunks_ = data.size();
  
  forward_data_.resize(nnet_.NumComponents() + 1);

  // First copy to a single matrix on the CPU, so we can copy to
  // GPU with a single copy command.
  Matrix<BaseFloat> temp_forward_data(num_splice * num_chunks_,
                                      tot_dim);
  
  for (int32 chunk = 0; chunk < num_chunks_; chunk++) {
    SubMatrix<BaseFloat> dest(temp_forward_data,
                              chunk * num_splice, num_splice,
                              0, feat_dim);

    Matrix<BaseFloat> full_src(data[chunk].input_frames);
    SubMatrix<BaseFloat> src(full_src, ignore_frames, num_splice, 0, feat_dim);
                             
    dest.CopyFromMat(src);
    if (spk_dim != 0) {
      SubMatrix<BaseFloat> spk_dest(temp_forward_data,
                                    chunk * num_splice, num_splice,
                                    feat_dim, spk_dim);
      spk_dest.CopyRowsFromVec(data[chunk].spk_info);
    }
  }
  forward_data_[0].Swap(&temp_forward_data); // Copy to GPU, if being used.
  // TODO : filter out the unnecessary rows from the input
  nnet_.ComputeChunkInfo(num_splice, num_chunks_, &chunk_info_out_);

}



void DoBackpropPerturbed(const Nnet &nnet,
                         int32 num_layers_before_input,
                         const CuMatrix<BaseFloat> &within_class_covar,
                         BaseFloat D,
                         const std::vector<NnetExample> &examples,
                         Nnet *nnet_to_update,
                         double *tot_objf_orig,
                         double *tot_objf_perturbed) {
  
  try {
    NnetPerturbedUpdater updater(nnet, num_layers_before_input,
                                 within_class_covar, nnet_to_update);

    updater.ComputeForMinibatch(examples, D, tot_objf_orig, tot_objf_perturbed);
  } catch (...) {
    KALDI_LOG << "Error doing backprop, nnet info is: " << nnet.Info();
    throw;
  }
}


NnetPerturbedTrainer::NnetPerturbedTrainer(
    const NnetPerturbedTrainerConfig &config,
    const SpMatrix<BaseFloat> &within_class_covar,    
    Nnet *nnet):
    config_(config), nnet_(nnet), logprob_this_phase_(0.0),
    logprob_perturbed_this_phase_(0.0), weight_this_phase_(0.0),
    logprob_total_(0.0), logprob_perturbed_total_(0.0),
    weight_total_(0.0),
    D_(config.initial_d) {
  InitWithinClassCovar(within_class_covar);
  num_phases_ = 0;
  bool first_time = true;
  BeginNewPhase(first_time);
}


// This function is used in class NnetPerturbedTrainer
// and the function DoBackpropPerturbedParallel.
void InitWithinClassCovar(
    const SpMatrix<BaseFloat> &within_class_covar,
    const Nnet &nnet,
    int32 *num_layers_before_input,
    CuMatrix<BaseFloat> *within_class_covar_out) {  

  CuSpMatrix<BaseFloat> orig_covar(within_class_covar);
  *num_layers_before_input = 0;
  KALDI_ASSERT(nnet.NumComponents() > *num_layers_before_input);
  const Component *comp = &(nnet.GetComponent(*num_layers_before_input));
  // Skip over any SpliceComponent that appears at the beginning of
  // the network.
  if (dynamic_cast<const SpliceComponent*>(comp) != NULL)
    (*num_layers_before_input)++;
  
  KALDI_ASSERT(nnet.NumComponents() > *num_layers_before_input);
  comp = &(nnet.GetComponent(*num_layers_before_input));

  const FixedAffineComponent *fa =
      dynamic_cast<const FixedAffineComponent*>(comp);
  if (fa != NULL) {
    (*num_layers_before_input)++;
    const CuMatrix<BaseFloat> &linear_params = fa->LinearParams();
    if (linear_params.NumCols() != orig_covar.NumCols()) {
      KALDI_ERR << "The neural network seems to expect a (spliced) feature "
                << "dimension of " << linear_params.NumCols() << ", but your "
                << "LDA stats have a dimension of " << orig_covar.NumCols();
    }
    CuMatrix<BaseFloat> temp(linear_params.NumRows(), orig_covar.NumRows());
    // temp = linear_params . orig_covar
    temp.AddMatSp(1.0, linear_params, kNoTrans, orig_covar, 0.0);
    within_class_covar_out->Resize(linear_params.NumRows(),
                                   linear_params.NumRows());
    // temp = linear_params . orig_covar . linear_params^T
    within_class_covar_out->AddMatMat(1.0, temp, kNoTrans,
                                      linear_params, kTrans, 0.0);
    // note: this should be symmetric, spot-test it like this:
    KALDI_ASSERT(ApproxEqual(TraceMatMat(*within_class_covar_out,
                                         *within_class_covar_out, kNoTrans),
                             TraceMatMat(*within_class_covar_out,
                                         *within_class_covar_out, kTrans)));
  } else {
    if (comp->InputDim() != orig_covar.NumCols()) {
      KALDI_ERR << "The neural network seems to expect a (spliced) feature "
                << "dimension of " << comp->InputDim() << ", but your "
                << "LDA stats have a dimension of " << orig_covar.NumCols();
    }
    within_class_covar_out->Resize(orig_covar.NumRows(), orig_covar.NumCols());
    within_class_covar_out->CopyFromSp(orig_covar);
  }
}
  


void NnetPerturbedTrainer::InitWithinClassCovar(
    const SpMatrix<BaseFloat> &within_class_covar) {
  kaldi::nnet2::InitWithinClassCovar(within_class_covar, *nnet_,
                                     &num_layers_before_input_,
                                     &within_class_covar_);
}  

void NnetPerturbedTrainer::TrainOnExample(const NnetExample &value) {
  buffer_.push_back(value);
  if (static_cast<int32>(buffer_.size()) == config_.minibatch_size)
    TrainOneMinibatch();
}

void NnetPerturbedTrainer::TrainOneMinibatch() {
  KALDI_ASSERT(!buffer_.empty());

  double tot_objf_orig, tot_objf_perturbed;
  DoBackpropPerturbed(*nnet_, num_layers_before_input_, within_class_covar_, D_,
                      buffer_, nnet_, &tot_objf_orig, &tot_objf_perturbed);

  logprob_this_phase_ += tot_objf_orig;
  logprob_perturbed_this_phase_ += tot_objf_perturbed;
  double weight = TotalNnetTrainingWeight(buffer_);
  UpdateD(tot_objf_orig / weight, tot_objf_perturbed / weight);
  weight_this_phase_ += weight;
  buffer_.clear();
  minibatches_seen_this_phase_++;
  if (minibatches_seen_this_phase_ == config_.minibatches_per_phase) {
    bool first_time = false;
    BeginNewPhase(first_time);
  }
}


void NnetPerturbedTrainer::UpdateD(BaseFloat orig_objf_per_example,                                   
                                   BaseFloat perturbed_objf_per_example) {
  
  BaseFloat diff = orig_objf_per_example - perturbed_objf_per_example;
  // note: diff should be positive in the normal case.
  KALDI_ASSERT(config_.target_objf_change > 0.0 && config_.max_d_factor > 1.0);
  BaseFloat objf_ratio = config_.target_objf_change /
      std::max<BaseFloat>(1.0e-20, diff),
      D_ratio = pow(objf_ratio, config_.tune_d_power);
  if (D_ratio > config_.max_d_factor)
    D_ratio = config_.max_d_factor;
  else if (D_ratio < 1.0 / config_.max_d_factor)
    D_ratio = 1.0 / config_.max_d_factor;
  BaseFloat D_new = D_ * D_ratio;
  
  KALDI_VLOG(3) << "Training objective function normal/perturbed is "
                << orig_objf_per_example << '/' << perturbed_objf_per_example
                << ", diff " << diff << " vs. target "
                << config_.target_objf_change
                << ", changing D by factor " << D_ratio << " to " << D_new;
  D_ = D_new;  
}

void NnetPerturbedTrainer::BeginNewPhase(bool first_time) {
  if (!first_time) {
    BaseFloat logprob = logprob_this_phase_/weight_this_phase_,
        logprob_perturbed = logprob_perturbed_this_phase_/weight_this_phase_,
        diff = logprob - logprob_perturbed;
    KALDI_LOG << "Training objective function normal->perturbed is "
              << logprob << " -> " << logprob_perturbed << ", diff "
              << diff << " vs. target " << config_.target_objf_change
              << ", over " << weight_this_phase_ << " frames, D is "
              << D_;
  }
  logprob_total_ += logprob_this_phase_;
  logprob_perturbed_total_ += logprob_perturbed_this_phase_;
  weight_total_ += weight_this_phase_;
  logprob_this_phase_ = 0.0;
  logprob_perturbed_this_phase_ = 0.0;
  weight_this_phase_ = 0.0;
  minibatches_seen_this_phase_ = 0;
  num_phases_++;
}


NnetPerturbedTrainer::~NnetPerturbedTrainer() {
  if (!buffer_.empty()) {
    KALDI_LOG << "Doing partial minibatch of size "
              << buffer_.size();
    TrainOneMinibatch();
    if (minibatches_seen_this_phase_ != 0) {
      bool first_time = false;
      BeginNewPhase(first_time);
    }
  }
  if (weight_total_ == 0.0) {
    KALDI_WARN << "No data seen.";
  } else {
    KALDI_LOG << "Did backprop on " << weight_total_
              << " examples, average log-prob normal->perturbed per frame is "
              << (logprob_total_ / weight_total_) << " -> "
              << (logprob_perturbed_total_ / weight_total_);
    KALDI_LOG << "[this line is to be parsed by a script:] log-prob-per-frame="
              << (logprob_total_ / weight_total_);
  }
}


// compare with DoBackpropParallelClass
class TrainParallelPerturbedClass: public MultiThreadable {
 public:
  // This constructor is only called for a temporary object
  // that we pass to the RunMultiThreaded function.
  TrainParallelPerturbedClass(const NnetPerturbedTrainerConfig &config,
                              const CuMatrix<BaseFloat> &within_class_covar,
                              int32 num_layers_before_input,
                              BaseFloat *D,
                              Nnet *nnet,
                              ExamplesRepository *repository,
                              double *log_prob_orig_ptr,
                              double *log_prob_perturbed_ptr,
                              double *tot_weight_ptr):
      config_(config), within_class_covar_(within_class_covar),
      num_layers_before_input_(num_layers_before_input), D_(D),
      nnet_(nnet), repository_(repository),
      log_prob_orig_ptr_(log_prob_orig_ptr),
      log_prob_perturbed_ptr_(log_prob_perturbed_ptr),
      tot_weight_ptr_(tot_weight_ptr),
      log_prob_orig_(0.0),
      log_prob_perturbed_(0.0),
      tot_weight_(0.0) { }

  // Use the default copy constructor.
  
  // This does the main function of the class.
  void operator () () {
    std::vector<NnetExample> examples;
    while (repository_->ProvideExamples(&examples)) {
      double objf_orig, objf_perturbed,
          weight = TotalNnetTrainingWeight(examples);
      DoBackpropPerturbed(*nnet_, num_layers_before_input_,
                          within_class_covar_, *D_,
                          examples, nnet_,
                          &objf_orig, &objf_perturbed);
      UpdateD(objf_orig / weight, objf_perturbed / weight);
      
      tot_weight_ += weight;
      log_prob_orig_ += objf_orig;
      log_prob_perturbed_ += objf_perturbed;
      KALDI_VLOG(4) << "Thread " << thread_id_ << " saw "
                    << tot_weight_ << " frames so far (weighted); likelihood "
                    << "per frame (orig->perturbed) so far is "
                    << (log_prob_orig_ / tot_weight_) << " -> "
                    << (log_prob_perturbed_ / tot_weight_);
      examples.clear();
    }    
  }
  
  ~TrainParallelPerturbedClass() {
    *log_prob_orig_ptr_ += log_prob_orig_;
    *log_prob_perturbed_ptr_ += log_prob_perturbed_;
    *tot_weight_ptr_ += tot_weight_;
  }
 private:
  void UpdateD(BaseFloat orig_logprob, BaseFloat perturbed_logprob) {
    BaseFloat diff = orig_logprob - perturbed_logprob;
    // note: diff should be positive in the normal case.
    KALDI_ASSERT(config_.target_objf_change > 0.0 && config_.max_d_factor > 1.0);
    // divide the power we raise the ratio to when tuning D, by the
    // number of threads; this should ensure stability of the update.
    BaseFloat tune_d_power = config_.tune_d_power / g_num_threads;
    BaseFloat objf_ratio = config_.target_objf_change /
        std::max<BaseFloat>(1.0e-20, diff),
        D_ratio = pow(objf_ratio, tune_d_power);
    if (D_ratio > config_.max_d_factor)
      D_ratio = config_.max_d_factor;
    else if (D_ratio < 1.0 / config_.max_d_factor)
      D_ratio = 1.0 / config_.max_d_factor;
    BaseFloat D_new = (*D_) * D_ratio;
    *D_ = D_new;  
    
    // Note: we are accessing *D_ from multiple threads without
    // locking, but the negative consequences of this contention are
    // very small (
    KALDI_VLOG(3) << "Training objective function normal->perturbed is "
                  << orig_logprob << " -> " << perturbed_logprob
                  << ", diff " << diff << " vs. target "
                  << config_.target_objf_change
                  << ", changing D by factor " << D_ratio << " to " << D_new;
  }

  const NnetPerturbedTrainerConfig &config_;
  const CuMatrix<BaseFloat> &within_class_covar_;
  int32 num_layers_before_input_;
  BaseFloat *D_;  // Constant D that controls how much to perturb the data.  We
                  // update this as well as use it.
  Nnet *nnet_;
  ExamplesRepository *repository_;

  double *log_prob_orig_ptr_;
  double *log_prob_perturbed_ptr_;
  double *tot_weight_ptr_;
  double log_prob_orig_;  // log-like times num frames (before perturbing features)
  double log_prob_perturbed_;  // log-like times num frames (after perturbing features)
  double tot_weight_;  // normalizing factor for the above.
};

void DoBackpropPerturbedParallel(const NnetPerturbedTrainerConfig &config,
                                 const SpMatrix<BaseFloat> &within_class_covar,
                                 SequentialNnetExampleReader *example_reader,
                                 double *tot_objf_orig,
                                 double *tot_objf_perturbed,
                                 double *tot_weight,
                                 Nnet *nnet) {

  // within_class_covar_processed is the within-class covar as CuMatrix, possibly
  // projected by the preconditioning transform in any FixedAffineComponent.
  CuMatrix<BaseFloat> within_class_covar_processed;
  int32 num_layers_before_input;
  InitWithinClassCovar(within_class_covar, *nnet,
                       &num_layers_before_input,
                       &within_class_covar_processed);
  BaseFloat D = config.initial_d;

  ExamplesRepository repository; // handles parallel programming issues regarding  

  *tot_objf_orig = *tot_objf_perturbed = *tot_weight = 0.0;

  TrainParallelPerturbedClass trainer_proto(config,
                                            within_class_covar_processed,
                                            num_layers_before_input, &D,
                                            nnet, &repository,
                                            tot_objf_orig,
                                            tot_objf_perturbed,
                                            tot_weight);

  {
    // The initialization of the following class spawns the threads that
    // process the examples.  They get re-joined in its destructor.
    MultiThreader<TrainParallelPerturbedClass> m(g_num_threads, trainer_proto);
    
    std::vector<NnetExample> examples;
    for (; !example_reader->Done(); example_reader->Next()) {
      examples.push_back(example_reader->Value());
      if (examples.size() == config.minibatch_size)
        repository.AcceptExamples(&examples);
    }
    if (!examples.empty()) // partial minibatch.
      repository.AcceptExamples(&examples);
    // Here, the destructor of "m" re-joins the threads, and
    // does the summing of the gradients if we're doing gradient
    // computation (i.e. &nnet != nnet_to_update).  This gets
    // done in the destructors of the objects of type
    // DoBackpropParallelClass.
    repository.ExamplesDone();
  }
  KALDI_LOG << "Did backprop on " << *tot_weight << " examples, average log-prob "
            << "per frame (orig->perturbed) is "
            << (*tot_objf_orig / *tot_weight) << " -> "
            << (*tot_objf_perturbed / *tot_weight) << " over "
            << *tot_weight << " samples (weighted).";
  
  KALDI_LOG << "[this line is to be parsed by a script:] log-prob-per-frame="
            << (*tot_objf_orig / *tot_weight);
}




} // namespace nnet2
} // namespace kaldi
