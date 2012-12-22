// nnet/mixup-nnet.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet-cpu/mixup-nnet.h"
#include "gmm/model-common.h" // for GetSplitTargets()
#include <numeric> // for std::accumulate

namespace kaldi {

static BaseFloat GetFirstLearningRate(const Nnet &nnet) {
  for (int32 c = 0; c < nnet.NumComponents(); c++) {
    const UpdatableComponent *uc =
        dynamic_cast<const UpdatableComponent*>(&(nnet.GetComponent(c)));
    if (uc != NULL)
      return uc->LearningRate();
  }
  KALDI_ERR << "Neural net has no updatable components";
  return 0.0;
}


/** This function makes sure the neural net ends with a
    MixtureProbComponent.  If it doesn't, it adds one
    (with a single mixture/matrix corresponding to each
    output element.)  [Before doing so, it makes sure
    that the last layer is a SoftmaxLayer, which is what
    we expect.  You can remove this check if there is some
    use-case that makes sense where the type of the previous
    layer is different.
 */
static void GiveNnetCorrectTopology(Nnet *nnet,
                                    AffineComponent **affine_component,
                                    SoftmaxComponent **softmax_component,
                                    MixtureProbComponent **mixture_prob_component) {
  int32 nc = nnet->NumComponents();
  KALDI_ASSERT(nc > 0);
  Component* component = &(nnet->GetComponent(nc - 1));
  if ((*mixture_prob_component =
       dynamic_cast<MixtureProbComponent*>(component)) == NULL) {
    KALDI_LOG << "Adding MixtureProbComponent to neural net.";
    int32 dim = component->OutputDim();
    // Give it the same learning rate as the first updatable layer we have.
    BaseFloat learning_rate = GetFirstLearningRate(*nnet),
        diag_element = 1.0; // actually it's a don't care.
    std::vector<int32> sizes(dim, 1); // a vector of all ones, of dimension "dim".
  
    *mixture_prob_component = new MixtureProbComponent();
    (*mixture_prob_component)->Init(learning_rate,
                                    diag_element,
                                    sizes);
    nnet->Append(*mixture_prob_component);
    nc++;
  }
  component = &(nnet->GetComponent(nc - 2));    
  if ((*softmax_component = dynamic_cast<SoftmaxComponent*>(component)) == NULL)
    KALDI_ERR << "Neural net has wrong topology: expected second-to-last "
              << "component to be SoftmaxComponent, type is "
              << component->Type();
  component = &(nnet->GetComponent(nc - 3));
  if ((*affine_component = dynamic_cast<AffineComponent*>(component)) == NULL)
    KALDI_ERR << "Neural net has wrong topology: expected third-to-last "
              << "component to be AffineComponent, type is "
              << component->Type();
}


/**
   This function works as follows.
   We first make sure the neural net has the correct topology, so its
   last component is a MixtureProbComponent.

   We then get the counts for each matrix in the MixtureProbComponent (these
   will either correspond to leaves in the decision tree, or level-1 leaves, if
   we have a 2-level-tree system).  We work out the total count for each of these
   matrices, by getting the count from the SoftmaxComponent.
   
   Then, for each matrix in the Mixturemixture-prob component, we
 */


void MixupNnetInternal(SoftmaxComponent *softmax,
                       MixtureProbComponent *mixture_prob) {
  
}

void MixupNnet(const NnetMixupConfig &mixup_config,
               Nnet *nnet) {
  AffineComponent *affine_component = NULL;
  SoftmaxComponent *softmax_component = NULL;
  MixtureProbComponent *mixture_prob_component = NULL;
  GiveNnetCorrectTopology(nnet,
                          &affine_component,
                          &softmax_component,
                          &mixture_prob_component); // Adds a MixtureProbComponent if needed.
  
  softmax_component->MixUp(mixup_config.num_mixtures,
                           mixup_config.power,
                           mixup_config.min_count,
                           mixup_config.perturb_stddev,
                           affine_component,
                           mixture_prob_component);
  nnet->Check(); // Checks that dimensions all match up.
}


/// Allocate mixtures to states via a power rule, and add any new mixtures.
void SoftmaxComponent::MixUp(int32 num_mixtures,
                             BaseFloat power,
                             BaseFloat min_count,
                             BaseFloat perturb_stddev, 
                             AffineComponent *ac,
                             MixtureProbComponent *mc) {
  
  // "counts" is derived from this->counts_ by summing.
  Vector<BaseFloat> counts(mc->params_.size());
  int32 old_dim = 0;
  for (size_t i = 0; i < mc->params_.size(); i++) {
    int32 this_input_dim = mc->params_[i].NumCols();
    BaseFloat this_tot_count = 0.0; /// Total the count out of
    /// all the output dims of the softmax layer that correspond
    /// to this mixture.  We'll use this total to allocate new quasi-Gaussians.
    for (int32 d = 0; d < this_input_dim; d++, old_dim++)
      this_tot_count += this->counts_(old_dim);
    counts(i) = this_tot_count;
  }
  KALDI_ASSERT(old_dim == counts_.Dim());
  KALDI_ASSERT(counts.Sum() > 0 && "Cannot do mixing up without counts.");

  std::vector<int32> targets; // #mixtures for each state.

  // Get the target number of mixtures for each state.
  GetSplitTargets(counts, num_mixtures, power, min_count, &targets);
  KALDI_ASSERT(targets.size() == mc->params_.size());
  // floor each target to the current #mixture components.
  for (size_t i = 0; i < targets.size(); i++)
    targets[i] = std::max(targets[i], mc->params_[i].NumCols());
  int32 new_dim = std::accumulate(targets.begin(), targets.end(),
                                  static_cast<int32>(0)),
      affine_input_dim = ac->InputDim();
  KALDI_ASSERT(new_dim >= old_dim);
  
  // bias and linear terms from affine component:
  const Vector<BaseFloat> &old_bias_term(ac->bias_params_);
  const Matrix<BaseFloat> &old_linear_term(ac->linear_params_);
  
  Vector<BaseFloat> new_bias_term(new_dim);
  Matrix<BaseFloat> new_linear_term(new_dim, affine_input_dim);
  Vector<BaseFloat> new_counts(new_dim);

  // old_offset and new_offset are offsets into the dimension at the
  // input/output of the softmax component, before and after mixing up
  // respectively.  They get incremented in the following loop.
  int32 old_offset = 0, new_offset = 0;
  for (size_t i = 0; i < mc->params_.size(); i++) {
    const Matrix<BaseFloat> &this_old_params(mc->params_[i]);
    int32 this_old_dim = this_old_params.NumCols(),
        this_new_dim = targets[i],
        this_cur_dim = this_old_dim; // this_cur_dim is loop variable.
    
    SubMatrix<BaseFloat> this_old_linear_term(old_linear_term,
                                              old_offset, this_old_dim,
                                              0, affine_input_dim),
        this_new_linear_term(new_linear_term,
                             new_offset, this_new_dim,
                             0, affine_input_dim);
    SubVector<BaseFloat> this_old_bias_term(old_bias_term,
                                            old_offset, this_old_dim),
        this_new_bias_term(new_bias_term, new_offset, this_new_dim),
        this_old_counts(this->counts_,
                        old_offset, this_old_dim),
        this_new_counts(new_counts,
                        new_offset, this_new_dim);
    Matrix<BaseFloat> this_new_params(this_old_params.NumRows(),
                                      this_new_dim);
    
    // Copy the same-dimensional part of the parameters and counts.
    this_new_linear_term.Range(0, this_old_dim, 0, affine_input_dim).
        CopyFromMat(this_old_linear_term);
    this_new_bias_term.Range(0, this_old_dim).
        CopyFromVec(this_old_bias_term);
    this_new_counts.Range(0, this_old_dim).
        CopyFromVec(this_old_counts);
    // this_new_params is the mixture weights.
    this_new_params.Range(0, this_old_params.NumRows(), 0, this_old_dim).
        CopyFromMat(this_old_params);
    // Add the new components...
    for (; this_cur_dim < this_new_dim; this_cur_dim++) {
      BaseFloat *count_begin = this_new_counts.Data(),
          *count_end  = count_begin + this_cur_dim,
          *count_max = std::max_element(count_begin, count_end);
      KALDI_ASSERT(*count_max > 0.0);
      *count_max *= 0.5;
      *count_end = *count_max; // count for the element we're adding.
      int32 max_index = static_cast<int32>(count_max - count_begin),
          new_index = this_cur_dim;
      SubVector<BaseFloat> cur_vec(this_new_linear_term, max_index),
          new_vec(this_new_linear_term, new_index);
      new_vec.CopyFromVec(cur_vec);
      Vector<BaseFloat> rand(affine_input_dim);
      rand.SetRandn();
      cur_vec.AddVec(perturb_stddev, rand);
      new_vec.AddVec(-perturb_stddev, rand);
      this_new_bias_term(max_index) += log(0.5);
      this_new_bias_term(new_index) = this_new_bias_term(max_index);
      // now copy the column of the MixtureProbComponent parameters.
      for (int32 j = 0; j < this_new_params.NumRows(); j++)
        this_new_params(j, new_index) = this_new_params(j, max_index);
    }
    old_offset += this_old_dim;
    new_offset += this_new_dim;
    mc->params_[i] = this_new_params;
  }
  KALDI_ASSERT(old_offset == old_dim && new_offset == new_dim);
  ac->SetParams(new_bias_term, new_linear_term);
  this->counts_ = new_counts;
  this->dim_ = new_dim;
  mc->input_dim_ = new_dim; // keep this up to date.
  // We already updated mc->params_.
  KALDI_LOG << "Mixed up from dimension of " << old_dim << " to " << new_dim
            << " in the softmax layer.";
}



  
} // namespace
