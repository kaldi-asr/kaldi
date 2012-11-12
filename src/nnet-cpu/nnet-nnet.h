// nnet-cpu/nnet-nnet.h

// Copyright 2011-2012  Karel Vesely
//                      Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET_CPU_NNET_NNET_H_
#define KALDI_NNET_CPU_NNET_NNET_H_

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"
#include "nnet-cpu/nnet-component.h"

#include <iostream>
#include <sstream>
#include <vector>


namespace kaldi {


/*
  This neural net is basically a series of Components, and is a fairly
  passive object that mainly acts as a store for the Components.  Training
  is handled by a separate class NnetTrainer(), and extracting likelihoods
  for decoding is handled by DecodableNnetCpu(). 
  
  There are a couple of things that make this class more than just a vector of
  Components.

   (1) It handles frame splicing (temporal context.)
   We'd like to encompass the approach described in
   http://www.fit.vutbr.cz/research/groups/speech/publi/2011/vesely_asru2011_00042.pdf
   where at a certain point they splice together frames -10, -5, 0, +5 and +10.  It
   seems that it's not necessarily best to splice together a contiguous sequence
   of frames.

   (2) It handles situations where the input features have two parts--
   a "frame-specific" part (the normal features), and a "speaker-specific", or at
   least utterance-specific part that does not vary with the frame index.
   These features are provided separately from the frame-specific ones, to avoid
   redundancy.
*/


class Nnet {
 public:
  /// Returns number of components-- think of this as similar to # of layers, but
  /// e.g. the nonlinearity and the linear part count as separate components,
  /// so the number of components will be more than the number of layers.
  int32 NumComponents() const { return components_.size(); }

  const Component &GetComponent(int32 component) const;

  Component &GetComponent(int32 component);
  
  /// Returns the LeftContext() summed over all the Components... this is the
  /// entire left-context in frames that the network requires.
  int32 LeftContext() const;

  /// Returns the LeftContext() summed over all the Components... this is the
  /// entire left-context in frames that the network requires.
  int32 RightContext() const;
  
  /// The output dimension of the network -- typically
  /// the number of pdfs.
  int32 OutputDim() const;

  /// Dimension of the input features, e.g. 13 or 40.  Does not
  /// take account of frame splicing-- that is done with the "chunk"
  /// mechanism, where you provide chunks of features over time.
  int32 InputDim() const; 
  
  void ZeroOccupancy(); // calls ZeroOccupancy() on the softmax layers.  This
  // resets the occupancy counters; it makes sense to do this once in
  // a while, e.g. at the start of an epoch of training.
  
  Nnet(const Nnet &other); // Copy constructor.
  
  Nnet() { }

  Nnet &operator = (const Nnet &other); // assignment operator.

  /// Initialize from config file.
  /// Each line of the config is either a comment line starting
  /// with whitespace then #, or it is a line that specifies one
  /// layer of the network, as accepted by Component::InitFromString().
  /// An example non-comment line is:
  /// AffineComponent learning-rate=0.01 l2-penalty=0.001 input-dim=10 output-dim=15 param-stddev=0.1
  void Init(std::istream &is);

  /// This Init method works from a vector of components.  It will take ownership
  /// of the pointers and resize the vector to zero to avoid a chance of the
  /// caller deallocating them.
  void Init(std::vector<Component*> *components);
  
  ~Nnet() { Destroy(); }

  /*  
  // Add a new tanh layer (hidden layer).  
  // Use #nodes of top hidden layer.  The new layer will have zero-valued parameters
  void AddTanhLayer(int32 left_context, int32 right_context,
                    BaseFloat learning_rate);
  */

  std::string Info() const; // some human-readable summary info.


  /*
  std::string LrateInfo() const; // some info on the learning rates,
  // in human-readable form.

  // the same, broken down by sets.
  std::string LrateInfo(const std::vector<std::vector<int32> > &final_sets)
      const;

  // the same, broken down by sets, for shrinkage rates.
  std::string SrateInfo(const std::vector<std::vector<int32> > &final_sets)
      const;
  // Mix up by increasing the dimension of the output of softmax layer (and the
  // input of the linear layer).  This is exactly analogous to mixing up
  // Gaussians in a GMM-HMM system, and we use a similar power rule to allocate
  // new ones [so a "category" gets an allocation of indices/Gaussians allocated
  // proportional to a power "power" of its total occupancy.
  void MixUp(int32 target_tot_neurons,
             BaseFloat power, // e.g. 0.2.
             BaseFloat perturb_stddev);
             
  void Init(const Nnet1InitInfo &init_info);
*/
  void Destroy();
  
  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  void SetZero(bool treat_as_gradient); // Sets all parameters to zero and if
  // treat_as_gradient == true, also sets the learning rates to 1.0 and shinkage
  // rates to zero and instructs the components to think of themselves as
  // storing the gradient (this part only affects components of type
  // LinearComponent).


  /// This is used to separately adjust learning rates of each layer,
  /// after each "phase" of training.  We basically ask (using the validation
  /// gradient), do we wish we had gone further in this direction?  Yes->
  /// increase learning rate, no -> decrease it. 
  void AdjustLearningRates(
      const VectorBase<BaseFloat> &old_model_old_gradient,
      const VectorBase<BaseFloat> &new_model_old_gradient,
      const VectorBase<BaseFloat> &old_model_new_gradient,
      const VectorBase<BaseFloat> &new_model_new_gradient,
      BaseFloat measure_at, // where to measure gradient, on line between old
                            // and new model; 0.5 < measure_at <= 1.0.
      BaseFloat learning_rate_ratio,
      BaseFloat max_learning_rate);
  
  // This sets *dot_prod to the dot prod of *this . validation_gradient,
  // separately for each component; zero for non-updatable components.
  // (The vector must have size equal to this->NumComponents()).
  // This is used in updating learning rates and shrinkage rates.
  void ComponentDotProducts(
      const Nnet &other,
      VectorBase<BaseFloat> *dot_prod) const;

  void Check() const; // Consistency check.
  
  friend class NnetUpdater;
  friend class DecodableNnet;
 private:
  std::vector<Component*> components_;
};



} // namespace kaldi

#endif


