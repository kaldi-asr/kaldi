// durmod/kaldi-durmod.h

// Copyright 2015, Hossein Hadian

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
#ifndef DURMOD_KALDI_DURMOD_H_
#define DURMOD_KALDI_DURMOD_H_

#include <iostream>
#include <vector>
#include <utility>
#include <string>

#include "base/kaldi-common.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-example-utils.h"
#include "nnet3/nnet-optimize.h"
#include "util/common-utils.h"
#include "util/stl-utils.h"
#include "fstext/deterministic-fst.h"

namespace kaldi {
  using nnet3::Nnet;
  using nnet3::NnetExample;
  using nnet3::NnetIo;
  using nnet3::CachingOptimizingCompiler;
  using nnet3::ComputationRequest;
  using nnet3::NnetComputation;
  using nnet3::NnetComputeOptions;
  using nnet3::NnetComputer;

struct PhoneDurationModelOptions {
  int32 left_context, right_context;
  int32 max_duration;

  void Register(OptionsItf *po);

  PhoneDurationModelOptions():
    left_context(3),
    right_context(2),
    max_duration(50) {}
};

class PhoneDurationModel {
  friend class PhoneDurationEgsMaker;
  friend class PhoneDurationModelDeterministicFst;
  friend class PhoneDurationScoreComputer;

 public:
  void Read(std::istream &is, bool binary);
  void Write(std::ostream &os, bool binary) const;

  PhoneDurationModel() {}
  PhoneDurationModel(const PhoneDurationModelOptions &opts,
                     const std::vector<std::vector<int32> > &roots,
                     const std::vector<std::vector<int32> > &questions):
    left_context_(opts.left_context),
    right_context_(opts.right_context),
    roots_(roots),
    questions_(questions),
    max_duration_(opts.max_duration) {}

  void InitNnet(int input_dim, int dim1,
                int dim2, int output_dim);
  inline int32 MaxDuration() const { return max_duration_; }
  inline int32 RightContext() const { return right_context_; }
  inline int32 LeftContext() const { return left_context_; }
  inline int32 FullContextSize() const {
    return left_context_ + right_context_ + 1;
  }
  const Nnet &GetNnet() const { return nnet_; }
  Nnet &GetNnet() { return nnet_; }
  void SetNnet(const Nnet &nnet) { nnet_ = nnet; }

 private:
  Nnet nnet_;

  // info related to generating features
  int32 left_context_, right_context_;
  std::vector<std::vector<int32> > roots_;
  std::vector<std::vector<int32> > questions_;
  int32 max_duration_;
};

class PhoneDurationEgsMaker {
 public:
  explicit PhoneDurationEgsMaker(const PhoneDurationModel &model);

  void InitFeatureMaker(const PhoneDurationModel &model);

  void AlignmentToNnetExamples(
      const std::vector<std::pair<int32, int32> > &alignment,
      std::vector<NnetExample> *egs) const;

  void MakeFeatureVector(
                    const std::vector<std::pair<int32, int32> > &phone_context,
                    int phone_index,
                    SparseVector<BaseFloat> *feat) const;

  void MakeNnetExample(
                    const std::vector<std::pair<int32, int32> > &phone_context,
                    int phone_index,
                    NnetExample *eg) const;
  int32 FeatureDim() const { return feature_dim_; }
  int32 NumBinaryFeatures() const { return num_binary_features_; }
  int32 NumPhoneIdentities() const { return num_phone_identities_; }
  int32 OutputDim() const { return max_duration_; }

 private:
  unordered_map<int32, std::vector<int32> > binary_feats_;
  unordered_map<int32, int32> phone_id_;
  int32 num_binary_features_;
  int32 num_phone_identities_;
  int32 left_context_, right_context_;
  int32 max_duration_;
  int32 feature_dim_;

  /// Normalizes a duration (measured in number of frames) to a float in the
  /// range [0,1] suitable for neural network training
  BaseFloat NormalizeDuration(int32 duration_in_frames) const;
};

class PhoneDurationScoreComputer {
 public:
  explicit PhoneDurationScoreComputer(const PhoneDurationModel &model):
      model_(model), compiler_(model.GetNnet()), egs_maker_(model) {}

  void ComputeOutputForExample(const NnetExample &eg,
                               Matrix<BaseFloat> *output);

  /// Computes the log prob for the middle phone (middle in the left/right
  /// context sense) in a phone context.
  BaseFloat
         GetLogProb(const std::vector<std::pair<int32, int32> > &phone_context);

 private:
  const PhoneDurationModel &model_;
  CachingOptimizingCompiler compiler_;
  PhoneDurationEgsMaker egs_maker_;
};

class PhoneDurationModelDeterministicFst
  : public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;
  
  // second argument is non-cost because it has a cache
  PhoneDurationModelDeterministicFst(const PhoneDurationModel &model,
                                     PhoneDurationScoreComputer *scorer);

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  StateId Start() { return start_state_; }

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  Weight Final(StateId s);

  bool GetArc(StateId s, Label ilabel, fst::StdArc* oarc);

 private:
  typedef unordered_map<std::vector<Label>,
    StateId, kaldi::VectorHasher<Label> > MapType;

  StateId start_state_;
  MapType context_to_state_;
  std::vector<std::vector<Label> > state_to_context_;
  int32 context_size_, right_context_;
  int32 max_duration_;
  PhoneDurationScoreComputer &scorer_;

  BaseFloat GetLogProb(const std::vector<Label> &context) const;
};

/// This function decodes an already encoded olabel to phone id and duration.
std::pair<int32, int32> OlabelToPhoneAndDuration(fst::StdArc::Label olabel,
                                                 int32 max_duration);
/// This function encodes two integers (phone-id and duration) to one integer
fst::StdArc::Label PhoneAndDurationToOlabel(int32 phone,
                                            int32 duration,
                                            int32 max_duration);
/// This function replaces the output labels of a compact lattice (which has
/// already been phone-aligned and has only 1 phone per arc) with the duration
/// and identity of the phones. This is done before composing with the on-demand
/// fst so that the fst can know the durations and phone-ids to compute the
/// log probabilities.
void DurationModelReplaceLabelsLattice(CompactLattice *in_clat,
                                       const TransitionModel &tmodel,
                                       int32 max_duration);
/// This function reverses the effects of the previous function and puts back
/// the original output labels.
void DurationModelReplaceLabelsBackLattice(CompactLattice *in_clat);

}  // namespace kaldi
#endif  // DURMOD_KALDI_DURMOD_H_

