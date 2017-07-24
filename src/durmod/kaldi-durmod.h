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
#include "lat/kaldi-lattice.h"
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

  void Register(OptionsItf *opts) {
    opts->Register("left-context", &left_context,
                   "Number of left context frames");
    opts->Register("right-context", &right_context,
                   "Number of right context frames");
  }
  PhoneDurationModelOptions():
    left_context(4),
    right_context(2) {}
};

struct NnetPhoneDurationModelOptions {
  int32 num_mixture_components;
  int32 max_duration;

  void Register(OptionsItf *opts) {
    opts->Register("num-mixture-components", &num_mixture_components,
                   "Number of mixture components to be used in the output " 
                   "log-normal distribution of the network.");
    opts->Register("max-duration", &max_duration,
                   "Max phone duration in frames. Durations longer than this "
                   "will be mapped to this value. Set it to 0 to enable "
                   "log-normal objective for the neural net instead of "
                   "cross-entropy.");
  }
  NnetPhoneDurationModelOptions():
    num_mixture_components(1),
    max_duration(50) {}
};

class PhoneDurationModel {
  friend class PhoneDurationFeatureMaker;

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
    questions_(questions) {}

  inline int32 RightContext() const { return right_context_; }
  inline int32 LeftContext() const { return left_context_; }
  inline int32 FullContextSize() const {
    return left_context_ + right_context_ + 1;
  }
  std::string Info() const;

 private:
  int32 left_context_, right_context_;
  std::vector<std::vector<int32> > roots_;
  std::vector<std::vector<int32> > questions_;
};

class PhoneDurationFeatureMaker {
 public:
  explicit PhoneDurationFeatureMaker(const PhoneDurationModel &model);

  void InitFeatureMaker(const PhoneDurationModel &model);

  /// This method extracts features for a phone. The inputs are a
  /// phone-duration context and an index into that context which indicates the
  /// middle phone (middle in the sense of left/right context).
  /// a phone-duration context is a sequence of <phone-ID, duration-in-frames>
  /// pairs.
  /// The extracted features consist of phone-IDs (which are represented in a
  /// 1-of-n encoding, suitable for neural networks), phone durations, and
  /// binary features determined from a set of questions
  /// (i.e. extra_questions.int)
  void MakeFeatureVector(
                 const std::vector<std::pair<int32, int32> > &phone_dur_context,
                 int phone_index,
                 SparseVector<BaseFloat> *feat) const;


  int32 FeatureDim() const { return feature_dim_; }
  int32 NumBinaryFeatures() const { return num_binary_features_; }
  int32 NumPhoneIdentities() const { return num_phone_identities_; }

  std::string Info() const;

 private:
  unordered_map<int32, std::vector<int32> > binary_feats_;
  unordered_map<int32, int32> phone_id_;
  int32 num_binary_features_;
  int32 num_phone_identities_;
  int32 left_context_, right_context_;
  int32 feature_dim_;

  /// Normalizes a duration (measured in number of frames) to a float in the
  /// range [0,1] suitable for neural network training
  BaseFloat NormalizeDuration(int32 duration_in_frames) const;
};

class NnetPhoneDurationModel {
 public:
  void Read(std::istream &is, bool binary);
  void Write(std::ostream &os, bool binary) const;

  NnetPhoneDurationModel() {}
  NnetPhoneDurationModel(const NnetPhoneDurationModelOptions &opts,
                         const PhoneDurationModel &duration_model,
                         const Nnet &nnet):
                         dur_model_(duration_model),
                         nnet_(nnet),
                         opts_(opts) {}

  inline int32 RightContext() const { return dur_model_.RightContext(); }
  inline int32 LeftContext() const { return dur_model_.LeftContext(); }
  inline int32 FullContextSize() const { return dur_model_.FullContextSize(); }
  inline bool IsNnetObjectiveLogNormal() const {
    return opts_.max_duration == 0;
  }
  inline int32 MaxDuration() const { return opts_.max_duration; }
  inline int32 NumMixtureComponents() const {
    return opts_.num_mixture_components;
  }

  std::string Info() const;
  const PhoneDurationModel &GetDurationModel() const { return dur_model_; }
  PhoneDurationModel &GetDurationModel() { return dur_model_; }

  const Nnet &GetNnet() const { return nnet_; }
  Nnet &GetNnet() { return nnet_; }
  void SetNnet(const Nnet &nnet) { nnet_ = nnet; }

 private:
  PhoneDurationModel dur_model_;
  Nnet nnet_;
  NnetPhoneDurationModelOptions opts_;
};

class AvgPhoneLogProbs {
 public:
  AvgPhoneLogProbs
  (int32 left_context, int32 right_context,
   const unordered_map<std::vector<int32>, BaseFloat, VectorHasher<int32> >
                                                      &context_to_avglogprob): 
  left_context_(left_context), right_context_(right_context),
  context_to_avglogprob_(context_to_avglogprob) {}
  AvgPhoneLogProbs() {}

  static std::string PhoneContext2Str(const std::vector<int32>& ctx) {
    std::stringstream ss;
    ss << "[";
    for (int i = 0; i < ctx.size(); i++) {
      ss << ctx[i];
      if (i < ctx.size() - 1)
        ss << ",";
    }
    ss << "]";
    return ss.str();
  }
  void Write(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<AvgPhoneLogProbs>");
    WriteBasicType(os, binary, static_cast<int32>(context_to_avglogprob_.size()));
    for (unordered_map<std::vector<int32>, BaseFloat, VectorHasher<int32> >::const_iterator
          it = context_to_avglogprob_.begin(); it != context_to_avglogprob_.end(); ++it) {
      WriteIntegerVector(os, binary, it->first);
      WriteBasicType(os, binary, it->second);
    }
    WriteToken(os, binary, "<LeftContext>");
    WriteBasicType(os, binary, left_context_);
    WriteToken(os, binary, "<RightContext>");
    WriteBasicType(os, binary, right_context_);
    WriteToken(os, binary, "</AvgPhoneLogProbs>");
  }
  void Read(std::istream &is, bool binary) {
    ExpectToken(is, binary, "<AvgPhoneLogProbs>");
    int32 size;
    ReadBasicType(is, binary, &size);
    for (int i = 0; i < size; i++) {
      std::vector<int32> ctx;
      BaseFloat avglogprob;
      ReadIntegerVector(is, binary, &ctx);
      ReadBasicType(is, binary, &avglogprob);
      context_to_avglogprob_[ctx] = avglogprob;
    }
    ExpectToken(is, binary, "<LeftContext>");
    ReadBasicType(is, binary, &left_context_);
    ExpectToken(is, binary, "<RightContext>");
    ReadBasicType(is, binary, &right_context_);
    ExpectToken(is, binary, "</AvgPhoneLogProbs>");
  }
  unordered_map<std::vector<int32>, BaseFloat, VectorHasher<int32> >&
  GetPhoneToAvgLogprobMap() {
    return context_to_avglogprob_;
  }
  int32 LeftContext() { return left_context_; }
  int32 RightContext() { return right_context_; }

 private:
  int32 left_context_, right_context_;
  unordered_map<std::vector<int32>, BaseFloat, VectorHasher<int32> >
                                                         context_to_avglogprob_;
};

class NnetPhoneDurationScoreComputer {
 public:
  explicit NnetPhoneDurationScoreComputer(const NnetPhoneDurationModel &model):
      model_(model),
      compiler_(model.GetNnet()),
      feature_maker_(model.GetDurationModel()) {}
  explicit NnetPhoneDurationScoreComputer(
                                      const NnetPhoneDurationModel &model,
                                      const Vector<BaseFloat> &priors):
                                      //tmp// const AvgPhoneLogProbs &avg_logprobs):
      model_(model),
      compiler_(model.GetNnet()),
      feature_maker_(model.GetDurationModel()),
      priors_(priors) {}
      //tmp// avg_logprobs_(avg_logprobs) {}

  void ComputeOutputForExample(const NnetExample &eg,
                               Matrix<BaseFloat> *output);

  /// Computes the log prob for the middle phone (middle in the left/right
  /// context sense) in a phone-duration context (please refer to
  /// PhoneDurationFeatureMaker::MakeFeatureVector).
  BaseFloat GetLogProb(
                const std::vector<std::pair<int32, int32> > &phone_dur_context);

 private:
  const NnetPhoneDurationModel &model_;
  CachingOptimizingCompiler compiler_;
  PhoneDurationFeatureMaker feature_maker_;
  //tmp// AvgPhoneLogProbs avg_logprobs_;
  Vector<BaseFloat> priors_;
};

class PhoneDurationModelDeterministicFst
  : public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  // The first argument is needed only for using PhoneAndDurationToOlabel and
  // OlabelToPhoneAndDuration functions.
  // The thrid argument is non-cost only because it has a cache (this class does
  // not take ownership of the pointer).
  PhoneDurationModelDeterministicFst(int32 num_phones,
                                     const PhoneDurationModel &model,
                                     NnetPhoneDurationScoreComputer *scorer);

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
  int32 num_phones_;  // We need this only for decoding olabels into phone-id
                      // and duration
  NnetPhoneDurationScoreComputer &scorer_;

  /// Uses the score-computer object to compute the log prob for the middle
  /// phone in the input context. The input argument is a phone-duration
  /// context similar to the one in NnetPhoneDurationScoreComputer::GetLogProb
  /// but with the difference that the phone-IDs and duration values are
  /// encoded in a single integer (i.e. fst Label).
  BaseFloat GetLogProb(const std::vector<Label> &context) const;
};



/// This function uses a feature maker to convert a phone duration context (for
/// definition, please refer to PhoneDurationFeatureMaker::MakeFeatureVector)
/// into an Nnet3 example.
void MakeNnetExample(const NnetPhoneDurationModel &nnet_durmodel,
                 const PhoneDurationFeatureMaker &feat_maker,
                 const std::vector<std::pair<int32, int32> > &phone_dur_context,
                 int phone_index,
                 NnetExample *eg);

/// This function uses MakeNnetExample to convert a sequence of (phone,duration)
/// pairs (i.e. alignment) into a set of Nnet3 examples
void AlignmentToNnetExamples(const NnetPhoneDurationModel &nnet_durmodel,
                         const PhoneDurationFeatureMaker &feat_maker,
                         const std::vector<std::pair<int32, int32> > &alignment,
                         std::vector<NnetExample> *egs);


/// This function decodes an already encoded olabel to phone id and duration.
std::pair<int32, int32> OlabelToPhoneAndDuration(fst::StdArc::Label olabel,
                                                 int32 num_phones);
/// This function encodes two integers (phone-id and duration) to one integer
fst::StdArc::Label PhoneAndDurationToOlabel(int32 phone,
                                            int32 duration,
                                            int32 num_phones);
/// This function replaces the output labels of a compact lattice (which has
/// already been phone-aligned and has only 1 phone per arc) with the duration
/// and identity of the phones. This is done before composing with the on-demand
/// fst so that the fst can know the durations and phone-ids to compute the
/// log probabilities.
void DurationModelReplaceLabelsLattice(CompactLattice *in_clat,
                                       const TransitionModel &tmodel,
                                       int32 num_phones);
/// This function reverses the effects of the previous function and puts back
/// the original output labels.
void DurationModelReplaceLabelsBackLattice(CompactLattice *in_clat);

}  // namespace kaldi
#endif  // DURMOD_KALDI_DURMOD_H_

