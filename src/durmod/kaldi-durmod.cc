// durmod/kaldi-durmod.cc

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
#include <algorithm>
#include "durmod/kaldi-durmod.h"

namespace kaldi {

void PhoneDurationModel::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<PhoneDurationModel>");
  ExpectToken(is, binary, "<LeftContext>");
  ReadBasicType(is, binary, &left_context_);
  ExpectToken(is, binary, "<RightContext>");
  ReadBasicType(is, binary, &right_context_);
  ExpectToken(is, binary, "<Roots>");
  int32 size;
  ReadBasicType(is, binary, &size);
  roots_.resize(size);
  for (int i = 0; i < roots_.size(); i++)
    ReadIntegerVector(is, binary, &(roots_[i]));
  ExpectToken(is, binary, "</Roots>");
  ExpectToken(is, binary, "<Questions>");
  ReadBasicType(is, binary, &size);
  questions_.resize(size);
  for (int i = 0; i < questions_.size(); i++)
    ReadIntegerVector(is, binary, &(questions_[i]));
  ExpectToken(is, binary, "</Questions>");
  ExpectToken(is, binary, "</PhoneDurationModel>");
}

void PhoneDurationModel::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PhoneDurationModel>");
  WriteToken(os, binary, "<LeftContext>");
  WriteBasicType(os, binary, left_context_);
  WriteToken(os, binary, "<RightContext>");
  WriteBasicType(os, binary, right_context_);
  WriteToken(os, binary, "<Roots>");
  WriteBasicType(os, binary, static_cast<int32>(roots_.size()));
  for (int i = 0; i < roots_.size(); i++)
    WriteIntegerVector(os, binary, roots_[i]);
  WriteToken(os, binary, "</Roots>");
  WriteToken(os, binary, "<Questions>");
  WriteBasicType(os, binary, static_cast<int32>(questions_.size()));
  for (int i = 0; i < questions_.size(); i++)
    WriteIntegerVector(os, binary, questions_[i]);
  WriteToken(os, binary, "</Questions>");
  WriteToken(os, binary, "</PhoneDurationModel>");
}

std::string PhoneDurationModel::Info() const {
  std::ostringstream os;
  os << "left-context: " << LeftContext()
     << std::endl
     << "right-context: " << RightContext()
     << std::endl
     << "full-context-size: " << FullContextSize()
     << std::endl;
  return os.str();
}

std::string PhoneDurationFeatureMaker::Info() const {
  std::ostringstream os;
  os << "feature-dim: " << FeatureDim()
     << std::endl
     << "num-binary-features: " << NumBinaryFeatures()
     << std::endl
     << "num-phone-identities: " << NumPhoneIdentities()
     << std::endl;
  return os.str();
}

PhoneDurationFeatureMaker::PhoneDurationFeatureMaker(
                                              const PhoneDurationModel &model) {
  InitFeatureMaker(model);
}

BaseFloat PhoneDurationFeatureMaker::NormalizeDuration(
                                               int32 duration_in_frames) const {
  BaseFloat normalized_duration =
                          2.0 / (1.0 + Exp(-0.01f * duration_in_frames)) - 1;
  return normalized_duration;
}

void PhoneDurationFeatureMaker::MakeFeatureVector(
                 const std::vector<std::pair<int32, int32> > &phone_dur_context,
                 int phone_index,
                 SparseVector<BaseFloat> *feat) const {
  std::vector<std::pair<MatrixIndexT, BaseFloat> > feat_elements;
  int feat_idx = 0;  // current index in the feat vector
  for (int i = (phone_index - left_context_);
           i <= (phone_index + right_context_); i++) {
    int phone_id;
    int phone_duration;
    std::vector<int32> binary_feats;
    if (i < 0 || i >= phone_dur_context.size()) {  // not available in the
                                                   // context: set all binary
                                                   // feat elements and duration
                                                   // to zero
      phone_id = 0;
      phone_duration = 0;
    } else {
      int32 phone = phone_dur_context[i].first;
      phone_duration = phone_dur_context[i].second;
      if (phone == 0) {  // a null phone (for edges)
        phone_id = 0;
      } else if (phone_id_.find(phone) != phone_id_.end()) {
        phone_id = phone_id_.find(phone)->second;
      } else {
        KALDI_ERR << "No phone identity found for phone "
                  << phone
                  << ". Check your roots.int";
      }
      if (binary_feats_.find(phone) != binary_feats_.end())
        binary_feats = binary_feats_.find(phone)->second;
    }

    // phone-id feature
    feat_elements.push_back(std::make_pair(feat_idx + phone_id, 1.0));
    feat_idx += num_phone_identities_;

    // binary features (from extra questions for eg.)
    for (int j = 0; j < binary_feats.size(); j++) {
      feat_elements.push_back(std::make_pair(feat_idx + binary_feats[j], 1.0));
    }
    feat_idx += num_binary_features_;

    // duration features
    if (i < phone_index) {
      feat_elements.push_back(
                             std::make_pair(feat_idx++,
                                            NormalizeDuration(phone_duration)));
    }
  }
  SparseVector<BaseFloat> tmp(feature_dim_, feat_elements);
  feat->CopyFromSvec(tmp);
}

void PhoneDurationFeatureMaker::InitFeatureMaker(
                                              const PhoneDurationModel &model) {
  num_binary_features_ = model.questions_.size();
  num_phone_identities_ = model.roots_.size() + 1;  // id=0 is for not-available
                                                    // phones (i.e. null phones
                                                    // which occur at edges).
  left_context_ = model.left_context_;
  right_context_ = model.right_context_;
  int input_dim_phones = num_phone_identities_ *
                                           (left_context_ + right_context_ + 1);
  int input_dim_durations = left_context_;
  int input_dim_binary = num_binary_features_ *
                                           (left_context_ + right_context_ + 1);
  feature_dim_ = input_dim_phones + input_dim_binary + input_dim_durations;

  // create the reverse map for questions membership
  for (int i = 0; i < model.questions_.size(); i++) {
    for (int j = 0; j < model.questions_[i].size(); j++) {
      int phone = model.questions_[i][j];
      binary_feats_[phone].push_back(i);
    }
  }

  // and the reverse map for phoneme tree roots
  for (int i = 0; i < model.roots_.size(); i++) {
    for (int j = 0; j < model.roots_[i].size(); j++) {
      int phone = model.roots_[i][j];
      KALDI_ASSERT(phone_id_[phone] == 0);  // only one root for each phone
      phone_id_[phone] = i + 1;  // phone identities start from 1
      if (binary_feats_.count(phone) <= 0) {
        KALDI_WARN << "Phone " << phone << " does not have any "
                   << "acoustic question associated.";
      }
    }
  }
}

void NnetPhoneDurationModel::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<NnetPhoneDurationModel>");
  dur_model_.Read(is, binary);
  nnet_.Read(is, binary);
  ExpectToken(is, binary, "<MaxDuration>");
  ReadBasicType(is, binary, &opts_.max_duration);
  ExpectToken(is, binary, "<NumComponents>");
  ReadBasicType(is, binary, &opts_.num_mixture_components);
  ExpectToken(is, binary, "</NnetPhoneDurationModel>");
}

void NnetPhoneDurationModel::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<NnetPhoneDurationModel>");
  dur_model_.Write(os, binary);
  nnet_.Write(os, binary);
  WriteToken(os, binary, "<MaxDuration>");
  WriteBasicType(os, binary, opts_.max_duration);
  WriteToken(os, binary, "<NumComponents>");
  WriteBasicType(os, binary, opts_.num_mixture_components);
  WriteToken(os, binary, "</NnetPhoneDurationModel>");
}

std::string NnetPhoneDurationModel::Info() const {
  std::ostringstream os;
  os << GetDurationModel().Info()
     << "lognormal-objective: "
     << (IsNnetObjectiveLogNormal() ? "true" : "false")
     << std::endl
     << "max-duration: " << MaxDuration()
     << std::endl
     << "num-mixture-components: " << NumMixtureComponents()
     << std::endl
     << "# Nnet3 info follows." << std::endl
     << GetNnet().Info();
  return os.str();
}

void NnetPhoneDurationScoreComputer::ComputeOutputForExample(
                                                    const NnetExample &eg,
                                                    Matrix<BaseFloat> *output) {
  ComputationRequest request;
  bool need_backprop = false, store_stats = false;
  GetComputationRequest(model_.GetNnet(), eg, need_backprop,
                        store_stats, &request);
  const NnetComputation &computation = *(compiler_.Compile(request));
  NnetComputeOptions options;
  //  if (GetVerboseLevel() >= 3)
  //    options.debug = true;
  NnetComputer computer(options, computation, model_.GetNnet(), NULL);
  computer.AcceptInputs(model_.GetNnet(), eg.io);
  computer.Run();
  const CuMatrixBase<BaseFloat> &nnet_output = computer.GetOutput("output");
  output->Resize(nnet_output.NumRows(), nnet_output.NumCols());
  nnet_output.CopyToMat(output);
}

BaseFloat NnetPhoneDurationScoreComputer::GetLogProb(
               const std::vector<std::pair<int32, int32> > &phone_dur_context) {
  KALDI_ASSERT(phone_dur_context.size() == model_.FullContextSize());
  NnetExample eg;
  MakeNnetExample(model_, feature_maker_,
                  phone_dur_context, model_.LeftContext(), &eg);
  Matrix<BaseFloat> output;
  ComputeOutputForExample(eg, &output);
  int32 phone_duration = phone_dur_context[model_.LeftContext()].second;
  // Please refer to MakeNnetExample() to see what the output nodes of the
  // network show.
  BaseFloat logprob;
  if (model_.IsNnetObjectiveLogNormal()) {
    BaseFloat mu = output(0, 0);
    BaseFloat sigma = Exp(output(0, 1));
    
    BaseFloat zeromean_logduration = Log(
                                   static_cast<BaseFloat>(phone_duration)) - mu;
    logprob = -Log(phone_duration * sigma * sqrtf(2.0 * M_PI)) -
               KALDI_SQR(zeromean_logduration) /
               (2 * KALDI_SQR(sigma));
  } else {
    int32 duration_id = (phone_duration > model_.MaxDuration()) ?
                                                  (model_.MaxDuration() - 1):
                                                  (phone_duration - 1);

    // Now we estimate the probabilities for the durations longer than
    // max_duration. To do this we distribute the probability mass at the last
    // node (i.e. the probability for duration==max_duration) to all the
    // durations equal to and longer than max_duration. We assume a geometric
    // form for this distribution and scale the probabilities such that the
    // whole distribution sums to 1. So assuming the probability mass at
    // duration==max_duration is P, then the probabilities
    // for some duration >= max_duration will be 
    // P/norm_sum * alpha^(duration-max_duration+1) where norm_sum is
    // equal to alpha/(1-alpha). We set alpha to exp(-1.0/max_duration) so
    // that the probabilities do not decline too rapidly.

    int32 actual_duration_id = phone_duration - 1;  // in case max duration was
                                                    // infinity (i.e. we
                                                    // had infinitely many nodes
                                                    // at the output of the
                                                    // network)
    logprob = output(0, duration_id);

    // make sure the distribution over all durations (1 to inf) sums to 1
    BaseFloat alpha = Exp(-1.0f / model_.MaxDuration());
    BaseFloat probability_normalization_sum = alpha / (1 - alpha);
    if (phone_duration >= model_.MaxDuration())
      logprob += (actual_duration_id - duration_id + 1) * Log(alpha) -
                                             Log(probability_normalization_sum);
  }
  if (phone_duration - 1 < priors_.Dim()) {
    //KALDI_LOG << "Using priors... priors.Dim() is " << priors_.Dim();
    logprob -= Log(priors_(phone_duration - 1));
  }
/** //tmp//
  unordered_map<std::vector<int32>,
                      BaseFloat,
                      VectorHasher<int32> >& phonecontext_to_avglogprob_map =
                                        avg_logprobs_.GetPhoneToAvgLogprobMap();
  if (phonecontext_to_avglogprob_map.size() != 0) {
    // logprob -= phone_avg_training_logprobs(phone);
    std::vector<int32> phone_context(avg_logprobs_.LeftContext() +
                                     avg_logprobs_.RightContext() + 1);
    for (int j = model_.LeftContext() - avg_logprobs_.LeftContext();
         j <= model_.LeftContext() + avg_logprobs_.RightContext(); j++) {
      phone_context[j - (model_.LeftContext() - avg_logprobs_.LeftContext())] =
                                                     phone_dur_context[j].first;
    }
    int32 phone = phone_dur_context[model_.LeftContext()].first;
    std::vector<int32> single_phone_context(1, phone);  // for back-off
    if (phonecontext_to_avglogprob_map[phone_context] != 0.0) {
      logprob -= phonecontext_to_avglogprob_map[phone_context];
    } else {  // back off
      KALDI_LOG << "WARNING: no logprob for phone context "
                << AvgPhoneLogProbs::PhoneContext2Str(phone_context)
                << ". Backing off to "
                << phonecontext_to_avglogprob_map[single_phone_context] + 2;
      logprob -= phonecontext_to_avglogprob_map[single_phone_context] + 2;
    }
  } */
  
  return logprob;
}

PhoneDurationModelDeterministicFst::PhoneDurationModelDeterministicFst(
                                      int32 num_phones,
                                      const PhoneDurationModel &model,
                                      NnetPhoneDurationScoreComputer *scorer):
                            context_size_(model.FullContextSize()),
                            right_context_(model.RightContext()),
                            num_phones_(num_phones),
                            scorer_(*scorer) {
  // In the beginning we have a Null left context.
  // Null means that the phone-identity and duration are both zero --> olabel=0
  std::vector<int32> start_context(model.LeftContext(), 0);
  state_to_context_.push_back(start_context);
  context_to_state_[start_context] = 0;
  start_state_ = 0;
}

fst::StdArc::Weight PhoneDurationModelDeterministicFst::Final(StateId state) {
  KALDI_ASSERT(state < static_cast<StateId>(state_to_context_.size()));
  std::vector<Label> seq = state_to_context_[state];
  BaseFloat logprob = 0;
  // handle the phones that have not been considered yet:
  for (int i = 0; i < right_context_; i++) {
    seq.push_back(0);  // add Null right context
    if (seq.size() >= context_size_) {
      if (seq.size() > context_size_)
        seq.erase(seq.begin());
      logprob += GetLogProb(seq);
    }
  }
  return Weight(-logprob);
}

bool PhoneDurationModelDeterministicFst::GetArc(StateId s,
                                                Label ilabel,
                                                fst::StdArc *oarc) {
  // The state ids increment with each state we encounter.
  // if the assert fails, then we are trying to access
  // unseen states that are not immediately traversable.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_context_.size());
  std::vector<Label> seq = state_to_context_[s];

  oarc->ilabel = ilabel;
  oarc->olabel = ilabel;

  // Update state info if the label is not epsilon
  seq.push_back(ilabel);
  if (seq.size() > context_size_) {
    // Remove oldest phone in the history.
    seq.erase(seq.begin());
  }
  std::pair<const std::vector<Label>, StateId> new_state(
    seq,
    static_cast<Label>(state_to_context_.size()));
  // Now get state id for destination state.
  typedef typename MapType::iterator IterType;
  std::pair<IterType, bool> result = context_to_state_.insert(new_state);
  if (result.second == true) {
    state_to_context_.push_back(seq);
  }
  if (seq.size() < context_size_) {  // in the beginning of the lattice
    oarc->weight = Weight::One();
  } else {
    BaseFloat logprob = GetLogProb(seq);
    oarc->weight = Weight(-logprob);
  }
  oarc->nextstate = result.first->second;  // The next state id.

  return true;
}

BaseFloat PhoneDurationModelDeterministicFst::GetLogProb(
                                      const std::vector<Label> &context) const {
  KALDI_ASSERT(context.size() == context_size_);
  std::vector<std::pair<int32, int32> > phone_dur_context(context_size_);

  // convert the encoded "phone+duration" to phones and durations pairs
  for (int i = 0; i < context.size(); i++)
    phone_dur_context[i] = OlabelToPhoneAndDuration(context[i],  num_phones_);

  BaseFloat logprob = scorer_.GetLogProb(phone_dur_context);
  return logprob;
}

void MakeNnetExample(
                 const NnetPhoneDurationModel &nnet_durmodel,
                 const PhoneDurationFeatureMaker &feat_maker,
                 const std::vector<std::pair<int32, int32> > &phone_dur_context,
                 int phone_index,
                 NnetExample *eg) {
  SparseVector<BaseFloat> feat;
  feat_maker.MakeFeatureVector(phone_dur_context, phone_index, &feat);
  int32 phone_duration = phone_dur_context[phone_index].second;
  SparseMatrix<BaseFloat> feat_mat(1, feat.Dim());
  feat_mat.SetRow(0, feat);

  NnetIo input, output;
  input.name = "input";
  input.features = feat_mat;
  input.indexes.resize(1);
  output.name = "output";
  output.indexes.resize(1);
  if (nnet_durmodel.IsNnetObjectiveLogNormal()) {
    Matrix<BaseFloat> output_mat(1, 1);
    output_mat(0, 0) = phone_duration;
    output.features = output_mat;
  } else {
    int32 output_dim = nnet_durmodel.MaxDuration();
    Posterior output_elements(1);
    // The nodes at the output of the network are indexed from
    // 0 to (max_duration_ - 1). Index 0 is for duration = 1.
    int32 duration_id = (phone_duration > output_dim) ?
                                                       (output_dim - 1):
                                                       (phone_duration - 1);
    output_elements[0].push_back(std::make_pair(duration_id, 1.0));
    SparseMatrix<BaseFloat> output_mat(output_dim, output_elements);
    output.features = output_mat;
  }
  eg->io.push_back(input);
  eg->io.push_back(output);
}

void AlignmentToNnetExamples(const NnetPhoneDurationModel &nnet_durmodel,
                         const PhoneDurationFeatureMaker &feat_maker,
                         const std::vector<std::pair<int32, int32> > &alignment,
                         std::vector<NnetExample> *egs) {
  for (int i = 0; i < alignment.size(); i++) {
    NnetExample eg;
    MakeNnetExample(nnet_durmodel, feat_maker, alignment, i, &eg);
    egs->push_back(eg);
  }
}

std::pair<int32, int32> OlabelToPhoneAndDuration(fst::StdArc::Label olabel,
                                                 int32 num_phones) {
  return std::make_pair(olabel % (num_phones + 1),
                        olabel / (num_phones + 1));
}

fst::StdArc::Label PhoneAndDurationToOlabel(int32 phone,
                                            int32 duration,
                                            int32 num_phones) {
  return phone + duration * (num_phones + 1);
}

void DurationModelReplaceLabelsLattice(CompactLattice *clat,
                                       const TransitionModel &tmodel,
                                       int32 num_phones) {
  // iterate over all arcs
  for (int s = 0; s < clat->NumStates(); s++) {
    for (fst::MutableArcIterator<CompactLattice> aiter(clat, s);
         !aiter.Done(); aiter.Next()) {
      const CompactLatticeArc& arc = aiter.Value();

      // find the duration and phone-id of the phone on the arc
      const std::vector<int32> &tid_seq = arc.weight.String();
      int32 duration = tid_seq.size();
      int32 phone_label = 0;
      if (duration != 0)  // if not an epsilon arc
        phone_label = tmodel.TransitionIdToPhone(tid_seq[0]);

      // encode phone+duration to 1 integer as olabel
      CompactLatticeArc arc2(arc);
      arc2.olabel = PhoneAndDurationToOlabel(phone_label,
                                             duration, num_phones);
      aiter.SetValue(arc2);
     }
  }
}

void DurationModelReplaceLabelsBackLattice(CompactLattice *clat) {
  // iterate over all arcs
  for (int s = 0; s < clat->NumStates(); s++) {
    for (fst::MutableArcIterator<CompactLattice> aiter(clat, s);
         !aiter.Done(); aiter.Next()) {
      const CompactLatticeArc& arc = aiter.Value();
      CompactLatticeArc arc2(arc);
      // set the olabel to be the same as ilabel
      arc2.olabel = arc2.ilabel;
      aiter.SetValue(arc2);
     }
  }
}


}  // namespace kaldi
