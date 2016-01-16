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

void PhoneDurationModelOptions::Register(OptionsItf *opts) {
  opts->Register("left-context", &left_context,
                 "Number of left context frames");
  opts->Register("right-context", &right_context,
                 "Number of right context frames");
  opts->Register("max-duration", &max_duration,
                 "Max phone duration in frames. Durations longer than this will"
                 "be mapped to this value.");
}

PhoneDurationFeatureMaker::PhoneDurationFeatureMaker(
                                              const PhoneDurationModel &model) {
  InitFeatureMaker(model);
}

BaseFloat PhoneDurationFeatureMaker::NormalizeDuration(
                                               int32 duration_in_frames) const {
  BaseFloat normalized_duration =
                          2.0 / (1.0 + Exp(-0.01f * duration_in_frames)) - 1;
  // normalized_duration = sqrt(duration_in_frames / max_duration_);
  // the above commented normalization resulted in much faster convergence.
  // TODO(hhadian): should be tested in practice
  return normalized_duration;
}

void PhoneDurationFeatureMaker::MakeFeatureVector(
                    const std::vector<std::pair<int32, int32> > &phone_context,
                    int phone_index,
                    SparseVector<BaseFloat> *feat) const {
  std::vector<std::pair<MatrixIndexT, BaseFloat> > feat_elements;
  int feat_idx = 0;  // current index in the feat vector
  for (int i = (phone_index - left_context_);
           i <= (phone_index + right_context_); i++) {
    int phone_id;
    int phone_duration;
    std::vector<int32> binary_feats;
    if (i < 0 || i >= phone_context.size()) {  // not available in the context:
                                               // set all binary feat elements
                                               // and duration to zero
      phone_id = 0;
      phone_duration = 0;
    } else {
      int32 phone = phone_context[i].first;
      phone_duration = phone_context[i].second;
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
  max_duration_ = model.max_duration_;
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
  int max_phone_idx = 0;  // TODO(hhadian): in the end, check if all phones have
                          // been categorized.
  for (int i = 0; i < model.roots_.size(); i++) {
    for (int j = 0; j < model.roots_[i].size(); j++) {
      int phone = model.roots_[i][j];
      KALDI_ASSERT(phone_id_[phone] == 0);  // only one root for each phone
      phone_id_[phone] = i + 1;  // phone identities start from 1
      if (max_phone_idx < phone)
        max_phone_idx = phone;
      if (binary_feats_.count(phone) <= 0) {
        KALDI_WARN << "Phone " << phone << " does not have any "
                   << "acoustic question associated.";
      }
    }
  }
}

void PhoneDurationModel::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<PhoneDurationModel>");
  ExpectToken(is, binary, "<LeftContext>");
  ReadBasicType(is, binary, &left_context_);
  ExpectToken(is, binary, "<RightContext>");
  ReadBasicType(is, binary, &right_context_);
  ExpectToken(is, binary, "<MaxDuration>");
  ReadBasicType(is, binary, &max_duration_);
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
  WriteToken(os, binary, "<MaxDuration>");
  WriteBasicType(os, binary, max_duration_);
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

void MakeNnetExample(const PhoneDurationFeatureMaker &feat_maker,
                     const std::vector<std::pair<int32, int32> > &phone_context,
                     int phone_index,
                     NnetExample *eg) {
  SparseVector<BaseFloat> feat;
  feat_maker.MakeFeatureVector(phone_context, phone_index, &feat);
  int32 phone_duration = phone_context[phone_index].second;
  SparseMatrix<BaseFloat> feat_mat(1, feat.Dim());
  feat_mat.SetRow(0, feat);
  int32 output_dim = feat_maker.OutputDim();
  Posterior output_elements(1);

  // The nodes at the output of the network are indexed from
  // 0 to (max_duration_ - 1). Index 0 is for duration = 1.
  int32 duration_id = (phone_duration > output_dim) ?
                                                       (output_dim - 1):
                                                       (phone_duration - 1);
  output_elements[0].push_back(std::make_pair(duration_id, 1.0));
  SparseMatrix<BaseFloat> output_mat(output_dim, output_elements);

  NnetIo input, output;
  input.name = "input";
  input.features = feat_mat;
  input.indexes.resize(1);
  output.name = "output";
  output.features = output_mat;
  output.indexes.resize(1);
  eg->io.push_back(input);
  eg->io.push_back(output);
}
void AlignmentToNnetExamples(const PhoneDurationFeatureMaker &feat_maker,
                         const std::vector<std::pair<int32, int32> > &alignment,
                         std::vector<NnetExample> *egs) {
  for (int i = 0; i < alignment.size(); i++) {
    NnetExample eg;
    MakeNnetExample(feat_maker, alignment, i, &eg);
    egs->push_back(eg);
  }
}


void NnetPhoneDurationModel::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<NnetPhoneDurationModel>");
  dur_model_.Read(is, binary);
  nnet_.Read(is, binary);
  ExpectToken(is, binary, "</NnetPhoneDurationModel>");
}

void NnetPhoneDurationModel::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<NnetPhoneDurationModel>");
  dur_model_.Write(os, binary);
  nnet_.Write(os, binary);
  WriteToken(os, binary, "</NnetPhoneDurationModel>");
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
  computer.Forward();
  const CuMatrixBase<BaseFloat> &nnet_output = computer.GetOutput("output");
  output->Resize(nnet_output.NumRows(), nnet_output.NumCols());
  nnet_output.CopyToMat(output);
}

BaseFloat NnetPhoneDurationScoreComputer::GetLogProb(
                   const std::vector<std::pair<int32, int32> > &phone_context) {
  KALDI_ASSERT(phone_context.size() == model_.FullContextSize());
  NnetExample eg;
  MakeNnetExample(feature_maker_, phone_context, model_.LeftContext(), &eg);
  Matrix<BaseFloat> output;
  ComputeOutputForExample(eg, &output);
  int32 phone_duration = phone_context[model_.LeftContext()].second;
  int32 duration_id = (phone_duration > model_.MaxDuration()) ?
                                                (model_.MaxDuration() - 1):
                                                (phone_duration - 1);

  int32 actual_duration_id = phone_duration - 1;  // if we had no max duration
  BaseFloat logprob = output(0, duration_id);
  // make sure the distribution over all durations (1 to inf) sums to 1
  if (actual_duration_id >= duration_id)
    logprob *= pow(0.5f, actual_duration_id - duration_id + 1);
  return logprob;
}

PhoneDurationModelDeterministicFst::PhoneDurationModelDeterministicFst(
                                        const PhoneDurationModel &model,
                                        NnetPhoneDurationScoreComputer *scorer):
                            context_size_(model.FullContextSize()),
                            right_context_(model.RightContext()),
                            max_duration_(model.MaxDuration()),
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
  if (ilabel == 0) {  // if this is an epsilon arc, move on
    oarc->nextstate = s;
    oarc->weight = Weight::One();
    return true;
  }

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

BaseFloat PhoneDurationModelDeterministicFst
           ::GetLogProb(const std::vector<Label> &context) const {
  KALDI_ASSERT(context.size() == context_size_);
  std::vector<std::pair<int32, int32> > phone_context(context_size_);

  // convert the encoded "phone+duration" to phones and durations pairs
  for (int i = 0; i < context.size(); i++)
    phone_context[i] = OlabelToPhoneAndDuration(context[i], max_duration_);

  BaseFloat logprob = scorer_.GetLogProb(phone_context);
  return logprob;
}

std::pair<int32, int32> OlabelToPhoneAndDuration(fst::StdArc::Label olabel,
                                                 int32 max_duration) {
  return std::make_pair(olabel / (max_duration + 1),
                        olabel % (max_duration + 1));
}

fst::StdArc::Label PhoneAndDurationToOlabel(int32 phone,
                                            int32 duration,
                                            int32 max_duration) {
  return (phone * (max_duration + 1)) + duration;
}

void DurationModelReplaceLabelsLattice(CompactLattice *clat,
                                       const TransitionModel &tmodel,
                                       int32 max_duration) {
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

      // take care of max-duration
      if (duration > max_duration)
        duration = max_duration;

      // encode phone+duration to 1 integer as olabel
      CompactLatticeArc arc2(arc);
      arc2.olabel = PhoneAndDurationToOlabel(phone_label,
                                             duration, max_duration);
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
