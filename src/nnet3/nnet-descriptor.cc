// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#include <iterator>
#include <sstream>
#include "nnet3/nnet-descriptor.h"

namespace kaldi {
namespace nnet3 {

static void ExpectToken(const std::string &token,
                        const std::string &what_we_are_parsing,
                        const std::string **next_token) {
  if (**next_token != token)
    KALDI_ERR << "Expected '" << token << "' while parsing "
              << what_we_are_parsing << ", got "
              << **next_token;
  else
    **next_token++;
}

static int32 ReadIntegerToken(const std::string &what_we_are_parsing,
                              const std::string **next_token) {
  int32 ans;
  if (!ConvertStringToInteger(**next_token, &ans))
    KALDI_ERR << "Expected integer while parsing "
              << what_we_are_parsing << ", got "
              << **next_token;
  **next_token++;  
  return ans;
}

//static
ForwardingDescriptor* ForwardingDescriptor::Parse(
    const std::vector<std::string> &node_names,
    const std::string **next_token) {
  if (**next_token == "Offset") {
    (*next_token)++;
    ExpectToken("(", "OffsetForwardingDescriptor", next_token);
    ForwardingDescriptor *src = Parse(node_names, next_token);
    Index offset;
    offset.t = ReadIntegerToken("OffsetForwardingDescriptor", next_token);
    if (**next_token == ",") {
      (*next_token)++;      
      offset.t = ReadIntegerToken("OffsetForwardingDescriptor", next_token);
    }
    ExpectToken(")", "OffsetForwardingDescriptor", next_token);
    return new OffsetForwardingDescriptor(src, offset);
  } else if (**next_token == "Switch") {
    (*next_token)++;
    ExpectToken("(", "SwitchingForwardingDescriptor", next_token);
    std::vector<ForwardingDescriptor*> vec;
    while (true) {
      ForwardingDescriptor *src = Parse(node_names, next_token);
      vec.push_back(src);
      if (**next_token == ",") {
        (*next_token)++;
      } else {
        ExpectToken(")", "SwitchingForwardingDescriptor", next_token);
        return new SwitchingForwardingDescriptor(vec);
      }
    }
  } else if (**next_token == "Round") {
    (*next_token)++;
    ExpectToken("(", "RoundingForwardingDescriptor", next_token);
    ForwardingDescriptor *src = Parse(node_names, next_token);
    ExpectToken(",", "RoundingForwardingDescriptor", next_token);
    int32 t_modulus = ReadIntegerToken("RoundingForwardingDescriptor", next_token);
    if (t_modulus <= 0)
      KALDI_ERR << "Invalid modulus " << t_modulus << " in Round(..) expression";
    ExpectToken(")", "RoundingForwardingDescriptor", next_token);    
    return new RoundingForwardingDescriptor(src, t_modulus);
  } else if (**next_token == "ReplaceIndex") {
    (*next_token)++;
    ExpectToken("(", "ReplaceIndexForwardingDescriptor", next_token);
    ForwardingDescriptor *src = Parse(node_names, next_token);
    ExpectToken(",", "ReplaceIndexForwardingDescriptor", next_token);
    ReplaceIndexForwardingDescriptor::VariableName variable_name;
    if (**next_token == "t") {
      variable_name = ReplaceIndexForwardingDescriptor::kT;
    } else if (**next_token == "x") {
      variable_name = ReplaceIndexForwardingDescriptor::kX;
    } else {
      KALDI_ERR << "Parsing ReplaceIndexForwardingDescriptor, expected "
                << "'t' or 'x', got " << **next_token;
    }
    (*next_token)++;
    ExpectToken(",", "ReplaceIndexForwardingDescriptor", next_token);
    int32 value = ReadIntegerToken("ReplaceIndexForwardingDescriptor",
                                   next_token);
    ExpectToken(")", "ReplaceIndexForwardingDescriptor", next_token);
    return new ReplaceIndexForwardingDescriptor(src, variable_name, value);
  } else {
    // Note, node_names will have any node names that aren't allowed to appear
    // in Descriptors (e.g. output nodes) replace with something that can never
    // appear as a token, e.g. "**", so they will never match.
    int32 num_nodes = node_names.size();
    for (int32 i = 0; i < num_nodes; i++) {
      if (**next_token == node_names[i]) {
        (*next_token)++;
        return new SimpleForwardingDescriptor(i);
      }
    }
    KALDI_ERR << "Parsing Decriptor, expected a Descriptor but got "
              << **next_token;
    return NULL;  // suppress compiler warning.
  }
}
                        


void Descriptor::MapToInputs(
    const Index &index,
    std::vector<Cindex> *dependencies) const {
  dependencies->clear();
  std::vector<SumDescriptor*>::const_iterator sum_iter = parts_.begin(),
      sum_end = parts_.end();
  std::vector<Cindex> this_part;
  for (; sum_iter != sum_end; ++sum_iter) {
    const SumDescriptor &sum_descriptor = **sum_iter;
    sum_descriptor.MapToInputs(index, &this_part);
    dependencies->insert(dependencies->end(),
                         this_part.begin(),
                         this_part.end());
  }
}

// TODO
int32 SimpleForwardingDescriptor::Dim(const Nnet &nnet) const {
  return 0;
}

Cindex SimpleForwardingDescriptor::MapToInput(const Index &index) const {
  return Cindex(src_node_, index);
}

ForwardingDescriptor *SimpleForwardingDescriptor::Copy() const {
  return new SimpleForwardingDescriptor(src_node_);
}

void SimpleForwardingDescriptor::WriteConfig(
    std::ostream &os,
    const std::vector<std::string> &node_names) const {
  KALDI_ASSERT(static_cast<size_t>(src_node_) < node_names.size() &&
               IsValidName(node_names[src_node_]));
  os << node_names[src_node_];
}

// TODO
void OffsetForwardingDescriptor::ComputeDependencies(std::vector<int32> *node_indexes) const {
  return;
}

Cindex OffsetForwardingDescriptor::MapToInput(const Index &ind) const {
  Cindex answer = src_->MapToInput(ind);
  answer.second = answer.second + offset_;
  return answer;
}


ForwardingDescriptor *OffsetForwardingDescriptor::Copy() const {
  return new OffsetForwardingDescriptor(src_->Copy(), offset_);
}

void OffsetForwardingDescriptor::WriteConfig(
    std::ostream &os,
    const std::vector<std::string> &node_names) const {
  KALDI_ASSERT(offset_.n == 0);
  os << "Offset(";
  src_->WriteConfig(os, node_names);
  os << ", " << offset_.t;
  if (offset_.x != 0)
    os << ", " << offset_.x;
  os << ")";
}


// TODO
void SwitchingForwardingDescriptor::ComputeDependencies(
                                      std::vector<int32> *node_indexes) const {
  return;
}

Cindex SwitchingForwardingDescriptor::MapToInput(const Index &ind) const {
  KALDI_ASSERT(!src_.empty());
  int32 size = src_.size(), mod = ind.t % size;
  // next line gets "mathematical" modulus, not broken "C" modulus.
  if (mod < 0) mod += size;
  return src_[mod]->MapToInput(ind);
}


ForwardingDescriptor *SwitchingForwardingDescriptor::Copy() const {
  std::vector<ForwardingDescriptor*> src_copy(src_.size());
  for (size_t i = 0; i < src_.size(); i++)
    src_copy[i] = src_[i]->Copy();
  return new SwitchingForwardingDescriptor(src_copy);
}


void SwitchingForwardingDescriptor::WriteConfig(
    std::ostream &os,
    const std::vector<std::string> &node_names) const {
  KALDI_ASSERT(!src_.empty());
  os << "Switch(";
  for (size_t i = 0; i < src_.size(); i++) {
    src_[i]->WriteConfig(os, node_names);
    if (i + 1 < src_.size())
      os << ", ";
  }
  os << ")";
}

// TODO
void RoundingForwardingDescriptor::ComputeDependencies(
                                     std::vector<int32> *node_indexes) const {
  return;
}

Cindex RoundingForwardingDescriptor::MapToInput(const Index &ind) const {
  KALDI_ASSERT(t_modulus_ >= 1);
  Cindex ans = src_->MapToInput(ind);
  int32 mod = ans.second.t % t_modulus_;
  if (mod < 0)
    mod += t_modulus_;
  ans.second.t -= mod;
  return ans;
}

ForwardingDescriptor *RoundingForwardingDescriptor::Copy() const {
  return new RoundingForwardingDescriptor(src_->Copy(), t_modulus_);
}

void RoundingForwardingDescriptor::WriteConfig(
    std::ostream &os,
    const std::vector<std::string> &node_names) const {
  os << "Round(";
  src_->WriteConfig(os, node_names);
  os << ", " << t_modulus_ << ")";
}

// TODO
void ReplaceIndexForwardingDescriptor::ComputeDependencies(
                                         std::vector<int32> *node_indexes)
                                         const {
  return;
}

Cindex ReplaceIndexForwardingDescriptor::MapToInput(const Index &ind) const {
  Cindex ans = src_->MapToInput(ind);
  switch (variable_name_) {
    case kT: ans.second.t = value_; break;
    case kX: ans.second.x = value_; break;
    default:  // kN or any other value is not allowed (doesn't make sense
      // to change the minibatch index in this way).
      KALDI_ERR << "Invalid variable name";
  }    
  return ans;
}

ForwardingDescriptor *ReplaceIndexForwardingDescriptor::Copy() const {
  return new ReplaceIndexForwardingDescriptor(src_->Copy(),
                                              variable_name_, value_);
                                              
}

void ReplaceIndexForwardingDescriptor::WriteConfig(
    std::ostream &os,
    const std::vector<std::string> &node_names) const {
  os << "ReplaceIndex(";
  src_->WriteConfig(os, node_names);
  KALDI_ASSERT(variable_name_ == kT || variable_name_ == kX);
  os << (variable_name_ == kT  ? "t" : "x") << ", "
     << value_ << ")";
}

// TODO
bool SumDescriptor::IsComputable(const Index &ind,
                            const CindexSet &cindex_set,
                            std::vector<Cindex> *required_inputs) const {
  return true;
}

// TODO
void SumDescriptor::WriteConfig(std::ostream &os,
       const std::vector<std::string> &node_names) const {
  return;
}

SumDescriptor *UnarySumDescriptor::Copy() const {
  return new UnarySumDescriptor(src_->Copy(), required_);
}

//TODO
void UnarySumDescriptor::MapToInputs(const Index &ind,
                           std::vector<Cindex> *dependencies) const {
  return;
}

//TODO
bool UnarySumDescriptor::IsComputable(const Index &ind,
                            const CindexSet &cindex_set,
                            std::vector<Cindex> *required_inputs) const {
  return true;
}

void UnarySumDescriptor::WriteConfig(
    std::ostream &os,
    const std::vector<std::string> &node_names) const {
  if (!required_) os << "IfDefined(";
  src_->WriteConfig(os, node_names);
  if (!required_) os << ")";  
}

// TODO
int32 UnarySumDescriptor::Dim(const Nnet &nnet) const {
  return 0;
}

// TODO
void UnarySumDescriptor::ComputeDependencies(
                           std::vector<int32> *node_indexes) const {
  return;
}

// TODO
int32 UnarySumDescriptor::Modulus() const {
  return 0;
}


// TODO
void BinarySumDescriptor::MapToInputs(const Index &ind,
                           std::vector<Cindex> *dependencies) const {
  return;
}

// TODO
bool BinarySumDescriptor::IsComputable(const Index &ind,
                            const CindexSet &cindex_set,
                            std::vector<Cindex> *required_inputs) const {
  return true;
}

// TODO
int32 BinarySumDescriptor::Dim(const Nnet &nnet) const {
  return 0;
}

// TODO
void BinarySumDescriptor::ComputeDependencies(
                            std::vector<int32> *node_indexes) const {
  return;
}

// TODO
int32 BinarySumDescriptor::Modulus() const {
  return 0;
}

SumDescriptor *BinarySumDescriptor::Copy() const {
  return new BinarySumDescriptor(op_, src1_->Copy(), src2_->Copy());
}

void BinarySumDescriptor::WriteConfig(
    std::ostream &os,
    const std::vector<std::string> &node_names) const {
  KALDI_ASSERT(op_ == kSum || op_ == kFailover);
  if (op_ == kSum) os << "Sum(";
  if (op_ == kFailover) os << "Failover(";
  src1_->WriteConfig(os, node_names);
  os << ", ";
  src2_->WriteConfig(os, node_names);
  os << ")";
}


//static
SumDescriptor* SumDescriptor::Parse(
    const std::vector<std::string> &node_names,
    const std::string **next_token) {

  if (**next_token == "IfDefined") {
    ExpectToken("(", "SumDescriptor", next_token);
    ForwardingDescriptor *src = ForwardingDescriptor::Parse(node_names,
                                                            next_token);
    ExpectToken(")", "SumDescriptor", next_token);
    bool required = false;
    return new UnarySumDescriptor(src, required);
  } else if (**next_token == "Sum" || **next_token == "Failover") {
    BinarySumDescriptor::Operation op_type = (**next_token == "Sum" ?
                                              BinarySumDescriptor::kSum :
                                              BinarySumDescriptor::kFailover);
    (*next_token)++;
    ExpectToken("(", "SumDescriptor", next_token);
    SumDescriptor *src1 = Parse(node_names, next_token);
    ExpectToken(",", "SumDescriptor", next_token);
    SumDescriptor *src2 = Parse(node_names, next_token);
    ExpectToken(")", "SumDescriptor", next_token);
    return new BinarySumDescriptor(op_type, src1, src2);
  } else {
    ForwardingDescriptor *src = ForwardingDescriptor::Parse(node_names,
                                                            next_token);
    bool required = true;
    return new UnarySumDescriptor(src, required);
  }
}

void SimpleForwardingDescriptor::ComputeDependencies(
    std::vector<int32> *node_indexes) const {
  node_indexes->push_back(src_node_);  
}

int32 SwitchingForwardingDescriptor::Modulus() const {
  int32 ans = src_.size();;
  for (int32 i = 0; i < src_.size(); i++)
    ans = Lcm(ans, src_[i]->Modulus());
  return ans;
}

void Descriptor::WriteConfig(std::ostream &os,
                             const std::vector<std::string> &node_names) {
  KALDI_ASSERT(parts_.size() > 0);
  if (parts_.size() == 1)
    parts_[0]->WriteConfig(os, node_names);
  else {
    os << "Append(";
    for (size_t i = 0; i < parts_.size(); i++) {
      parts_[i]->WriteConfig(os, node_names);
      if (i + 1 < parts_.size())
        os << ", ";
    }
    os << ")";
  }
}
void Descriptor::Destroy() {
  for (size_t i = 0; i < parts_.size(); i++)
    delete parts_[i];
  parts_.clear();
}

bool Descriptor::Parse(const std::vector<std::string> &node_names,
                       const std::string **next_token) {
  Destroy();
  try {
    if (**next_token == "Append") {
      (*next_token)++;
      ExpectToken("(", "Descriptor", next_token);
      while (1) {
        SumDescriptor *ptr = SumDescriptor::Parse(node_names,
                                                  next_token);
        parts_.push_back(ptr);
        if (**next_token == ",") {
          (*next_token)++;
          continue;
        } else {
          ExpectToken(")", "Descriptor", next_token);
          ExpectToken("end of input", "Descriptor", next_token);
          return true;
        }
      }
    } else {
      SumDescriptor *ptr = SumDescriptor::Parse(node_names, next_token);
      parts_.push_back(ptr);
      ExpectToken("end of input", "Descriptor", next_token);
      return true;
    }
  } catch (...) {
    return false;
  }
}


Descriptor& Descriptor::operator=(const Descriptor &other) {
  Destroy();
  for (size_t i = 0; i < other.parts_.size(); i++)
    parts_.push_back(other.parts_[i]->Copy());
  return *this;
}

int32 Descriptor::Modulus() const {
  int32 ans = 1;
  for (size_t i = 0; i < parts_.size(); i++)
    ans = Lcm(ans, parts_[i]->Modulus());
  return ans;  
}


} // namespace nnet3
} // namespace kaldi
