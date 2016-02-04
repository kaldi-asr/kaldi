// nnet3/nnet-descriptor.cc

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
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-computation-graph.h"

namespace kaldi {
namespace nnet3 {

static std::string ParsingContext(const std::string *token_ptr) {
  if (*token_ptr == "end of input")
    return "";
  std::string next_few_tokens = ", next part of line is: ";
  // in the next line, *token_ptr should never equal "" but it's to mitigate the
  // effect of bugs where we read past the end of the array.
  while (*token_ptr != "end of input" && *token_ptr != "" &&
         next_few_tokens.size() < 40) {
    next_few_tokens = (next_few_tokens + " ") + *token_ptr;
    token_ptr++;
  }
  if (*token_ptr != "end of input")
    next_few_tokens = next_few_tokens + " ...";
  return next_few_tokens;
}

static void ExpectToken(const std::string &token,
                        const std::string &what_we_are_parsing,
                        const std::string **next_token) {
  if (**next_token != token)
    KALDI_ERR << "Expected '" << token << "' while parsing "
              << what_we_are_parsing << ", got "
              << **next_token << ParsingContext(*next_token);
  else  // advance token pointer.
    (*next_token)++;
}

static int32 ReadIntegerToken(const std::string &what_we_are_parsing,
                              const std::string **next_token) {
  int32 ans;
  if (!ConvertStringToInteger(**next_token, &ans))
    KALDI_ERR << "Expected integer while parsing "
              << what_we_are_parsing << ", got '"
              << **next_token << "'" << ParsingContext(*next_token);
  (*next_token)++;  // advance token pointer.
  return ans;
}


void Descriptor::GetDependencies(
    const Index &index,
    std::vector<Cindex> *dependencies) const {
  dependencies->clear();
  std::vector<SumDescriptor*>::const_iterator sum_iter = parts_.begin(),
      sum_end = parts_.end();
  std::vector<Cindex> this_part;
  for (; sum_iter != sum_end; ++sum_iter)
    (*sum_iter)->GetDependencies(index, dependencies);
}

int32 SimpleForwardingDescriptor::Dim(const Nnet &nnet) const {
  return nnet.GetNode(src_node_).Dim(nnet);
}

Cindex SimpleForwardingDescriptor::MapToInput(const Index &index) const {
  return Cindex(src_node_, index);
}

ForwardingDescriptor *SimpleForwardingDescriptor::Copy() const {
  return new SimpleForwardingDescriptor(src_node_);
}

void SimpleForwardingDescriptor::GetNodeDependencies(
    std::vector<int32> *node_indexes) const {
  node_indexes->push_back(src_node_);
}

void SimpleForwardingDescriptor::WriteConfig(
    std::ostream &os,
    const std::vector<std::string> &node_names) const {
  KALDI_ASSERT(static_cast<size_t>(src_node_) < node_names.size() &&
               IsValidName(node_names[src_node_]));
  os << node_names[src_node_];
}

void OffsetForwardingDescriptor::GetNodeDependencies(
    std::vector<int32> *node_indexes) const {
  src_->GetNodeDependencies(node_indexes);
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


void SwitchingForwardingDescriptor::GetNodeDependencies(
    std::vector<int32> *node_indexes) const {
  for (size_t i = 0; i < src_.size(); i++)
    src_[i]->GetNodeDependencies(node_indexes);
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

void RoundingForwardingDescriptor::GetNodeDependencies(
    std::vector<int32> *node_indexes) const {
  src_->GetNodeDependencies(node_indexes);
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

void ReplaceIndexForwardingDescriptor::GetNodeDependencies(
    std::vector<int32> *node_indexes) const {
  src_->GetNodeDependencies(node_indexes);
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
  os << ", " << (variable_name_ == kT  ? "t" : "x") << ", "
     << value_ << ")";
}

SumDescriptor *OptionalSumDescriptor::Copy() const {
  return new OptionalSumDescriptor(src_->Copy());
}

void OptionalSumDescriptor::GetDependencies(
    const Index &ind,
    std::vector<Cindex> *dependencies) const {
  src_->GetDependencies(ind, dependencies);
}

void OptionalSumDescriptor::WriteConfig(
    std::ostream &os,
    const std::vector<std::string> &node_names) const {
  os << "IfDefined(";
  src_->WriteConfig(os, node_names);
  os << ")";
}

int32 OptionalSumDescriptor::Dim(const Nnet &nnet) const {
  return src_->Dim(nnet);
}

void OptionalSumDescriptor::GetNodeDependencies(
    std::vector<int32> *node_indexes) const {
  src_->GetNodeDependencies(node_indexes);
}

SumDescriptor *SimpleSumDescriptor::Copy() const {
  return new SimpleSumDescriptor(src_->Copy());
}

void SimpleSumDescriptor::GetDependencies(const Index &ind,
                                          std::vector<Cindex> *dependencies) const {
  dependencies->push_back(src_->MapToInput(ind));
}

bool SimpleSumDescriptor::IsComputable(
    const Index &ind,
    const CindexSet &cindex_set,
    std::vector<Cindex> *used_inputs) const {
  Cindex c = src_->MapToInput(ind);
  bool src_present  = cindex_set(c);
  if (src_present && used_inputs != NULL)
    used_inputs->push_back(c);
  return src_present;
}

void SimpleSumDescriptor::WriteConfig(
    std::ostream &os,
    const std::vector<std::string> &node_names) const {
  src_->WriteConfig(os, node_names);
}

int32 SimpleSumDescriptor::Dim(const Nnet &nnet) const {
  return src_->Dim(nnet);
}

void SimpleSumDescriptor::GetNodeDependencies(
    std::vector<int32> *node_indexes) const {
  src_->GetNodeDependencies(node_indexes);
}

void BinarySumDescriptor::GetDependencies(
    const Index &ind, std::vector<Cindex> *dependencies) const {
  src1_->GetDependencies(ind, dependencies);
  src2_->GetDependencies(ind, dependencies);
}

bool BinarySumDescriptor::IsComputable(
    const Index &ind,
    const CindexSet &cindex_set,
    std::vector<Cindex> *used_inputs) const {
  KALDI_PARANOID_ASSERT(op_ == kSum || op_ == kFailover);
  std::vector<Cindex> src1_inputs, src2_inputs;
  bool r = (used_inputs != NULL);
  bool src1_computable = src1_->IsComputable(ind, cindex_set,
                                             r ? &src1_inputs: NULL),
      src2_computable = src2_->IsComputable(ind, cindex_set,
                                            r ? &src2_inputs : NULL);
  if (op_ == kSum) {
    if (src1_computable && src2_computable) {
      if (r) {
        used_inputs->insert(used_inputs->end(),
                            src1_inputs.begin(), src1_inputs.end());
        used_inputs->insert(used_inputs->end(),
                            src2_inputs.begin(), src2_inputs.end());
      }
      return true;
    } else {
      return false;
    }
  } else {
    KALDI_ASSERT(op_ == kFailover);
    if (src1_computable) {
      if (r)
        used_inputs->insert(used_inputs->end(),
                            src1_inputs.begin(), src1_inputs.end());
      return true;
    } else if (src2_computable) {
      if (r)
        used_inputs->insert(used_inputs->end(),
                            src2_inputs.begin(), src2_inputs.end());
      return true;
    } else {
      return false;
    }
  }
}

int32 BinarySumDescriptor::Dim(const Nnet &nnet) const {
  int32 dim1 = src1_->Dim(nnet), dim2 = src2_->Dim(nnet);
  if (dim1 != dim2)
    KALDI_ERR << "Neural net contains " << (op_ == kSum ? "Sum" : "Failover")
              << " expression with inconsistent dimension: " << dim1
              << " vs. " << dim2;
  return dim1;
}

void BinarySumDescriptor::GetNodeDependencies(
    std::vector<int32> *node_indexes) const {
  src1_->GetNodeDependencies(node_indexes);
  src2_->GetNodeDependencies(node_indexes);
}

int32 BinarySumDescriptor::Modulus() const {
  return Lcm(src1_->Modulus(), src2_->Modulus());
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

int32 SwitchingForwardingDescriptor::Modulus() const {
  int32 ans = src_.size();;
  for (int32 i = 0; i < src_.size(); i++)
    ans = Lcm(ans, src_[i]->Modulus());
  return ans;
}

bool Descriptor::Parse(const std::vector<std::string> &node_names,
                       const std::string **next_token) {
  GeneralDescriptor *gen_desc;
  try {
    gen_desc = GeneralDescriptor::Parse(node_names,
                                        next_token);
  } catch (...) {
    return false;
  }
  if (**next_token != "end of input")
    KALDI_ERR << "Parsing Descriptor, expected end of input but got "
              << "'" <<  *next_token << "'";
  Descriptor *desc = gen_desc->ConvertToDescriptor();
  *this = *desc;
  delete desc;
  delete gen_desc;
  return true;
}

void Descriptor::WriteConfig(std::ostream &os,
                             const std::vector<std::string> &node_names) const {
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

int32 Descriptor::Dim(const Nnet &nnet) const {
  int32 num_parts = parts_.size();
  int32 dim = 0;
  for (int32 part = 0; part < num_parts; part++)
    dim += parts_[part]->Dim(nnet);
  KALDI_ASSERT(dim > 0);
  return dim;
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


bool Descriptor::IsComputable(const Index &ind,
                              const CindexSet &cindex_set,
                              std::vector<Cindex> *input_terms) const {
  if (input_terms)
    input_terms->clear();
  for (size_t i = 0; i < parts_.size(); i++) {
    // if any of the parts is not computable, the whole is not computable.
    if (!parts_[i]->IsComputable(ind, cindex_set, input_terms)) {
      if (input_terms)
        input_terms->clear();
      return false;
    }
  }
  return true;
}

const SumDescriptor& Descriptor::Part(int32 n) const {
  KALDI_ASSERT(static_cast<size_t>(n) < parts_.size());
  return *(parts_[n]);
}

void Descriptor::GetNodeDependencies(std::vector<int32> *node_indexes) const {
  node_indexes->clear();
  for (size_t i = 0; i < parts_.size(); i++)
    parts_[i]->GetNodeDependencies(node_indexes);
}


// static
GeneralDescriptor* GeneralDescriptor::Parse(
    const std::vector<std::string> &node_names,
    const std::string **next_token) {

  DescriptorType t;
  if (**next_token == "Append") {
    t = kAppend;
  } else if (**next_token == "Sum") {
    t = kSum;
  } else if (**next_token == "Failover") {
    t = kFailover;
  } else if (**next_token == "IfDefined") {
    t = kIfDefined;
  } else if (**next_token == "Offset") {
    t = kOffset;
  } else if (**next_token == "Switch") {
    t = kSwitch;
  } else if (**next_token == "Round") {
    t = kRound;
  } else if (**next_token == "ReplaceIndex") {
    t = kReplaceIndex;
  } else {
    // what we read wasn't a reserved name like Offset, etc.
    // We expect a node name in that case.
    for (size_t i = 0; i < node_names.size(); i++) {
      if (**next_token == node_names[i]) {
        GeneralDescriptor *ans = new GeneralDescriptor(kNodeName, i);
        (*next_token)++;
        return ans;
      }
    }
    KALDI_ERR << "Expected a Descriptor, got instead "
              << **next_token;
    t = kNodeName;  // suppress compiler warning.
  }
  (*next_token)++;
  ExpectToken("(", "Descriptor", next_token);
  GeneralDescriptor *ans = new GeneralDescriptor(t);
  switch (t) {
    case kAppend: case kSum: case kSwitch:
      ans->ParseAppendOrSumOrSwitch(node_names, next_token); break;
    case kFailover: ans->ParseFailover(node_names, next_token); break;
    case kIfDefined: ans->ParseIfDefined(node_names, next_token); break;
    case kOffset: ans->ParseOffset(node_names, next_token); break;
    case kRound: ans->ParseRound(node_names, next_token); break;
    case kReplaceIndex: ans->ParseReplaceIndex(node_names, next_token); break;
    default:
      KALDI_ERR << "Code error";
  }
  return ans;
}

void GeneralDescriptor::ParseAppendOrSumOrSwitch(
    const std::vector<std::string> &node_names,
    const std::string **next_token) {
  descriptors_.push_back(Parse(node_names, next_token));
  while (true) {
    if (**next_token == ")") {
      (*next_token)++;
      return;
    } else if (**next_token == ",") {
      (*next_token)++;
      descriptors_.push_back(Parse(node_names, next_token));
    } else {
      KALDI_ERR << "Expected ',' or ')', got "
                << **next_token;
    }
  }
}

void GeneralDescriptor::ParseIfDefined(
    const std::vector<std::string> &node_names,
    const std::string **next_token) {
  descriptors_.push_back(Parse(node_names, next_token));
  ExpectToken(")", "IfDefined", next_token);
}

void GeneralDescriptor::ParseFailover(
    const std::vector<std::string> &node_names,
    const std::string **next_token) {
  descriptors_.push_back(Parse(node_names, next_token));
  ExpectToken(",", "Failover", next_token);
  descriptors_.push_back(Parse(node_names, next_token));
  ExpectToken(")", "Failover", next_token);
}

void GeneralDescriptor::ParseOffset(
    const std::vector<std::string> &node_names,
    const std::string **next_token) {
  descriptors_.push_back(Parse(node_names, next_token));
  ExpectToken(",", "Offset", next_token);
  value1_ = ReadIntegerToken("Offset", next_token);
  if (**next_token == ",") {
    (*next_token)++;
    value2_ = ReadIntegerToken("Offset", next_token);
  } else {
    value2_ = 0;
  }
  ExpectToken(")", "Offset", next_token);
}


void GeneralDescriptor::ParseRound(
    const std::vector<std::string> &node_names,
    const std::string **next_token) {
  descriptors_.push_back(Parse(node_names, next_token));
  ExpectToken(",", "Round", next_token);
  value1_ = ReadIntegerToken("Round", next_token);
  ExpectToken(")", "Round", next_token);
}


void GeneralDescriptor::ParseReplaceIndex(
    const std::vector<std::string> &node_names,
    const std::string **next_token) {
  descriptors_.push_back(Parse(node_names, next_token));
  ExpectToken(",", "ReplaceIndex", next_token);
  if (**next_token == "t") {
    value1_ = int32(ReplaceIndexForwardingDescriptor::kT);
    (*next_token)++;
  } else if (**next_token == "x") {
    value1_ = int32(ReplaceIndexForwardingDescriptor::kX);
    (*next_token)++;
  } else {
    KALDI_ERR << "Expected 't' or 'x', got " << **next_token;
  }
  ExpectToken(",", "ReplaceIndex", next_token);
  value2_ = ReadIntegerToken("Replace", next_token);
  ExpectToken(")", "ReplaceIndex", next_token);
}


int32 GeneralDescriptor::NumAppendTerms() const {
  int32 ans = 0;
  switch (descriptor_type_) {
    case kNodeName: ans = 1; break;
    case kAppend: {
      for (size_t i = 0; i < descriptors_.size(); i++)
        ans += descriptors_[i]->NumAppendTerms();
      break;
    }
    default:
      KALDI_ASSERT(descriptors_.size() > 0);
      ans = descriptors_[0]->NumAppendTerms();
      for (size_t i = 1; i < descriptors_.size(); i++)
        KALDI_ASSERT(descriptors_[i]->NumAppendTerms() == ans);
  }
  return ans;
}

GeneralDescriptor* GeneralDescriptor::GetAppendTerm(int32 term) const {
  switch (descriptor_type_) {
    case kNodeName:
      KALDI_ASSERT(term == 0);
      return new GeneralDescriptor(kNodeName, value1_);
    case kAppend: {
      int32 cur_term = term;
      for (size_t i = 0; i < descriptors_.size(); i++) {
        int32 this_num_terms = descriptors_[i]->NumAppendTerms();
        if (cur_term < this_num_terms)
          return descriptors_[i]->GetAppendTerm(cur_term);
        else
          cur_term -= this_num_terms;
      }
      KALDI_ERR << "Code error, getting append term.";
      return NULL; // avoid compiler warning
    }
    default: {
      GeneralDescriptor *ans = new GeneralDescriptor(descriptor_type_,
                                                     value1_, value2_);
      ans->descriptors_.resize(descriptors_.size());
      for (size_t i = 0; i < descriptors_.size(); i++)
        ans->descriptors_[i] = descriptors_[i]->GetAppendTerm(term);
      return ans;
    }
  }
}


// this is only called at the top level.
GeneralDescriptor* GeneralDescriptor::NormalizeAppend() const {
  int32 num_terms = NumAppendTerms();
  KALDI_ASSERT(num_terms > 0);
  if (num_terms == 1) {
    return GetAppendTerm(0);
  } else {
    GeneralDescriptor *ans = new GeneralDescriptor(kAppend);
    ans->descriptors_.resize(num_terms);
    for (size_t i = 0; i < num_terms; i++) {
      ans->descriptors_[i] = GetAppendTerm(i);
    }
    return ans;
  }
}


//static
bool GeneralDescriptor::Normalize(GeneralDescriptor *parent) {
  bool changed = false;
  switch (parent->descriptor_type_) {
    case kOffset: {  // this block combines Offset(Offset(x, ..), ..).
      KALDI_ASSERT(parent->descriptors_.size() == 1);
      GeneralDescriptor *child = parent->descriptors_[0];
      if (child->descriptor_type_ == kOffset) {
        KALDI_ASSERT(child->descriptors_.size() == 1);
        GeneralDescriptor *grandchild = child->descriptors_[0];
        parent->value1_ += child->value1_;
        parent->value2_ += child->value2_;
        child->descriptors_.clear();  // avoid delete in destructor.
        delete child;
        parent->descriptors_[0] = grandchild;
        changed = true;
      } else if (parent->value1_ == 0 && parent->value2_ == 0) {
        // remove redundant Offset expression like Offset(x, 0).
        parent->descriptors_.swap(child->descriptors_);
        parent->descriptor_type_ = child->descriptor_type_;
        parent->value1_ = child->value1_;
        parent->value2_ = child->value2_;
        child->descriptors_.clear();  // avoid delete in destructor.
        delete child;
        changed = true;
        break;  // break from the switch ('parent' is no longer of type
        // kOffset)', so we don't want to carry through.
      }
    }
      // ... and continue through to the next case statement.
    case kSwitch: case kRound: case kReplaceIndex: { // ..and kOffset:
      KALDI_ASSERT(parent->descriptors_.size() >= 1);
      GeneralDescriptor *child = parent->descriptors_[0];
      KALDI_ASSERT(child->descriptor_type_ != kAppend);  // would be code error
      // (already did
      // NormalizeAppend()).
      if (child->descriptor_type_ == kSum ||
          child->descriptor_type_ == kFailover ||
          child->descriptor_type_ == kIfDefined) {
        if (parent->descriptors_.size() > 1) {
          KALDI_ASSERT(parent->descriptor_type_ == kSwitch);
          KALDI_ERR << "Sum(), Failover() or IfDefined() expression inside Switch(), "
                    << "we can't currently normalize this.";
        }
        // this is a forbidden case of a sum descriptor inside a forwarding
        // descriptor.  we need to rearrange.  E.g. Offset(Sum(x, y), 1) becomes
        // Sum(Offset(x, 1), Offset(y, 1)).
        for (size_t i = 0; i < child->descriptors_.size(); i++) {
          GeneralDescriptor *grandchild = child->descriptors_[i];
          GeneralDescriptor *modified_grandchild =
              new GeneralDescriptor(parent->descriptor_type_,
                                    parent->value1_,
                                    parent->value2_);
          // modified_grandchild takes ownership of grandchild.
          modified_grandchild->descriptors_.push_back(grandchild);
          child->descriptors_[i] = modified_grandchild;
        }
        // copy all members from child to parent.
        parent->descriptor_type_ = child->descriptor_type_;
        parent->value1_ = child->value1_;
        parent->value2_ = child->value2_;
        parent->descriptors_.swap(child->descriptors_);
        child->descriptors_.clear();  // avoid delete in destructor of 'child'
        delete child;
        changed = true;
      }
      break;
    }
    case kSum: {
      KALDI_ASSERT(!parent->descriptors_.empty());
      if (parent->descriptors_.size() == 1) {
        // convert Sum(x) to just x.
        GeneralDescriptor *child = parent->descriptors_[0];
        parent->descriptor_type_ = child->descriptor_type_;
        parent->descriptors_.swap(child->descriptors_);
        parent->value1_ = child->value1_;
        parent->value2_ = child->value2_;
        child->descriptors_.clear();  // avoid delete in destructor.
        delete child;
        changed = true;
      } else if (parent->descriptors_.size() > 2) {
        // convert Sum(a, b, c, ...) to Sum(a, Sum(b, c, ...)).
        GeneralDescriptor *new_child = new GeneralDescriptor(kSum);
        // assign b, c, .. to the descriptors of new_child.
        new_child->descriptors_.insert(new_child->descriptors_.begin(),
                                       parent->descriptors_.begin() + 1,
                                       parent->descriptors_.end());
        parent->descriptors_.erase(parent->descriptors_.begin() + 1,
                                   parent->descriptors_.end());
        parent->descriptors_.push_back(new_child);
        changed = true;
      }
      break;
    }
    default: { } // empty statement.
  }
  // ... and recurse.
  for (size_t i = 0; i < parent->descriptors_.size(); i++)
    changed = changed || Normalize(parent->descriptors_[i]);
  return changed;
}

GeneralDescriptor* GeneralDescriptor::GetNormalizedDescriptor() const {
  GeneralDescriptor *ans = NormalizeAppend();
  while (Normalize(ans));  // keep normalizing as long as it changes.
  return ans;
}

void GeneralDescriptor::Print(const std::vector<std::string> &node_names,
                              std::ostream &os) {
  switch (descriptor_type_) {
    // first handle all the expressions of the form "Operator(<desc1>, ... <descN>)".
    case kAppend: os << "Append("; break;
    case kSum: os << "Sum("; break;
    case kFailover: os << "Failover("; break;
    case kIfDefined: os << "IfDefined("; break;
    case kSwitch: os << "Switch("; break;
      // now handle the exceptions.
    case kOffset: case kRound: {
      os << "Offset(";
      KALDI_ASSERT(descriptors_.size() == 1);
      descriptors_[0]->Print(node_names, os);
      os << ", " << value1_;
      if (descriptor_type_ == kOffset && value2_ != 0) os << ", " << value2_;
      os << ")";
      return;
    }
    case kReplaceIndex: {
      os << "ReplaceIndex(";
      KALDI_ASSERT(descriptors_.size() == 1);
      descriptors_[0]->Print(node_names, os);
      KALDI_ASSERT(value1_ == int32(ReplaceIndexForwardingDescriptor::kT) ||
                   value1_ == int32(ReplaceIndexForwardingDescriptor::kX));
      if (value1_ == int32(ReplaceIndexForwardingDescriptor::kT)) {
        os << ", t, ";
      } else {
        os << ", x, ";
      }
      os << value2_ << ")";
      return;
    }
    case kNodeName: {
      KALDI_ASSERT(static_cast<size_t>(value1_) < node_names.size());
      os << node_names[value1_];
      return;
    }
  }
  for (size_t i = 0; i < descriptors_.size(); i++) {
    if (i > 0) os << ", ";
    descriptors_[i]->Print(node_names, os);
  }
  os << ")";
}


Descriptor* GeneralDescriptor::ConvertToDescriptor() {
  GeneralDescriptor *normalized = GetNormalizedDescriptor();
  std::vector<SumDescriptor*> sum_descriptors;
  if (normalized->descriptor_type_ == kAppend) {
    for (size_t i = 0; i < normalized->descriptors_.size(); i++)
      sum_descriptors.push_back(
          normalized->descriptors_[i]->ConvertToSumDescriptor());
  } else {
    sum_descriptors.push_back(normalized->ConvertToSumDescriptor());
  }
  Descriptor *ans = new Descriptor(sum_descriptors);
  delete normalized;
  return ans;
}

SumDescriptor *GeneralDescriptor::ConvertToSumDescriptor() const {
  KALDI_ASSERT(descriptor_type_ != kAppend &&
               "Badly normalized descriptor");
  switch (descriptor_type_) {
    case kAppend:
      KALDI_ERR << "Badly normalized descriptor";
    case kSum: case kFailover: {
      KALDI_ASSERT(descriptors_.size() == 2 && "Bad descriptor");
      return new BinarySumDescriptor(
          descriptor_type_ == kSum ?
          BinarySumDescriptor::kSum : BinarySumDescriptor::kFailover,
          descriptors_[0]->ConvertToSumDescriptor(),
          descriptors_[1]->ConvertToSumDescriptor());
    }
    case kIfDefined: {
      KALDI_ASSERT(descriptors_.size() == 1 && "Bad descriptor");
      return new OptionalSumDescriptor(
          descriptors_[0]->ConvertToSumDescriptor());
    }
    default: {
      return new SimpleSumDescriptor(this->ConvertToForwardingDescriptor());
    }
  }
}


ForwardingDescriptor *GeneralDescriptor::ConvertToForwardingDescriptor() const {
  switch (this->descriptor_type_) {
    case kNodeName: return new SimpleForwardingDescriptor(value1_);
    case kOffset: {
      KALDI_ASSERT(descriptors_.size() == 1 && "bad descriptor");
      return new OffsetForwardingDescriptor(
          descriptors_[0]->ConvertToForwardingDescriptor(),
          Index(0, value1_, value2_));
    }
    case kSwitch: {
      std::vector<ForwardingDescriptor*> descriptors;
      for (size_t i = 0; i < descriptors_.size(); i++)
        descriptors.push_back(descriptors_[i]->ConvertToForwardingDescriptor());
      return new SwitchingForwardingDescriptor(descriptors);
    }
    case kRound: {
      KALDI_ASSERT(descriptors_.size() == 1 && "bad descriptor");
      return new RoundingForwardingDescriptor(
          descriptors_[0]->ConvertToForwardingDescriptor(),
          value1_);
    }
    case kReplaceIndex: {
      KALDI_ASSERT(descriptors_.size() == 1 && "bad descriptor");
      KALDI_ASSERT(value1_ == int32(ReplaceIndexForwardingDescriptor::kT) ||
                   value1_ == int32(ReplaceIndexForwardingDescriptor::kX));
      return new ReplaceIndexForwardingDescriptor(
          descriptors_[0]->ConvertToForwardingDescriptor(),
          value1_ == int32(ReplaceIndexForwardingDescriptor::kT) ?
          ReplaceIndexForwardingDescriptor::kT :
          ReplaceIndexForwardingDescriptor::kX,
          value2_);
    }
    default:
      KALDI_ERR << "Invalid descriptor type (failure in normalization?)";
      return NULL;
  }
}


} // namespace nnet3
} // namespace kaldi
