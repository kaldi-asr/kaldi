// segmenter/information-bottleneck-clusterable.cc

// Copyright 2017   Vimal Manohar (Johns Hopkins University)

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

#include "segmenter/information-bottleneck-clusterable.h"

namespace kaldi {

void InformationBottleneckClusterable::AddStats(
    int32 id, BaseFloat count,
    const VectorBase<BaseFloat> &relevance_dist) {
  std::map<int32, BaseFloat>::iterator it = counts_.find(id);
  KALDI_ASSERT(it == counts_.end() || it->first != id);
  counts_.insert(it, std::make_pair(id, count));

  double sum = relevance_dist.Sum();
  KALDI_ASSERT (sum != 0.0);

  p_yp_c_.Scale(total_count_);
  p_yp_c_.AddVec(count / sum, relevance_dist);
  total_count_ += count;
  p_yp_c_.Scale(1.0 / total_count_);
}

BaseFloat InformationBottleneckClusterable::Objf(
    BaseFloat relevance_factor, BaseFloat input_factor) const {
  double relevance_entropy = 0.0, count = 0.0;
  for (int32 i = 0; i < p_yp_c_.Dim(); i++) {
    if (p_yp_c_(i) > 1e-20) {
      relevance_entropy -= p_yp_c_(i) * Log(p_yp_c_(i));
      count += p_yp_c_(i);
    }
  }
  relevance_entropy = total_count_ * (relevance_entropy / count - Log(count));

  double input_entropy = total_count_ * Log(total_count_);
  for (std::map<int32, BaseFloat>::const_iterator it = counts_.begin();
       it != counts_.end(); ++it) {
    input_entropy -= it->second * Log(it->second);
  }

  BaseFloat objf = -relevance_factor * relevance_entropy 
    + input_factor * input_entropy;
  return objf;
}

void InformationBottleneckClusterable::Add(const Clusterable &other_in) {
  KALDI_ASSERT(other_in.Type() == "information-bottleneck");
  const InformationBottleneckClusterable *other = 
    static_cast<const InformationBottleneckClusterable*> (&other_in);

  for (std::map<int32, BaseFloat>::const_iterator it = other->counts_.begin();
       it != other->counts_.end(); ++it) {
    std::map<int32, BaseFloat>::iterator hint_it = counts_.lower_bound(
        it->first);
    KALDI_ASSERT (hint_it == counts_.end() || hint_it->first != it->first);
    counts_.insert(hint_it, *it);
  }

  p_yp_c_.Scale(total_count_);
  p_yp_c_.AddVec(other->total_count_, other->p_yp_c_);
  total_count_ += other->total_count_;
  p_yp_c_.Scale(1.0 / total_count_);
}

void InformationBottleneckClusterable::Sub(const Clusterable &other_in) {
  KALDI_ASSERT(other_in.Type() == "information-bottleneck");
  const InformationBottleneckClusterable *other = 
    static_cast<const InformationBottleneckClusterable*> (&other_in);

  for (std::map<int32, BaseFloat>::const_iterator it = other->counts_.begin();
       it != other->counts_.end(); ++it) {
    std::map<int32, BaseFloat>::iterator hint_it = counts_.lower_bound(
        it->first);
    KALDI_ASSERT (hint_it->first == it->first);
    counts_.erase(hint_it);
  }

  p_yp_c_.Scale(total_count_);
  p_yp_c_.AddVec(-other->total_count_, other->p_yp_c_);
  total_count_ -= other->total_count_;
  p_yp_c_.Scale(1.0 / total_count_);
}

Clusterable* InformationBottleneckClusterable::Copy() const {
  InformationBottleneckClusterable *ans = 
    new InformationBottleneckClusterable(RelevanceDim());
  ans->Add(*this);
  return ans;
}

void InformationBottleneckClusterable::Scale(BaseFloat f) {
  KALDI_ASSERT(f >= 0.0);
  for (std::map<int32, BaseFloat>::iterator it = counts_.begin();
       it != counts_.end(); ++it) {
    it->second *= f;
  }
  total_count_ *= f;
}

void InformationBottleneckClusterable::Write(
    std::ostream &os, bool binary) const {
  WriteToken(os, binary, "IBCL");   // magic string.
  WriteBasicType(os, binary, counts_.size());
  BaseFloat total_count = 0.0;
  for (std::map<int32, BaseFloat>::const_iterator it = counts_.begin();
       it != counts_.end(); ++it) {
    WriteBasicType(os, binary, it->first);
    WriteBasicType(os, binary, it->second);
    total_count += it->second;
  }
  KALDI_ASSERT(ApproxEqual(total_count_, total_count));
  WriteToken(os, binary, "<RelevanceDistribution>");
  p_yp_c_.Write(os, binary);
}
  
Clusterable* InformationBottleneckClusterable::ReadNew(
    std::istream &is, bool binary) const {
  InformationBottleneckClusterable *ibc = 
    new InformationBottleneckClusterable();
  ibc->Read(is, binary);
  return ibc;
}

void InformationBottleneckClusterable::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "IBCL");  // magic string.
  int32 size;
  ReadBasicType(is, binary, &size);

  for (int32 i = 0; i < 2 * size; i++) {
    int32 id; 
    BaseFloat count;
    ReadBasicType(is, binary, &id);
    ReadBasicType(is, binary, &count);
    std::pair<std::map<int32, BaseFloat>::iterator, bool> ret;
    ret = counts_.insert(std::make_pair(id, count));
    if (!ret.second) {
      KALDI_ERR << "Duplicate element " << id << " when reading counts";
    }
    total_count_ += count;
  }
  
  ExpectToken(is, binary, "<RelevanceDistribution>");
  p_yp_c_.Read(is, binary);
}
 
BaseFloat InformationBottleneckClusterable::ObjfPlus(
    const Clusterable &other, BaseFloat relevance_factor, 
    BaseFloat input_factor) const {
  InformationBottleneckClusterable *copy = static_cast<InformationBottleneckClusterable*>(Copy());
  copy->Add(other);
  BaseFloat ans = copy->Objf(relevance_factor, input_factor);
  delete copy;
  return ans;
}

BaseFloat InformationBottleneckClusterable::ObjfMinus(
    const Clusterable &other, BaseFloat relevance_factor, 
    BaseFloat input_factor) const {
  InformationBottleneckClusterable *copy = static_cast<InformationBottleneckClusterable*>(Copy());
  copy->Add(other);
  BaseFloat ans = copy->Objf(relevance_factor, input_factor);
  delete copy;
  return ans;
}

BaseFloat InformationBottleneckClusterable::Distance(
    const Clusterable &other_in, BaseFloat relevance_factor,
    BaseFloat input_factor) const {
  KALDI_ASSERT(other_in.Type() == "information-bottleneck");
  const InformationBottleneckClusterable *other = 
    static_cast<const InformationBottleneckClusterable*> (&other_in);

  BaseFloat normalizer = this->Normalizer() + other->Normalizer();
  BaseFloat pi_i = this->Normalizer() / normalizer;
  BaseFloat pi_j = other->Normalizer() / normalizer;

  // Compute the distribution q_Y(y) = p(y|{c_i} + {c_j})
  Vector<BaseFloat> relevance_dist(this->RelevanceDim());
  relevance_dist.AddVec(pi_i, this->RelevanceDist());
  relevance_dist.AddVec(pi_j, other->RelevanceDist());

  BaseFloat relevance_divergence
    = pi_i * KLDivergence(this->RelevanceDist(), relevance_dist)
    + pi_j * KLDivergence(other->RelevanceDist(), relevance_dist);

  BaseFloat input_divergence 
    = Log(normalizer) - pi_i * Log(this->Normalizer()) 
    - pi_j * Log(other->Normalizer());

  KALDI_ASSERT(relevance_divergence > -1e-4);
  KALDI_ASSERT(input_divergence > -1e-4);
  return (normalizer * (relevance_factor * relevance_divergence 
                        - input_factor * input_divergence));
}

BaseFloat KLDivergence(const VectorBase<BaseFloat> &p1, 
                       const VectorBase<BaseFloat> &p2) {
  KALDI_ASSERT(p1.Dim() == p2.Dim());
  
  double ans = 0.0, sum = 0.0;
  for (int32 i = 0; i < p1.Dim(); i++) {
    if (p1(i) > 1e-20) {
      ans += p1(i) * Log(p1(i) / p2(i));
      sum += p1(i);
    }
  }
  return ans / sum - Log(sum);
}

}  // end namespace kaldi
