// tree/context-dep.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include "tree/context-dep.h"
#include "base/kaldi-math.h"
#include "tree/build-tree.h"

namespace kaldi {

bool ContextDependency::Compute(const std::vector<int32> &phoneseq,
                                 int32 pdf_class,
                                 int32 *pdf_id) const {
  KALDI_ASSERT(static_cast<int32>(phoneseq.size()) == N_);
  EventType  event_vec;
  event_vec.reserve(N_+1);
  event_vec.push_back(std::make_pair
                      (static_cast<EventKeyType>(kPdfClass),  // -1
                       static_cast<EventValueType>(pdf_class)));
  KALDI_COMPILE_TIME_ASSERT(kPdfClass < 0);  // or it would not be sorted.
  for (int32 i = 0;i < N_;i++) {
    event_vec.push_back(std::make_pair
                        (static_cast<EventKeyType>(i), static_cast<EventValueType>(phoneseq[i])));
    KALDI_ASSERT(static_cast<EventAnswerType>(phoneseq[i]) != -1);  // >=0 ?
  }
  KALDI_ASSERT(pdf_id != NULL);
  return to_pdf_->Map(event_vec, pdf_id);
}

ContextDependency *GenRandContextDependency(const std::vector<int32> &phone_ids,
                                            bool ensure_all_covered,
                                            std::vector<int32> *hmm_lengths) {
  KALDI_ASSERT(IsSortedAndUniq(phone_ids));
  int32 num_phones = phone_ids.size();
  int32 num_stats = 1 + (Rand() % 15) * (Rand() % 15);  // up to 14^2 + 1 separate stats.
  int32 N = 2 + Rand() % 3;  // 2, 3 or 4.
  int32 P = Rand() % N;
  float ctx_dep_prob = 0.7 + 0.3*RandUniform();
  int32 max_phone = *std::max_element(phone_ids.begin(), phone_ids.end());
  hmm_lengths->clear();
  hmm_lengths->resize(max_phone+1, -1);
  std::vector<bool> is_ctx_dep(max_phone+1);

  for (int32 i = 0; i <= max_phone; i++) {
    (*hmm_lengths)[i] = 1 + Rand() % 3;
    is_ctx_dep[i] = (RandUniform() < ctx_dep_prob);  // true w.p. ctx_dep_prob.
  }
  for (size_t i = 0;i < (size_t) num_phones;i++) {
    KALDI_VLOG(2) <<  "For idx = "<< i << ", (phone_id, hmm_length, is_ctx_dep) == " << (phone_ids[i]) << " " << ((*hmm_lengths)[phone_ids[i]]) << " " << (is_ctx_dep[phone_ids[i]]);
  }
  // Generate rand stats.
  BuildTreeStatsType stats;
  size_t dim = 3 + Rand() % 20;
  GenRandStats(dim, num_stats, N, P, phone_ids, *hmm_lengths, is_ctx_dep, ensure_all_covered, &stats);

  // Now build the tree.

  Questions qopts;
  int32 num_quest = Rand() % 10, num_iters = rand () % 5;
  qopts.InitRand(stats, num_quest, num_iters, kAllKeysUnion);  // This was tested in build-tree-utils-test.cc

  float thresh = 100.0 * RandUniform();

  EventMap *tree = NULL;
  std::vector<std::vector<int32> > phone_sets(phone_ids.size());
  for (size_t i = 0; i < phone_ids.size(); i++)
    phone_sets[i].push_back(phone_ids[i]);
  std::vector<bool> share_roots(phone_sets.size(), true),
      do_split(phone_sets.size(), true);

  tree = BuildTree(qopts, phone_sets, *hmm_lengths, share_roots,
                   do_split, stats, thresh, 1000, 0.0, P);
  DeleteBuildTreeStats(&stats);
  return new ContextDependency(N, P, tree);
}


ContextDependency *GenRandContextDependencyLarge(const std::vector<int32> &phone_ids,
                                                 int N, int P,
                                                 bool ensure_all_covered,
                                                 std::vector<int32> *hmm_lengths) {
  KALDI_ASSERT(IsSortedAndUniq(phone_ids));
  int32 num_phones = phone_ids.size();
  int32 num_stats = 3000;  // each is a separate context.
  float ctx_dep_prob = 0.9;
  KALDI_ASSERT(num_phones > 0);
  hmm_lengths->clear();
  int32 max_phone = *std::max_element(phone_ids.begin(), phone_ids.end());
  hmm_lengths->resize(max_phone+1, -1);
  std::vector<bool> is_ctx_dep(max_phone+1);

  for (int32 i = 0; i <= max_phone; i++) {
    (*hmm_lengths)[i] = 1 + Rand() % 3;
    is_ctx_dep[i] = (RandUniform() < ctx_dep_prob);  // true w.p. ctx_dep_prob.
  }
  for (size_t i = 0;i < (size_t) num_phones;i++) {
    KALDI_VLOG(2) <<  "For idx = "<< i << ", (phone_id, hmm_length, is_ctx_dep) == " << (phone_ids[i]) << " " << ((*hmm_lengths)[phone_ids[i]]) << " " << (is_ctx_dep[phone_ids[i]]);
  }
  // Generate rand stats.
  BuildTreeStatsType stats;
  size_t dim = 3 + Rand() % 20;
  GenRandStats(dim, num_stats, N, P, phone_ids, *hmm_lengths, is_ctx_dep, ensure_all_covered, &stats);

  // Now build the tree.

  Questions qopts;
  int32 num_quest = 40, num_iters = 0;
  qopts.InitRand(stats, num_quest, num_iters, kAllKeysUnion);  // This was tested in build-tree-utils-test.cc

  float thresh = 100.0 * RandUniform();

  EventMap *tree = NULL;
  std::vector<std::vector<int32> > phone_sets(phone_ids.size());
  for (size_t i = 0; i < phone_ids.size(); i++)
    phone_sets[i].push_back(phone_ids[i]);
  std::vector<bool> share_roots(phone_sets.size(), true),
      do_split(phone_sets.size(), true);

  tree = BuildTree(qopts, phone_sets, *hmm_lengths, share_roots,
                   do_split, stats, thresh, 1000, 0.0, P);
  DeleteBuildTreeStats(&stats);
  return new ContextDependency(N, P, tree);
}


void ContextDependency::Write (std::ostream &os, bool binary) const {
  WriteToken(os, binary, "ContextDependency");
  WriteBasicType(os, binary, N_);
  WriteBasicType(os, binary, P_);
  WriteToken(os, binary, "ToPdf");
  to_pdf_->Write(os, binary);
  WriteToken(os, binary, "EndContextDependency");
}


void ContextDependency::Read (std::istream &is, bool binary) {
  if (to_pdf_) {
    delete to_pdf_;
    to_pdf_ = NULL;
  }
  ExpectToken(is, binary, "ContextDependency");
  ReadBasicType(is, binary, &N_);
  ReadBasicType(is, binary, &P_);
  EventMap *to_pdf = NULL;
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "ToLength") {  // back-compat.
    EventMap *to_num_pdf_classes = EventMap::Read(is, binary);
    if (to_num_pdf_classes) delete to_num_pdf_classes;
    ReadToken(is, binary, &token);
  }
  if (token == "ToPdf") {
    to_pdf = EventMap::Read(is , binary);
  } else {
    KALDI_ERR << "Got unexpected token " << token
              << " reading context-dependency object.";
  }
  ExpectToken(is, binary, "EndContextDependency");
  to_pdf_ = to_pdf;
}

void ContextDependency::GetPdfInfo(const std::vector<int32> &phones,
                                   const std::vector<int32> &num_pdf_classes,  // indexed by phone,
                                   std::vector<std::vector<std::pair<int32, int32> > > *pdf_info) const {  

  EventType vec;
  KALDI_ASSERT(pdf_info != NULL);
  pdf_info->resize(NumPdfs());
  for (size_t i = 0 ; i < phones.size(); i++) {
    int32 phone = phones[i];
    vec.clear();
    vec.push_back(std::make_pair(static_cast<EventKeyType>(P_),
                                 static_cast<EventValueType>(phone)));
    // Now get length.
    KALDI_ASSERT(static_cast<size_t>(phone) < num_pdf_classes.size());
    EventAnswerType len = num_pdf_classes[phone];

    for (int32 pos = 0; pos < len; pos++) {
      vec.resize(2);
      vec[0] = std::make_pair(static_cast<EventKeyType>(P_),
                              static_cast<EventValueType>(phone));
      vec[1] = std::make_pair(kPdfClass, static_cast<EventValueType>(pos));
      std::sort(vec.begin(), vec.end());
      std::vector<EventAnswerType> pdfs;  // pdfs that can be at this pos as this phone.
      to_pdf_->MultiMap(vec, &pdfs);
      SortAndUniq(&pdfs);
      if (pdfs.empty()) {
        KALDI_WARN << "ContextDependency::GetPdfInfo, no pdfs returned for position "<< pos << " of phone " << phone << ".   Continuing but this is a serious error.";
      }
      for (size_t j = 0; j < pdfs.size(); j++) {
        KALDI_ASSERT(static_cast<size_t>(pdfs[j]) < pdf_info->size());
        (*pdf_info)[pdfs[j]].push_back(std::make_pair(phone, pos));
      }
    }
  }
  for (size_t i = 0; i < pdf_info->size(); i++) {
    std::sort( ((*pdf_info)[i]).begin(),  ((*pdf_info)[i]).end());
    KALDI_ASSERT(IsSortedAndUniq( ((*pdf_info)[i])));  // should have no dups.
  }
}



ContextDependency*
MonophoneContextDependency(const std::vector<int32> phones,
                           const std::vector<int32> phone2num_pdf_classes) {
  std::vector<std::vector<int32> > phone_sets(phones.size());
  for (size_t i = 0; i < phones.size(); i++) phone_sets[i].push_back(phones[i]);
  std::vector<bool> share_roots(phones.size(), false);  // don't share roots.
  // N is context size, P = position of central phone (must be 0).
  int32 num_leaves = 0, P = 0, N = 1;
  EventMap *pdf_map = GetStubMap(P, phone_sets, phone2num_pdf_classes, share_roots, &num_leaves);
  return new ContextDependency(N, P, pdf_map);
}

ContextDependency*
MonophoneContextDependencyShared(const std::vector<std::vector<int32> > phone_sets,
                                 const std::vector<int32> phone2num_pdf_classes) {
  std::vector<bool> share_roots(phone_sets.size(), false);  // don't share roots.
  // N is context size, P = position of central phone (must be 0).
  int32 num_leaves = 0, P = 0, N = 1;
  EventMap *pdf_map = GetStubMap(P, phone_sets, phone2num_pdf_classes, share_roots, &num_leaves);
  return new ContextDependency(N, P, pdf_map);
}





} // end namespace kaldi.
