// kws/kws-scoring.cc

// Copyright (c) 2015, Johns Hopkins University (Yenda Trmal<jtrmal@gmail.com>)

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

#include <utility>
#include <vector>
#include <limits>
#include <algorithm>

#include "kws/kws-scoring.h"

namespace kaldi {
namespace kws_internal {

class KwTermLower {
 public:
  explicit KwTermLower(const int threshold): threshold_(threshold) {}

  bool operator() (const KwsTerm &left, const KwsTerm &right) {
    if ( (left.start_time() + threshold_) < right.start_time() ) {
      return true;
    } else {
      return (left.end_time() + threshold_) < right.end_time();
    }
  }

 private:
    const int threshold_;
};

class KwTermEqual {
 public:
  KwTermEqual(const int max_distance, const KwsTerm &inst):
    max_distance_(max_distance), inst_(inst) {}

  bool operator() (const KwsTerm &left, const KwsTerm &right) {
    bool ret = true;

    ret &= (left.kw_id() == right.kw_id());
    ret &= (left.utt_id() == right.utt_id());

    float center_left = (left.start_time() + left.end_time())/2;
    float center_right = (right.start_time() + right.end_time())/2;

// This was an old definition of the criterion "the hyp is within
// max_distance_ area from the ref". The positive thing about the
// definition is, that it allows binary search through the collection
//        ret &= fabs(left.tbeg - right.tbeg) <= max_distance_;
//        ret &= fabs(left.tend - right.tend) <= max_distance_;

// This is the newer definition -- should be equivalent to what F4DE uses
    ret &= fabs(center_left - center_right) <= max_distance_;

    return ret;
  }

  bool operator() (const KwsTerm &right) {
    return (*this)(inst_, right);
  }

 private:
    const int max_distance_;
    const KwsTerm inst_;
};


struct KwScoreStats {
  int32 nof_corr;
  int32 nof_fa;
  int32 nof_misses;
  int32 nof_corr_ndet;
  int32 nof_unseen;
  int32 nof_targets;

  KwScoreStats(): nof_corr(0),
                  nof_fa(0),
                  nof_misses(0),
                  nof_corr_ndet(0),
                  nof_unseen(0),
                  nof_targets(0) {}
};

struct ThrSweepStats {
  int32 nof_corr;
  int32 nof_fa;

  ThrSweepStats(): nof_corr(0),
                   nof_fa(0) {}
};

typedef unordered_map <float, ThrSweepStats> SweepThresholdStats;
typedef unordered_map <std::string, KwScoreStats> KwStats;
typedef unordered_map <std::string, SweepThresholdStats> PerKwSweepStats;

}  // namespace kws_internal


void KwsTermsAlignerOptions::Register(OptionsItf *opts) {
  opts->Register("max_distance", &max_distance,
                 "Max distance on the ref and hyp centers "
                 "to be considered as a potential match");
}

KwsTermsAligner::KwsTermsAligner(const KwsTermsAlignerOptions &opts):
  opts_(opts),
  nof_refs_(0),
  nof_hyps_(0) { }


KwsAlignment KwsTermsAligner::AlignTerms() {
  KwsAlignment alignment;

  used_ref_terms_.clear();
  std::list<KwsTerm>::iterator it = hyps_.begin();
  for (; it != hyps_.end(); ++it) {
    AlignedTermsPair ref_hyp_pair;
    ref_hyp_pair.hyp = *it;
    ref_hyp_pair.aligner_score = -std::numeric_limits<float>::infinity();

    int ref_idx = FindBestRefIndex(*it);
    if (ref_idx >= 0) {  // If found
      int utt_id = it->utt_id();
      std::string kw_id = it->kw_id();

      ref_hyp_pair.ref = refs_[utt_id][kw_id][ref_idx];
      used_ref_terms_[utt_id][kw_id][ref_idx] = true;
      ref_hyp_pair.aligner_score = AlignerScore(ref_hyp_pair.ref,
                                                ref_hyp_pair.hyp);
    }

    alignment.Add(ref_hyp_pair);
  }
  KALDI_LOG << "Alignment size before adding unseen: " << alignment.size();
  // Finally, find the terms in ref which have not been seen in hyp
  // and add them into the alignment
  FillUnmatchedRefs(&alignment);
  KALDI_LOG << "Alignment size after  adding unseen: " << alignment.size();
  return alignment;
}

void KwsTermsAligner::FillUnmatchedRefs(KwsAlignment *ali) {
  // We have to traverse the whole ref_ structure and check
  // against the used_ref_terms_ structure if the given ref term
  // was already used or not. If not, we will add it to the alignment
  typedef unordered_map<std::string, TermArray> KwList;
  typedef KwList::iterator KwIndex;
  typedef unordered_map<int, KwList >::iterator  UttIndex;

  for (UttIndex utt = refs_.begin(); utt != refs_.end(); ++utt) {
    int utt_id = utt->first;
    for (KwIndex kw = refs_[utt_id].begin(); kw != refs_[utt_id].end(); ++kw) {
      std::string kw_id = kw->first;
      for (TermIterator term = refs_[utt_id][kw_id].begin();
                        term != refs_[utt_id][kw_id].end(); ++term ) {
        int idx = term - refs_[utt_id][kw_id].begin();
        if (!used_ref_terms_[utt_id][kw_id][idx]) {
          AlignedTermsPair missed_hyp;
          missed_hyp.aligner_score = -std::numeric_limits<float>::infinity();
          missed_hyp.ref = refs_[utt_id][kw_id][idx];
          ali->Add(missed_hyp);
        }
      }
    }
  }
}

int KwsTermsAligner::FindBestRefIndex(const KwsTerm &term) {
  if (!RefExistsMaybe(term)) {
    return -1;
  }
  int utt_id = term.utt_id();
  std::string kw_id = term.kw_id();

  TermIterator start_mark = refs_[utt_id][kw_id].begin();
  TermIterator end_mark = refs_[utt_id][kw_id].end();

  TermIterator it = FindNextRef(term, start_mark, end_mark);
  if (it == end_mark) {
    return  -1;
  }

  int   best_ref_idx = -1;
  float best_ref_score = -std::numeric_limits<float>::infinity();
  do {
    float current_score = AlignerScore(*it, term);
    int current_index = it - start_mark;
    if ((current_score > best_ref_score) &&
        (!used_ref_terms_[utt_id][kw_id][current_index])) {
      best_ref_idx = current_index;
      best_ref_score = current_score;
    }

    it = FindNextRef(term, ++it, end_mark);
  } while (it != end_mark);

  return best_ref_idx;
}


bool KwsTermsAligner::RefExistsMaybe(const KwsTerm &term) {
  int utt_id = term.utt_id();
  std::string kw_id = term.kw_id();
  if (refs_.count(utt_id) != 0) {
    if (refs_[utt_id].count(kw_id) != 0) {
      return true;
    }
  }
  return false;
}



KwsTermsAligner::TermIterator KwsTermsAligner::FindNextRef(
                                          const KwsTerm &ref,
                                          const TermIterator &prev,
                                          const TermIterator &last) {
  return std::find_if(prev, last,
      kws_internal::KwTermEqual(opts_.max_distance, ref));
}

float KwsTermsAligner::AlignerScore(const KwsTerm &ref, const KwsTerm &hyp) {
  float overlap = std::min(ref.end_time(), hyp.end_time())
                  - std::max(ref.start_time(), hyp.start_time());
  float join = std::max(ref.end_time(), hyp.end_time())
               - std::min(ref.start_time(), hyp.start_time());
  return static_cast<float>(overlap) / join;
}

void KwsAlignment::WriteCsv(std::iostream &os, const float frames_per_sec) {
  AlignedTerms::const_iterator it = begin();
  os << "language,file,channel,termid,term,ref_bt,ref_et,"
    << "sys_bt,sys_et,sys_score,sys_decision,alignment\n";

  while ( it != end() ) {
    int file = it->ref.valid() ? it->ref.utt_id() : it->hyp.utt_id();
    std::string termid = it->ref.valid() ? it->ref.kw_id() : it->hyp.kw_id();
    std::string term = termid;
    std::string lang = "";
    int channel = 1;

    os << lang << ","
      << file << ","
      << channel << ","
      << termid << ","
      << term << ",";

    if (it->ref.valid()) {
      os << it->ref.start_time() / static_cast<float>(frames_per_sec) << ","
        << it->ref.end_time() / static_cast<float>(frames_per_sec) << ",";
    } else {
      os << "," << ",";
    }
    if (it->hyp.valid()) {
      os << it->hyp.start_time() / static_cast<float>(frames_per_sec) << ","
        << it->hyp.end_time() / static_cast<float>(frames_per_sec) << ","
        << it->hyp.score() << ","
        << (it->hyp.score() >= 0.5 ? "YES" : "NO") << ",";
    } else {
      os << "," << "," << "," << ",";
    }

    if (it->ref.valid() && it->hyp.valid()) {
      os << (it->hyp.score() >= 0.5 ? "CORR" : "MISS");
    } else if (it->ref.valid()) {
      os << "MISS";
    } else if (it->hyp.valid()) {
      os << (it->hyp.score() >= 0.5 ? "FA" : "CORR!DET");
    }
    os << std::endl;
    it++;
  }
}


TwvMetricsOptions::TwvMetricsOptions(): cost_fa(0.1f),
                                        value_corr(1.0f),
                                        prior_probability(1e-4f),
                                        score_threshold(0.5f),
                                        sweep_step(0.05f),
                                        audio_duration(0.0f) {}

void TwvMetricsOptions::Register(OptionsItf *opts) {
  opts->Register("cost-fa", &cost_fa,
                 "The cost of an incorrect detection");
  opts->Register("value-corr", &value_corr,
                 "The value (gain) of a correct detection");
  opts->Register("prior-kw-probability", &prior_probability,
                 "The prior probability of a keyword");
  opts->Register("score-threshold", &score_threshold,
                 "The score threshold for computation of ATWV");
  opts->Register("sweep-step", &sweep_step,
                 "Size of the bin during sweeping for the oracle measures");

  // We won't set the audio duration here, as it's supposed to be
  // a mandatory argument, not optional
}

class TwvMetricsStats {
 public:
  kws_internal::KwScoreStats global_keyword_stats;
  kws_internal::KwStats keyword_stats;
  kws_internal::PerKwSweepStats otwv_sweep_cache;
  std::list<float> sweep_threshold_values;
};

TwvMetrics::TwvMetrics(const TwvMetricsOptions &opts):
  audio_duration_(opts.audio_duration),
  atwv_decision_threshold_(opts.score_threshold),
  beta_(opts.beta()) {
  stats_ = new TwvMetricsStats();
  if (opts.sweep_step > 0.0) {
    for (float i=0.0; i <= 1; i+=opts.sweep_step) {
       stats_->sweep_threshold_values.push_back(i);
    }
  }
}

TwvMetrics::~TwvMetrics() {
  delete stats_;
}

void TwvMetrics::AddEvent(const KwsTerm &ref,
                          const KwsTerm &hyp,
                          float ali_score) {
  if (ref.valid() && hyp.valid()) {
    RefAndHypSeen(hyp.kw_id(), hyp.score());
  } else if (hyp.valid()) {
    OnlyHypSeen(hyp.kw_id(), hyp.score());
  } else if (ref.valid()) {
    OnlyRefSeen(ref.kw_id(), ref.score());
  } else {
    KALDI_ASSERT(ref.valid() || hyp.valid());
  }
}

void TwvMetrics::RefAndHypSeen(const std::string &kw_id, float score) {
  std::list<float>::iterator i = stats_->sweep_threshold_values.begin();
  for (; i != stats_->sweep_threshold_values.end(); ++i) {
    float decision_threshold = *i;
    if ( score >= decision_threshold )
      stats_->otwv_sweep_cache[kw_id][decision_threshold].nof_corr++;
  }
  if (score >= atwv_decision_threshold_) {
    stats_->global_keyword_stats.nof_corr++;
    stats_->keyword_stats[kw_id].nof_corr++;
  } else {
    stats_->global_keyword_stats.nof_misses++;
    stats_->keyword_stats[kw_id].nof_misses++;
  }
  stats_->global_keyword_stats.nof_targets++;
  stats_->keyword_stats[kw_id].nof_targets++;
}

void TwvMetrics::OnlyHypSeen(const std::string &kw_id, float score) {
  std::list<float>::iterator i = stats_->sweep_threshold_values.begin();
  for (; i != stats_->sweep_threshold_values.end(); ++i) {
    float decision_threshold = *i;
    if ( score >= decision_threshold )
      stats_->otwv_sweep_cache[kw_id][decision_threshold].nof_fa++;
  }
  if (score >= atwv_decision_threshold_) {
    stats_->global_keyword_stats.nof_fa++;
    stats_->keyword_stats[kw_id].nof_fa++;
  } else {
    stats_->global_keyword_stats.nof_corr_ndet++;
    stats_->keyword_stats[kw_id].nof_corr_ndet++;
  }
}

void TwvMetrics::OnlyRefSeen(const std::string &kw_id, float score) {
  stats_->global_keyword_stats.nof_targets++;
  stats_->keyword_stats[kw_id].nof_targets++;
  stats_->global_keyword_stats.nof_unseen++;
  stats_->keyword_stats[kw_id].nof_unseen++;
}

void TwvMetrics::AddAlignment(const KwsAlignment &ali) {
  KwsAlignment::AlignedTerms::const_iterator it = ali.begin();
  int k = 0;
  while (it != ali.end()) {
    AddEvent(it->ref, it->hyp, it->aligner_score);
    ++it;
    ++k;
  }
  KALDI_VLOG(4) << "Processed " << k << " alignment entries";
}

void TwvMetrics::Reset() {
  delete stats_;
  stats_ = new TwvMetricsStats;
}

float TwvMetrics::Atwv() {
  typedef kws_internal::KwStats::iterator KwIterator;
  int32 nof_kw = 0;
  float atwv = 0;

  for (KwIterator it = stats_->keyword_stats.begin();
                  it != stats_->keyword_stats.end(); ++it ) {
    if (it->second.nof_targets == 0) {
      continue;
    }
    float nof_targets = static_cast<float>(it->second.nof_targets);
    float pmiss = 1 - it->second.nof_corr / nof_targets;
    float pfa = it->second.nof_fa / (audio_duration_ - nof_targets);
    float twv = 1 - pmiss - beta_ * pfa;

    atwv = atwv * (nof_kw)/(nof_kw + 1.0) + twv / (nof_kw + 1.0);
    nof_kw++;
  }
  return atwv;
}

float TwvMetrics::Stwv() {
  typedef kws_internal::KwStats::iterator KwIterator;
  int32 nof_kw = 0;
  float stwv = 0;

  for (KwIterator it = stats_->keyword_stats.begin();
                  it != stats_->keyword_stats.end(); ++it ) {
    if (it->second.nof_targets == 0) {
      continue;
    }
    float nof_targets = static_cast<float>(it->second.nof_targets);
    float recall = 1 - it->second.nof_unseen / nof_targets;

    stwv = stwv * (nof_kw)/(nof_kw + 1.0) + recall / (nof_kw + 1.0);
    nof_kw++;
  }
  return stwv;
}

void TwvMetrics::GetOracleMeasures(float *final_mtwv,
                                  float *final_mtwv_threshold,
                                  float *final_otwv) {
  typedef kws_internal::KwStats::iterator KwIterator;

  int32 nof_kw = 0;
  float otwv = 0;

  unordered_map<float, double> mtwv_sweep;
  for (KwIterator it = stats_->keyword_stats.begin();
                  it != stats_->keyword_stats.end(); ++it ) {
    if (it->second.nof_targets == 0) {
      continue;
    }
    std::string kw_id = it->first;

    float local_otwv = -9999;
    float local_otwv_threshold = -1.0;
    std::list<float>::iterator i = stats_->sweep_threshold_values.begin();
    for (; i != stats_->sweep_threshold_values.end(); ++i) {
      float decision_threshold = *i;

      float nof_targets = static_cast<float>(it->second.nof_targets);
      float nof_true = stats_->otwv_sweep_cache[kw_id][decision_threshold].nof_corr;
      float nof_fa = stats_->otwv_sweep_cache[kw_id][decision_threshold].nof_fa;
      float pmiss = 1 - nof_true / nof_targets;
      float pfa = nof_fa / (audio_duration_ - nof_targets);
      float twv = 1 - pmiss - beta_ * pfa;

      if (twv > local_otwv) {
        local_otwv = twv;
        local_otwv_threshold = decision_threshold;
      }
      mtwv_sweep[decision_threshold] = twv / (nof_kw + 1.0) +
            mtwv_sweep[decision_threshold] * (nof_kw)/(nof_kw + 1.0);
    }
    KALDI_ASSERT(local_otwv_threshold >= 0);
    otwv = otwv * (nof_kw)/(nof_kw + 1.0) + local_otwv / (nof_kw + 1.0);
    nof_kw++;
  }

  float mtwv = -9999;
  float mtwv_threshold = -1;
  std::list<float>::iterator i = stats_->sweep_threshold_values.begin();
  for (; i != stats_->sweep_threshold_values.end(); ++i) {
    float decision_threshold = *i;

    if (mtwv_sweep[decision_threshold] > mtwv) {
      mtwv = mtwv_sweep[decision_threshold];
      mtwv_threshold = decision_threshold;
    }
  }
  KALDI_ASSERT(mtwv_threshold >= 0);
  *final_mtwv = mtwv;
  *final_mtwv_threshold = mtwv_threshold;
  *final_otwv = otwv;
}
}  // namespace kaldi


