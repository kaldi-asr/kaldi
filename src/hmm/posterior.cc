// hmm/posterior.cc

// Copyright 2009-2011  Microsoft Corporation
//           2013-2014  Johns Hopkins University (author: Daniel Povey)
//                2014  Guoguo Chen
//                2014  Guoguo Chen

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

#include <vector>
#include "hmm/posterior.h"
#include "util/kaldi-table.h"
#include "util/stl-utils.h"


namespace kaldi {

// static
bool PosteriorHolder::Write(std::ostream &os, bool binary, const T &t) {
  InitKaldiOutputStream(os, binary);  // Puts binary header if binary mode.
  try {
    if (binary) {
      int32 sz = t.size();
      WriteBasicType(os, binary, sz);
      for (Posterior::const_iterator iter = t.begin(); iter != t.end(); ++iter) {
        int32 sz2 = iter->size();
        WriteBasicType(os, binary, sz2);
        for (std::vector<std::pair<int32, BaseFloat> >::const_iterator iter2=iter->begin();
             iter2 != iter->end();
             iter2++) {
          WriteBasicType(os, binary, iter2->first);
          WriteBasicType(os, binary, iter2->second);
        }
      }
    } else {  // In text-mode, choose a human-friendly, script-friendly format.
      // format is [ 1235 0.6 12 0.4 ] [ 34 1.0 ] ...
      // We could have used the same code as in the binary case above,
      // but this would have resulted in less readable output.
      for (Posterior::const_iterator iter = t.begin(); iter != t.end(); ++iter) {
        os << "[ ";
        for (std::vector<std::pair<int32, BaseFloat> >::const_iterator iter2=iter->begin();
             iter2 != iter->end();
             iter2++) {
          os << iter2->first << ' ' << iter2->second << ' ';
        }
        os << "] ";
      }
      os << '\n';  // newline terminate the record.
    }
    return os.good();
  } catch(const std::exception &e) {
    KALDI_WARN << "Exception caught writing table of posteriors";
    if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
    return false;  // Write failure.
  }
}

bool PosteriorHolder::Read(std::istream &is) {
  t_.clear();

  bool is_binary;
  if (!InitKaldiInputStream(is, &is_binary)) {
    KALDI_WARN << "Reading Table object, failed reading binary header";
    return false;
  }
  try {
    if (is_binary) {
      int32 sz;
      ReadBasicType(is, true, &sz);
      if (sz < 0)
        KALDI_ERR << "Reading posteriors: got negative size";
      t_.resize(sz);
      for (Posterior::iterator iter = t_.begin(); iter != t_.end(); ++iter) {
        int32 sz2;
        ReadBasicType(is, true, &sz2);
        if (sz2 < 0)
          KALDI_ERR << "Reading posteriors: got negative size";
        iter->resize(sz2);
        for (std::vector<std::pair<int32, BaseFloat> >::iterator iter2=iter->begin();
             iter2 != iter->end();
             iter2++) {
          ReadBasicType(is, true, &(iter2->first));
          ReadBasicType(is, true, &(iter2->second));
        }
      }
    } else {
      std::string line;
      getline(is, line);  // this will discard the \n, if present.
      if (is.fail()) {
        KALDI_WARN << "holder of Posterior: error reading line " << (is.eof() ? "[eof]" : "");
        return false;  // probably eof.  fail in any case.
      }
      std::istringstream line_is(line);
      while (1) {
        std::string str;
        line_is >> std::ws;  // eat up whitespace.
        if (line_is.eof()) break;
        line_is >> str;
        if (str != "[") KALDI_ERR << "Reading Posterior object: expecting [, got "
                                  << str << " (if this is an integer, possibly "
                            "you gave alignments in place of posteriors?)";
        std::vector<std::pair<int32, BaseFloat> > this_vec;
        while (1) {
          line_is >> std::ws;
          if (line_is.peek() == ']') {
            line_is.get();
            break;
          }
          int32 i; BaseFloat p;
          line_is >> i >> p;
          if (line_is.fail())
            KALDI_ERR << "Error reading Posterior object (could not get data after \"[\");";
          this_vec.push_back(std::make_pair(i, p));
        }
        t_.push_back(this_vec);
      }
    }
    return true;
  } catch (std::exception &e) {
    KALDI_WARN << "Exception caught reading table of posteriors";
    if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
    t_.clear();
    return false;
  }
}

// static
bool GaussPostHolder::Write(std::ostream &os, bool binary, const T &t) {
  InitKaldiOutputStream(os, binary);  // Puts binary header if binary mode.
  try {
    // We don't bother making this a one-line format.
    int32 sz = t.size();
    WriteBasicType(os, binary, sz);
    for (GaussPost::const_iterator iter = t.begin(); iter != t.end(); ++iter) {
      int32 sz2 = iter->size();
      WriteBasicType(os, binary, sz2);
      for (std::vector<std::pair<int32, Vector<BaseFloat> > >::const_iterator iter2=iter->begin();
           iter2 != iter->end();
           iter2++) {
        WriteBasicType(os, binary, iter2->first);
        iter2->second.Write(os, binary);
      }
    }
    if(!binary) os << '\n';
    return os.good();
  } catch (const std::exception &e) {
    KALDI_WARN << "Exception caught writing table of posteriors";
    if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
    return false;  // Write failure.
  }
}

bool GaussPostHolder::Read(std::istream &is) {
  t_.clear();

  bool is_binary;
  if (!InitKaldiInputStream(is, &is_binary)) {
    KALDI_WARN << "Reading Table object, failed reading binary header";
    return false;
  }
  try {
    int32 sz;
    ReadBasicType(is, is_binary, &sz);
    if (sz < 0)
      KALDI_ERR << "Reading posteriors: got negative size";
    t_.resize(sz);
    for (GaussPost::iterator iter = t_.begin(); iter != t_.end(); ++iter) {
      int32 sz2;
      ReadBasicType(is, is_binary, &sz2);
      if (sz2 < 0)
        KALDI_ERR << "Reading posteriors: got negative size";
      iter->resize(sz2);
      for (std::vector<std::pair<int32, Vector<BaseFloat> > >::iterator
               iter2=iter->begin();
           iter2 != iter->end();
           iter2++) {
        ReadBasicType(is, is_binary, &(iter2->first));
        iter2->second.Read(is, is_binary);
      }
    }
    return true;
  } catch (std::exception &e) {
    KALDI_WARN << "Exception caught reading table of posteriors";
    if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
    t_.clear();
    return false;
  }
}


void ScalePosterior(BaseFloat scale, Posterior *post) {
  if (scale == 1.0) return;
  for (size_t i = 0; i < post->size(); i++) {
    if (scale == 0.0) {
      (*post)[i].clear();
    } else {
      for (size_t j = 0; j < (*post)[i].size(); j++)
        (*post)[i][j].second *= scale;
    }
  }
}

BaseFloat TotalPosterior(const Posterior &post) {
  double sum =  0.0;
  size_t T = post.size();
  for (size_t t = 0; t < T; t++) {
    size_t I = post[t].size();
    for (size_t i = 0; i < I; i++) {
      sum += post[t][i].second;
    }
  }
  return sum;
}

bool PosteriorEntriesAreDisjoint(
    const std::vector<std::pair<int32,BaseFloat> > &post_elem1,
    const std::vector<std::pair<int32,BaseFloat> > &post_elem2) {
  unordered_set<int32> set1;
  for (size_t i = 0; i < post_elem1.size(); i++) set1.insert(post_elem1[i].first);
  for (size_t i = 0; i < post_elem2.size(); i++)
    if (set1.count(post_elem2[i].first) != 0) return false;
  return true; // The sets are disjoint.
}

// For each frame, merges the posteriors in post1 into post2,
// frame-by-frame, combining any duplicated entries.
// note: Posterior is vector<vector<pair<int32, BaseFloat> > >
// Returns the number of frames for which the two posteriors
// were disjoint (no common transition-ids or whatever index
// we are using).
int32 MergePosteriors(const Posterior &post1,
                      const Posterior &post2,
                      bool merge,
                      bool drop_frames,
                      Posterior *post) {
  KALDI_ASSERT(post1.size() == post2.size()); // precondition.
  post->resize(post1.size());

  int32 num_disjoint = 0;
  for (size_t i = 0; i < post->size(); i++) {
    (*post)[i].reserve(post1[i].size() + post2[i].size());
    (*post)[i].insert((*post)[i].end(),
                      post1[i].begin(), post1[i].end());
    (*post)[i].insert((*post)[i].end(),
                      post2[i].begin(), post2[i].end());
    if (merge) { // combine and sum up entries with same transition-id.
      MergePairVectorSumming(&((*post)[i])); // This sorts on
      // the transition-id merges the entries with the same
      // key (i.e. same .first element; same transition-id), and
      // gets rid of entries with zero .second element.
    } else { // just to keep them pretty, merge them.
      std::sort( (*post)[i].begin(), (*post)[i].end() );
    }
    if (PosteriorEntriesAreDisjoint(post1[i], post2[i])) {
      num_disjoint++;
      if (drop_frames)
        (*post)[i].clear();
    }
  }
  return num_disjoint;
}

void AlignmentToPosterior(const std::vector<int32> &ali,
                          Posterior *post) {
  post->clear();
  post->resize(ali.size());
  for (size_t i = 0; i < ali.size(); i++) {
    (*post)[i].resize(1);
    (*post)[i][0].first = ali[i];
    (*post)[i][0].second = 1.0;
  }
}

struct ComparePosteriorByPdfs {
  const TransitionModel *tmodel_;
  ComparePosteriorByPdfs(const TransitionModel &tmodel): tmodel_(&tmodel) {}
  bool operator() (const std::pair<int32, BaseFloat> &a,
                   const std::pair<int32, BaseFloat> &b) {
    if (tmodel_->TransitionIdToPdf(a.first)
        < tmodel_->TransitionIdToPdf(b.first))
      return true;
    else
      return false;
  }
};

void SortPosteriorByPdfs(const TransitionModel &tmodel,
                         Posterior *post) {
  ComparePosteriorByPdfs compare(tmodel);
  for (size_t i = 0; i < post->size(); i++) {
    sort((*post)[i].begin(), (*post)[i].end(), compare);
  }
}

void ConvertPosteriorToPdfs(const TransitionModel &tmodel,
                            const Posterior &post_in,
                            Posterior *post_out) {
  post_out->clear();
  post_out->resize(post_in.size());
  for (size_t i = 0; i < post_out->size(); i++) {
    unordered_map<int32, BaseFloat> pdf_to_post;
    for (size_t j = 0; j < post_in[i].size(); j++) {
      int32 tid = post_in[i][j].first,
          pdf_id = tmodel.TransitionIdToPdf(tid);
      BaseFloat post = post_in[i][j].second;
      if (pdf_to_post.count(pdf_id) == 0)
        pdf_to_post[pdf_id] = post;
      else
        pdf_to_post[pdf_id] += post;
    }
    (*post_out)[i].reserve(pdf_to_post.size());
    for (unordered_map<int32, BaseFloat>::const_iterator iter =
             pdf_to_post.begin(); iter != pdf_to_post.end(); ++iter) {
      if (iter->second != 0.0)
        (*post_out)[i].push_back(
            std::make_pair(iter->first, iter->second));
    }
  }
}

void ConvertPosteriorToPhones(const TransitionModel &tmodel,
                              const Posterior &post_in,
                              Posterior *post_out) {
  post_out->clear();
  post_out->resize(post_in.size());
  for (size_t i = 0; i < post_out->size(); i++) {
    std::map<int32, BaseFloat> phone_to_post;
    for (size_t j = 0; j < post_in[i].size(); j++) {
      int32 tid = post_in[i][j].first,
          phone_id = tmodel.TransitionIdToPhone(tid);
      BaseFloat post = post_in[i][j].second;
      if (phone_to_post.count(phone_id) == 0)
        phone_to_post[phone_id] = post;
      else
        phone_to_post[phone_id] += post;
    }
    (*post_out)[i].reserve(phone_to_post.size());
    for (std::map<int32, BaseFloat>::const_iterator iter =
             phone_to_post.begin(); iter != phone_to_post.end(); ++iter) {
      if (iter->second != 0.0)
        (*post_out)[i].push_back(
            std::make_pair(iter->first, iter->second));
    }
  }
}


void WeightSilencePost(const TransitionModel &trans_model,
                       const ConstIntegerSet<int32> &silence_set,
                       BaseFloat silence_scale,
                       Posterior *post) {
  for (size_t i = 0; i < post->size(); i++) {
    std::vector<std::pair<int32, BaseFloat> > this_post;
    this_post.reserve((*post)[i].size());
    for (size_t j = 0; j < (*post)[i].size(); j++) {
      int32 tid = (*post)[i][j].first,
          phone = trans_model.TransitionIdToPhone(tid);
      BaseFloat weight = (*post)[i][j].second;
      if (silence_set.count(phone) != 0) {  // is a silence.
        if (silence_scale != 0.0)
          this_post.push_back(std::make_pair(tid, weight*silence_scale));
      } else {
        this_post.push_back(std::make_pair(tid, weight));
      }
    }
    (*post)[i].swap(this_post);
  }
}


void WeightSilencePostDistributed(const TransitionModel &trans_model,
                                  const ConstIntegerSet<int32> &silence_set,
                                  BaseFloat silence_scale,
                                  Posterior *post) {
  for (size_t i = 0; i < post->size(); i++) {
    std::vector<std::pair<int32, BaseFloat> > this_post;
    this_post.reserve((*post)[i].size());
    BaseFloat sil_weight = 0.0, nonsil_weight = 0.0;   
    for (size_t j = 0; j < (*post)[i].size(); j++) {
      int32 tid = (*post)[i][j].first,
          phone = trans_model.TransitionIdToPhone(tid);
      BaseFloat weight = (*post)[i][j].second;
      if (silence_set.count(phone) != 0) sil_weight += weight;
      else nonsil_weight += weight;
    }
    KALDI_ASSERT(sil_weight >= 0.0 && nonsil_weight >= 0.0); // This "distributed"
    // weighting approach doesn't make sense if we have negative weights.
    if (sil_weight + nonsil_weight == 0.0) continue;
    BaseFloat frame_scale = (sil_weight * silence_scale + nonsil_weight) /
                            (sil_weight + nonsil_weight);
    if (frame_scale != 0.0) {
      for (size_t j = 0; j < (*post)[i].size(); j++) {
        int32 tid = (*post)[i][j].first;
        BaseFloat weight = (*post)[i][j].second;    
        this_post.push_back(std::make_pair(tid, weight * frame_scale));
      }
    }
    (*post)[i].swap(this_post);    
  }
}

// comparator object that can be used to sort from greatest to
// least posterior.
struct CompareReverseSecond {
  // view this as an "<" operator used for sorting, except it behaves like
  // a ">" operator on the .second field of the pair because we want the
  // sort to be in reverse order (greatest to least) on posterior.
  bool operator() (const std::pair<int32, BaseFloat> &a,
                   const std::pair<int32, BaseFloat> &b) {
    return (a.second > b.second);
  }
};

BaseFloat VectorToPosteriorEntry(
    const VectorBase<BaseFloat> &log_likes,
    int32 num_gselect,
    BaseFloat min_post,
    std::vector<std::pair<int32, BaseFloat> > *post_entry) {
  KALDI_ASSERT(num_gselect > 0 && min_post >= 0 && min_post < 1.0);
  // we name num_gauss assuming each entry in log_likes represents a Gaussian;
  // it doesn't matter if they don't.
  int32 num_gauss = log_likes.Dim();
  KALDI_ASSERT(num_gauss > 0);
  if (num_gselect > num_gauss)
    num_gselect = num_gauss;
  Vector<BaseFloat> log_likes_normalized(log_likes);
  BaseFloat ans = log_likes_normalized.ApplySoftMax();
  std::vector<std::pair<int32, BaseFloat> > temp_post(num_gauss);
  for (int32 g = 0; g < num_gauss; g++)
    temp_post[g] = std::pair<int32, BaseFloat>(g, log_likes_normalized(g));
  CompareReverseSecond compare;
  // Sort in decreasing order on posterior.  For efficiency we
  // first do nth_element and then sort, as we only need the part we're
  // going to output, to be sorted.
  std::nth_element(temp_post.begin(),
                   temp_post.begin() + num_gselect, temp_post.end(),
                   compare);
  std::sort(temp_post.begin(), temp_post.begin() + num_gselect,
            compare);

  post_entry->clear();
  post_entry->insert(post_entry->end(),
                     temp_post.begin(), temp_post.begin() + num_gselect);
  while (post_entry->size() > 1 && post_entry->back().second < min_post)
    post_entry->pop_back();  
  // Now renormalize to sum to one after pruning.
  BaseFloat tot = 0.0;
  size_t size = post_entry->size();
  for (size_t i = 0; i < size; i++)
    tot += (*post_entry)[i].second;
  BaseFloat inv_tot = 1.0 / tot;
  for (size_t i = 0; i < size; i++)
    (*post_entry)[i].second *= inv_tot;
  return ans;
}


} // End namespace kaldi
