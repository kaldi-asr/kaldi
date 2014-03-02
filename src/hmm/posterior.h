// hmm/posterior.h

// Copyright 2009-2011     Microsoft Corporation
//           2013-2014     Johns Hopkins University (author: Daniel Povey)
//                2014     Guoguo Chen


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

#ifndef KALDI_HMM_POSTERIOR_H_
#define KALDI_HMM_POSTERIOR_H_

#include "base/kaldi-common.h"
#include "tree/context-dep.h"
#include "util/const-integer-set.h"
#include "util/kaldi-table.h"
#include "hmm/transition-model.h"


namespace kaldi {


/// \addtogroup posterior_group
/// @{

/// Posterior is a typedef for storing acoustic-state (actually, transition-id)
/// posteriors over an utterance.  The "int32" is a transition-id, and the BaseFloat
/// is a probability (typically between zero and one).
typedef std::vector<std::vector<std::pair<int32, BaseFloat> > > Posterior;

/// GaussPost is a typedef for storing Gaussian-level posteriors for an utterance.
/// the "int32" is a transition-id, and the Vector<BaseFloat> is a vector of
/// Gaussian posteriors.
/// WARNING: We changed "int32" from transition-id to pdf-id, and the change is
/// applied for all programs using GaussPost. This is for efficiency purpose. We
/// also changed the name slightly from GauPost to GaussPost to reduce the
/// chance that the change will go un-noticed in downstream code.
typedef std::vector<std::vector<std::pair<int32, Vector<BaseFloat> > > > GaussPost;


// PosteriorHolder is a holder for Posterior, which is
// std::vector<std::vector<std::pair<int32, BaseFloat> > >
// This is used for storing posteriors of transition id's for an
// utterance.
class PosteriorHolder {
 public:
  typedef Posterior T;

  PosteriorHolder() { }

  static bool Write(std::ostream &os, bool binary, const T &t);
  
  void Clear() { Posterior tmp; std::swap(tmp, t_); }

  // Reads into the holder.
  bool Read(std::istream &is);
  
  // Kaldi objects always have the stream open in binary mode for
  // reading.
  static bool IsReadInBinary() { return true; }

  const T &Value() const { return t_; }
  
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(PosteriorHolder);
  T t_;
};


// GaussPostHolder is a holder for GaussPost, which is
// std::vector<std::vector<std::pair<int32, Vector<BaseFloat> > > >
// This is used for storing posteriors of transition id's for an
// utterance.
class GaussPostHolder {
 public:
  typedef GaussPost T;

  GaussPostHolder() { }

  static bool Write(std::ostream &os, bool binary, const T &t);  

  void Clear() {  GaussPost tmp;  std::swap(tmp, t_); }

  // Reads into the holder.
  bool Read(std::istream &is);
  
  // Kaldi objects always have the stream open in binary mode for
  // reading.
  static bool IsReadInBinary() { return true; }

  const T &Value() const { return t_; }
  
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(GaussPostHolder);
  T t_;
};


// Posterior is a typedef: vector<vector<pair<int32, BaseFloat> > >,
// representing posteriors over (typically) transition-ids for an
// utterance.
typedef TableWriter<PosteriorHolder> PosteriorWriter;
typedef SequentialTableReader<PosteriorHolder> SequentialPosteriorReader;
typedef RandomAccessTableReader<PosteriorHolder> RandomAccessPosteriorReader;


// typedef std::vector<std::vector<std::pair<int32, Vector<BaseFloat> > > > GaussPost;
typedef TableWriter<GaussPostHolder> GaussPostWriter;
typedef SequentialTableReader<GaussPostHolder> SequentialGaussPostReader;
typedef RandomAccessTableReader<GaussPostHolder> RandomAccessGaussPostReader;


/// Scales the BaseFloat (weight) element in the posterior entries.
void ScalePosterior(BaseFloat scale, Posterior *post);


/// Returns true if the two lists of pairs have no common .first element.
bool PosteriorEntriesAreDisjoint(
    const std::vector<std::pair<int32, BaseFloat> > &post_elem1,
    const std::vector<std::pair<int32, BaseFloat> > &post_elem2);


/// Merge two sets of posteriors, which must have the same length.  If "merge"
/// is true, it will make a common entry whenever there are duplicated entries,
/// adding up the weights.  If "drop_frames" is true, for frames where the
/// two sets of posteriors were originally disjoint, makes no entries for that
/// frame (relates to frame dropping, or drop_frames, see Vesely et al, ICASSP
/// 2013).  Returns the number of frames for which the two posteriors were
/// disjoint (i.e. no common transition-ids or whatever index we are using).
int32 MergePosteriors(const Posterior &post1,
                      const Posterior &post2,
                      bool merge,
                      bool drop_frames,
                      Posterior *post);

/// Convert an alignment to a posterior (with a scale of 1.0 on
/// each entry).
void AlignmentToPosterior(const std::vector<int32> &ali,
                          Posterior *post);

/// Sorts posterior entries so that transition-ids with same pdf-id are next to
/// each other.
void SortPosteriorByPdfs(const TransitionModel &tmodel,
                         Posterior *post);

/// Converts a posterior over transition-ids to be a posterior
/// over pdf-ids.
void ConvertPosteriorToPdfs(const TransitionModel &tmodel,
                            const Posterior &post_in,
                            Posterior *post_out);

/// Converts a posterior over transition-ids to be a posterior
/// over phones.
void ConvertPosteriorToPhones(const TransitionModel &tmodel,
                              const Posterior &post_in,
                              Posterior *post_out);

/// Weight any silence phones in the posterior (i.e. any phones
/// in the set "silence_set" by scale "silence_scale".
/// The interface was changed in Feb 2014 to do the modification
/// "in-place" rather than having separate input and output.
void WeightSilencePost(const TransitionModel &trans_model,
                       const ConstIntegerSet<int32> &silence_set,
                       BaseFloat silence_scale,
                       Posterior *post);

/// This is similar to WeightSilencePost, except that on each frame it
/// works out the amount by which the overall posterior would be reduced,
/// and scales down everything on that frame by the same amount.  It
/// has the effect that frames that are mostly silence get down-weighted.
/// The interface was changed in Feb 2014 to do the modification
/// "in-place" rather than having separate input and output.
void WeightSilencePostDistributed(const TransitionModel &trans_model,
                                  const ConstIntegerSet<int32> &silence_set,
                                  BaseFloat silence_scale,
                                  Posterior *post);

/// @} end "addtogroup posterior_group"


} // end namespace kaldi


#endif
