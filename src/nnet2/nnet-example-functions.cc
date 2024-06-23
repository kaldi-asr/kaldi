// nnet2/nnet-example-functions.cc

// Copyright 2012-2013  Johns Hopkins University (author: Daniel Povey)

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

#include "nnet2/nnet-example-functions.h"
#include "lat/lattice-functions.h"
#include "lat/lattice-functions-transition-model.h"

namespace kaldi {
namespace nnet2 {


bool LatticeToDiscriminativeExample(
    const std::vector<int32> &alignment,
    const Matrix<BaseFloat> &feats,
    const CompactLattice &clat,
    BaseFloat weight,
    int32 left_context,
    int32 right_context,
    DiscriminativeNnetExample *eg) {
  KALDI_ASSERT(left_context >= 0 && right_context >= 0);
  int32 num_frames = alignment.size();
  if (num_frames == 0) {
    KALDI_WARN << "Empty alignment";
    return false;
  }
  if (num_frames != feats.NumRows()) {
    KALDI_WARN << "Dimension mismatch: alignment " << num_frames
               << " versus feats " << feats.NumRows();
    return false;
  }
  std::vector<int32> times;
  int32 num_frames_clat = CompactLatticeStateTimes(clat, &times);  
  if (num_frames_clat != num_frames) {
    KALDI_WARN << "Numerator/frames versus denlat frames mismatch: "
               << num_frames << " versus " << num_frames_clat;
    return false;
  }
  eg->weight = weight;
  eg->num_ali = alignment;
  eg->den_lat = clat;

  int32 feat_dim = feats.NumCols();
  eg->input_frames.Resize(left_context + num_frames + right_context,
                          feat_dim);
  eg->input_frames.Range(left_context, num_frames,
                         0, feat_dim).CopyFromMat(feats);

  // Duplicate the first and last frames.
  for (int32 t = 0; t < left_context; t++)
    eg->input_frames.Row(t).CopyFromVec(feats.Row(0));
  for (int32 t = 0; t < right_context; t++)
    eg->input_frames.Row(left_context + num_frames + t).CopyFromVec(
        feats.Row(num_frames - 1));

  eg->left_context = left_context;
  eg->Check();
  return true;
}






/**
   For each frame, judge:
     - does it produce a nonzero derivative? [this differs MMI vs MPE]
     - can it be split here [or what is the penalty for splitting here.]
         - depends whether lattice has just one path at that point.

   Time taken to process segment of a certain length: [must be sub-linear.]
      [use quadratic function that's max at specified segment length and zero at zero.]

   No penalty for processing frames we don't need to process (already implicit in
   segment-processing time above.)

   Penalty for splitting where we should not split.  [Make it propto log(#paths).]
   
 */





class DiscriminativeExampleSplitter {
 public:
  DiscriminativeExampleSplitter(
      const SplitDiscriminativeExampleConfig &config,
      const TransitionModel &tmodel,
      const DiscriminativeNnetExample &eg,
      std::vector<DiscriminativeNnetExample> *egs_out):
      config_(config), tmodel_(tmodel), eg_(eg), egs_out_(egs_out) { }

  void Excise(SplitExampleStats *stats) {
    eg_.Check();
    PrepareLattice(false);
    ComputeFrameInfo();
    if (!config_.excise) {
      egs_out_->resize(1);
      (*egs_out_)[0] = eg_;
    } else {
      DoExcise(stats);
    }
  }
  
  void Split(SplitExampleStats *stats) {
    if (!config_.split) {
      egs_out_->resize(1);
      (*egs_out_)[0] = eg_;
    } else {
      eg_.Check();    
      PrepareLattice(true);
      ComputeFrameInfo();
      DoSplit(stats);
    }
  }

 private:
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Label Label;

  // converts compact lattice to lat_.  You should set first_time to true if
  // this is being called from DoSplit, but false if being called from DoExcise
  // (this saves some time, since we avoid some preparation steps that we know
  // are unnecessary because they were done before
  void PrepareLattice(bool first_time); 

  void CollapseTransitionIds(); // Modifies the transition-ids on lat_ so that
                                // on each frame, there is just one with any
                                // given pdf-id.  This allows us to determinize
                                // and minimize more completely.
  
  bool ComputeFrameInfo();

  static void RemoveAllOutputSymbols (Lattice *lat);

  void OutputOneSplit(int32 seg_begin, int32 seg_end);
  
  void DoSplit(SplitExampleStats *stats);

  void DoExcise(SplitExampleStats *stats);
  
  int32 NumFrames() const { return static_cast<int32>(eg_.num_ali.size()); }

  int32 RightContext() { return eg_.input_frames.NumRows() - NumFrames() - eg_.left_context; }
  

  // Put in lat_out, a slice of "clat" with first frame at time "seg_begin" and
  // with last frame at time "seg_end - 1".
  void CreateOutputLattice(int32 seg_begin, int32 seg_end,
                           CompactLattice *clat_out);

  // Returns the state-id in this output lattice (creates a
  // new state if needed).
  StateId GetOutputStateId(StateId s,
                           unordered_map<StateId, StateId> *state_map,
                           Lattice *lat_out);           

  struct FrameInfo {
    int32 den_state_count;
    int32 den_pdf_count; // number of distinct pdfs in denominator lattice
    bool multiple_transition_ids; // true if there are multiple distinct
                                  // transition-ids in the denominator lattice
                                  // at this point
    bool num_den_overlap; // true if num and den overlap.

    bool nonzero_derivative; // True if we need to keep this frame because the
    // derivative is nonzero on this frame.
    bool can_excise_frame; // True if the frame, if part of a segment, can be
    // excised, *but ignoring the effect of acoustic
    // context*.  I.e. true if the likelihoods and
    // derivatives from this frame do not matter because
    // the derivatives are zero and the likelihoods don't
    // affect lattice posteriors (because pdfs are all
    // the same on this frame, or if doing mpfe,
    // transition-ids are all the same.

    // start_state says, for a segment starting at frame t, what is the
    // earliest state in lat_ that we have to consider including in the split
    // lattice?  This relates to a kind of optimization for efficiency.
    StateId start_state;

    // end_state says, for a segment whose final frame is time t (i.e.  whose
    // "segment end" is time t+1), what is the latest state in lat_ that we have
    // to consider including in the split lattice?  This relates to a kind of
    // optimization for efficiency.
    StateId end_state;  
    FrameInfo(): den_state_count(0), den_pdf_count(0),
                 multiple_transition_ids(false),
                 num_den_overlap(false), nonzero_derivative(false),
                 can_excise_frame(false),
                 start_state(std::numeric_limits<int32>::max()), end_state(0) { }
  };
  
  
  // The following variables are set in the initializer:
  const SplitDiscriminativeExampleConfig &config_;
  const TransitionModel &tmodel_;
  const DiscriminativeNnetExample &eg_;
  std::vector<DiscriminativeNnetExample> *egs_out_;
  
  Lattice lat_; // lattice generated from eg_.den_lat, with epsilons removed etc.


  // The other variables are computed by Split() or functions called from it.

  std::vector<FrameInfo> frame_info_;
  
  // state_times_ says, for each state in lat_, what its start time is.
  std::vector<int32> state_times_;

};

// Make sure that for any given pdf-id and any given frame, the den-lat has
// only one transition-id mapping to that pdf-id, on the same frame.
// It helps us to more completely minimize the lattice.  Note: we
// can't do this if the criterion is MPFE, because in that case the
// objective function will be affected by the phone-identities being
// different even if the pdf-ids are the same.
void DiscriminativeExampleSplitter::CollapseTransitionIds() {
  std::vector<int32> times;
  TopSort(&lat_); // Topologically sort the lattice (required by
                  // LatticeStateTimes)
  int32 num_frames = LatticeStateTimes(lat_, &times);  
  StateId num_states = lat_.NumStates();

  std::vector<std::map<int32, int32> > pdf_to_tid(num_frames);
  for (StateId s = 0; s < num_states; s++) {
    int32 t = times[s];
    for (fst::MutableArcIterator<Lattice> aiter(&lat_, s);
         !aiter.Done(); aiter.Next()) {
      KALDI_ASSERT(t >= 0 && t < num_frames);
      Arc arc = aiter.Value();
      KALDI_ASSERT(arc.ilabel != 0 && arc.ilabel == arc.olabel);
      int32 pdf = tmodel_.TransitionIdToPdf(arc.ilabel);
      if (pdf_to_tid[t].count(pdf) != 0) {
        arc.ilabel = arc.olabel = pdf_to_tid[t][pdf];
        aiter.SetValue(arc);
      } else {
        pdf_to_tid[t][pdf] = arc.ilabel;
      }
    }
  }    
}


void DiscriminativeExampleSplitter::PrepareLattice(bool first_time) {
  ::fst::ConvertLattice(eg_.den_lat, &lat_);

  Project(&lat_, fst::PROJECT_INPUT); // Get rid of the word labels and put the
                                      // transition-ids on both sides.
  
  RmEpsilon(&lat_); // Remove epsilons.. this simplifies
                    // certain things.

  if (first_time) {
    if (config_.collapse_transition_ids && config_.criterion != "mpfe")
      CollapseTransitionIds();
  
    if (config_.determinize) {
      if (!config_.minimize) {
        Lattice det_lat;
        Determinize(lat_, &det_lat);
        lat_ = det_lat;
      } else {
        Lattice tmp_lat;
        Reverse(lat_, &tmp_lat);
        Determinize(tmp_lat, &lat_);
        Reverse(lat_, &tmp_lat);
        Determinize(tmp_lat, &lat_);
        RmEpsilon(&lat_);
        // Previously we determinized, then did
        // Minimize(&lat_);
        // but this was too slow.
      }
    }
  }
  TopSort(&lat_); // Topologically sort the lattice.
}

// this function computes various arrays that say something about
// this frame of the lattice.
bool DiscriminativeExampleSplitter::ComputeFrameInfo() {
  
  int32 num_frames = NumFrames();

  frame_info_.clear();
  frame_info_.resize(num_frames + 1);
  
  LatticeStateTimes(lat_, &state_times_);

  std::vector<std::set<int32> > pdfs_per_frame(num_frames),
      tids_per_frame(num_frames);
  
  int32 num_states = lat_.NumStates();
  
  for (int32 state = 0; state < num_states; state++) {
    int32 t = state_times_[state];
    KALDI_ASSERT(t >= 0 && t <= num_frames);
    frame_info_[t].den_state_count++;
    for (fst::ArcIterator<Lattice> aiter(lat_, state); !aiter.Done();
         aiter.Next()) {
      const LatticeArc &arc = aiter.Value();
      KALDI_ASSERT(arc.ilabel != 0 && arc.ilabel == arc.olabel); 
      int32 transition_id = arc.ilabel,
          pdf_id = tmodel_.TransitionIdToPdf(transition_id);
      tids_per_frame[t].insert(transition_id);
      pdfs_per_frame[t].insert(pdf_id);
    }
    if (t < num_frames)
      frame_info_[t+1].start_state = std::min(state,
                                              frame_info_[t+1].start_state);
    frame_info_[t].end_state = std::max(state,
                                        frame_info_[t].end_state);
  }

  for (int32 i = 1; i <= NumFrames(); i++)
    frame_info_[i].end_state = std::max(frame_info_[i-1].end_state,
                                        frame_info_[i].end_state);
  for (int32 i = NumFrames() - 1; i >= 0; i--)
    frame_info_[i].start_state = std::min(frame_info_[i+1].start_state,
                                          frame_info_[i].start_state);
  
  for (int32 t = 0; t < num_frames; t++) {
    FrameInfo &frame_info = frame_info_[t];
    int32 transition_id = eg_.num_ali[t],
        pdf_id = tmodel_.TransitionIdToPdf(transition_id);
    frame_info.num_den_overlap = (pdfs_per_frame[t].count(pdf_id) != 0);
    frame_info.multiple_transition_ids = (tids_per_frame[t].size() > 1);
    KALDI_ASSERT(!pdfs_per_frame[t].empty());
    frame_info.den_pdf_count = pdfs_per_frame[t].size();

    if (config_.criterion == "mpfe" || config_.criterion == "smbr") {
      frame_info.nonzero_derivative = (frame_info.den_pdf_count > 1);
    } else {
      KALDI_ASSERT(config_.criterion == "mmi");
      if (config_.drop_frames) {
        // With frame dropping, we'll get nonzero derivative only
        // if num and den overlap, *and* den has >1 active pdf.
        frame_info.nonzero_derivative = frame_info.num_den_overlap  &&
            frame_info.den_state_count > 1;
      } else {
        // Without frame dropping, we'll get nonzero derivative if num and den
        // do not overlap , or den has >1 active pdf.
        frame_info.nonzero_derivative = !frame_info.num_den_overlap ||
            frame_info.den_state_count > 1;
      }
    }
    // If a frame is part of a segment, but it's not going to contribute
    // to the derivative and the den lattice has only one pdf active
    // at that time, then this frame can be excised from the lattice
    // because it will not affect the posteriors around it.
    if (config_.criterion == "mpfe") {
      frame_info.can_excise_frame =
          !frame_info.nonzero_derivative && \
          !frame_info.multiple_transition_ids;
      // in the mpfe case, if there are multiple transition-ids on a
      // frame there may be multiple phones on a frame, which could
      // contribute to the objective function even if they share pdf-ids.
      // (this was an issue that came up during testing).
    } else {      
      frame_info.can_excise_frame =
          !frame_info.nonzero_derivative && frame_info.den_pdf_count == 1;
    }
  }
  return true;
}


/* Excising a frame means removing a frame from the lattice and removing the
   corresponding feature.  We can only do this if it would not affect the
   derivatives because the current frame has zero derivative and also all the
   den-lat pdfs are the same on this frame (so removing the frame doesn't affect
   the lattice posteriors).  But we can't remove a frame if doing so would
   affect the acoustic context.  Generally speaking we must keep all frames
   that are within LeftContext() to the left and RightContext() to the right
   of a frame that we can't excise, *but* it's OK at the edges of a segment
   even if they are that close to other frames, because we anyway keep a few
   frames of context at the edges, and we can just make sure to keep the
   *right* few frames of context.
   */
void DiscriminativeExampleSplitter::DoExcise(SplitExampleStats *stats) {
  int32 left_context = eg_.left_context,
      right_context = RightContext(),
      num_frames = NumFrames();
  // Compute, for each frame, whether we can excise it.
  // 
  std::vector<bool> can_excise(num_frames, false);
  
  bool need_some_frame = false;
  for (int32 t = 0; t < num_frames; t++) {
    can_excise[t] = frame_info_[t].can_excise_frame;
    if (!can_excise[t])
      need_some_frame = true;
  }
  if (!need_some_frame) { // We don't need any frame within this file, so simply
                          // delete the segment.
    KALDI_WARN << "Example completely removed when excising."; // unexpected,
    // as the segment should have been deleted when splitting.
    egs_out_->clear();
    return;
  }
  egs_out_->resize(1);
  DiscriminativeNnetExample &eg_out = (*egs_out_)[0];

  // start_t and end_t will be the central part of the segment, excluding any
  // frames at the edges that we can excise.
  int32 start_t, end_t;
  for (start_t = 0; can_excise[start_t]; start_t++);
  for (end_t = num_frames; can_excise[end_t-1]; end_t--);

  // for frames from start_t to end_t-1, do not excise them if
  // they are within the context-window of a frame that we need to keep.
  // Note: we do t2 = t - right_context to t + left_context, because we're
  // concerned whether frame t2 has frame t in its window... it might
  // seem a bit backwards.
  std::vector<bool> will_excise(can_excise);
  for (int32 t = start_t; t < end_t; t++) {
    for (int32 t2 = t - right_context; t2 <= t + left_context; t2++)
      if (t2 >= start_t && t2 < end_t && !can_excise[t2])
        will_excise[t] = false; // can't excise this frame, it's needed for
                                // context.
  }

  // Remove all un-needed frames from the lattice by replacing the
  // symbols with epsilon and then removing the epsilons.
  // Note, this operation is destructive (it changes lat_).
  int32 num_states = lat_.NumStates();
  for (int32 state = 0; state < num_states; state++) {
    int32 t = state_times_[state];
    for (::fst::MutableArcIterator<Lattice> aiter(&lat_, state); !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      if (will_excise[t]) {
        arc.ilabel = arc.olabel = 0;
        aiter.SetValue(arc);
      }
    }
  }
  RmEpsilon(&lat_);
  RemoveAllOutputSymbols(&lat_);
  ConvertLattice(lat_, &eg_out.den_lat);

  eg_out.num_ali.clear();
  int32 num_frames_kept = 0;
  for (int32 t = 0; t < num_frames; t++) {
    if (!will_excise[t]) {
      eg_out.num_ali.push_back(eg_.num_ali[t]);
      num_frames_kept++;
    }
  }

  stats->num_frames_kept_after_excise += num_frames_kept;
  stats->longest_segment_after_excise = std::max(stats->longest_segment_after_excise,
                                                 num_frames_kept);
  
  int32 num_frames_kept_plus = num_frames_kept + left_context + right_context;
  eg_out.input_frames.Resize(num_frames_kept_plus,
                             eg_.input_frames.NumCols());

  // the left-context of the output will be shifted to the right by
  // start_t.
  for (int32 i = 0; i < left_context; i++) {
    SubVector<BaseFloat> dst(eg_out.input_frames, i);
    SubVector<BaseFloat> src(eg_.input_frames, start_t + i);
    dst.CopyFromVec(src);
  }
  // the right-context will also be shifted, we take the frames
  // to the right of end_t.
  for (int32 i = 0; i < right_context; i++) {
    SubVector<BaseFloat> dst(eg_out.input_frames,
                             num_frames_kept + left_context + i);
    SubVector<BaseFloat> src(eg_.input_frames,
                             end_t + left_context + i);
    dst.CopyFromVec(src);
  }
  // now copy the central frames (those that were not excised).
  int32 dst_t = 0;
  for (int32 t = start_t; t < end_t; t++) {
    if (!will_excise[t]) {
      SubVector<BaseFloat> dst(eg_out.input_frames,
                               left_context + dst_t);
      SubVector<BaseFloat> src(eg_.input_frames,
                               left_context + t);
      dst.CopyFromVec(src);
      dst_t++;
    }
  }
  KALDI_ASSERT(dst_t == num_frames_kept);


  eg_out.weight = eg_.weight;
  eg_out.left_context = eg_.left_context;
  eg_out.spk_info = eg_.spk_info;

  eg_out.Check();
}


void DiscriminativeExampleSplitter::DoSplit(SplitExampleStats *stats) {
  std::vector<int32> split_points;
  int32 num_frames = NumFrames();
  {
    // Make the "split points" 0 and num_frames, and
    // any frame that has just one state on it and the previous
    // frame had >1 state.  This gives us one split for each
    // "pinch point" in the lattice.  Later we may move each split
    // to a more optimal location.
    split_points.push_back(0);
    for (int32 t = 1; t < num_frames; t++) {
      if (frame_info_[t].den_state_count == 1 &&
          frame_info_[t-1].den_state_count > 1)
        split_points.push_back(t);
    }
    split_points.push_back(num_frames);
  }

  std::vector<bool> is_kept(split_points.size() - 1);
  { // A "split" is a pair of successive split points.  Work out for each split
    // whether we must keep it (we must if it contains at least one frame for
    // which "nonzero_derivative" == true.)
    for (size_t s = 0; s < is_kept.size(); s++) {
      int32 start = split_points[s], end = split_points[s+1];
      bool keep_this_split = false;
      for (int32 t = start; t < end; t++)
        if (frame_info_[t].nonzero_derivative)
          keep_this_split = true;
      is_kept[s] = keep_this_split;
    }
  }

  egs_out_->clear();
  egs_out_->reserve(is_kept.size());

  stats->num_lattices++;
  stats->longest_lattice = std::max(stats->longest_lattice, num_frames);
  stats->num_segments += is_kept.size();
  stats->num_frames_orig += num_frames;
  for (int32 t = 0; t < num_frames; t++)
    if (frame_info_[t].nonzero_derivative)
      stats->num_frames_must_keep++;
  
  for (size_t s = 0; s < is_kept.size(); s++) {
    if (is_kept[s]) {
      stats->num_kept_segments++;
      OutputOneSplit(split_points[s], split_points[s+1]);
      int32 segment_len = split_points[s+1] - split_points[s];
      stats->num_frames_kept_after_split += segment_len;
      stats->longest_segment_after_split =
          std::max(stats->longest_segment_after_split, segment_len);
    }
  }
}



void SplitExampleStats::Print() {
  KALDI_LOG << "Split " << num_lattices << " lattices.  Stats:";
  double kept_segs_per_lat = num_kept_segments * 1.0 / num_lattices,
      segs_per_lat = num_segments * 1.0 / num_lattices;
      
  KALDI_LOG << "Made on average " << segs_per_lat << " segments per lattice, "
            << "of which " << kept_segs_per_lat << " were kept.";

  double percent_needed = num_frames_must_keep * 100.0 / num_frames_orig,
    percent_after_split = num_frames_kept_after_split * 100.0 / num_frames_orig,
   percent_after_excise = num_frames_kept_after_excise * 100.0 / num_frames_orig;
      
  KALDI_LOG << "Needed to keep " << percent_needed << "% of frames, after split "
            << "kept " << percent_after_split << "%, after excising frames kept "
            << percent_after_excise << "%.";

  KALDI_LOG << "Longest lattice had " << longest_lattice
            << " frames, longest segment after splitting had "
            << longest_segment_after_split
            << " frames, longest segment after excising had "
            << longest_segment_after_excise;
}

void DiscriminativeExampleSplitter::OutputOneSplit(int32 seg_begin,
                                                   int32 seg_end) {
  KALDI_ASSERT(seg_begin >= 0 && seg_end > seg_begin && seg_end <= NumFrames());
  egs_out_->resize(egs_out_->size() + 1);
  int32 left_context = eg_.left_context, right_context = RightContext(),
      tot_context = left_context + right_context;
  DiscriminativeNnetExample &eg_out = egs_out_->back();
  eg_out.weight = eg_.weight;

  eg_out.num_ali.insert(eg_out.num_ali.end(),
                        eg_.num_ali.begin() + seg_begin,
                        eg_.num_ali.begin() + seg_end);

  CreateOutputLattice(seg_begin, seg_end, &(eg_out.den_lat));
  
  eg_out.input_frames = eg_.input_frames.Range(seg_begin, seg_end - seg_begin +
                                               tot_context,
                                               0, eg_.input_frames.NumCols());

  eg_out.left_context = eg_.left_context;

  eg_out.spk_info = eg_.spk_info;
  
  eg_out.Check();  
}

// static
void DiscriminativeExampleSplitter::RemoveAllOutputSymbols(Lattice *lat) {
  for (StateId s = 0; s < lat->NumStates(); s++) {
    for (::fst::MutableArcIterator<Lattice> aiter(lat, s); !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      arc.olabel = 0;
      aiter.SetValue(arc);
    }
  }  
}

DiscriminativeExampleSplitter::StateId
DiscriminativeExampleSplitter::GetOutputStateId(
    StateId s, unordered_map<StateId, StateId> *state_map, Lattice *lat_out) {
  if (state_map->count(s) == 0) {
    return ((*state_map)[s] = lat_out->AddState());
  } else {
    return (*state_map)[s];
  }
}

void DiscriminativeExampleSplitter::CreateOutputLattice(
    int32 seg_begin, int32 seg_end,
    CompactLattice *clat_out) {
  Lattice lat_out;

  // Below, state_map will map from states in the original lattice
  // lat_ to ones in the new lattice lat_out.
  unordered_map<StateId, StateId> state_map;

  // The range of the loop over s could be made over the
  // entire lattice, but we limit it for efficiency.
  
  for (StateId s = frame_info_[seg_begin].start_state;
       s <= frame_info_[seg_end].end_state; s++) {
    int32 t = state_times_[s];

    if (t < seg_begin || t > seg_end) // state out of range.
      continue;

    int32 this_state = GetOutputStateId(s, &state_map, &lat_out);

    if (t == seg_begin) // note: we only split on frames with just one
      lat_out.SetStart(this_state); // state, so we reach this only once.
    
    if (t == seg_end) { // Make it final and don't process its arcs out.
      if (seg_end == NumFrames()) {
        lat_out.SetFinal(this_state, lat_.Final(s));
      } else {
        lat_out.SetFinal(this_state, LatticeWeight::One());
      }
      continue; // don't process arcs out of this state.
    }
    
    for (fst::ArcIterator<Lattice> aiter(lat_, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      StateId next_state = GetOutputStateId(arc.nextstate,
                                            &state_map, &lat_out);
      KALDI_ASSERT(arc.ilabel != 0 && arc.ilabel == arc.olabel); // We expect no epsilons.
      lat_out.AddArc(this_state, Arc(arc.ilabel, arc.olabel, arc.weight,
                                      next_state));
    }
  }
  Connect(&lat_out); // this is not really necessary, it's only to make sure
                     // the assert below fails when it should. TODO: remove it.
  KALDI_ASSERT(lat_out.NumStates() > 0);
  RemoveAllOutputSymbols(&lat_out);
  ConvertLattice(lat_out, clat_out);
}

/*
void DiscriminativeExampleSplitter::SelfTest() {
  bool splits_ok = true; // True iff we split only
                         // on frames where there was
                         // one arc crossing.

  // we can't do any of this excising frames if we want to
  // preserve equivalence.
  std::fill(can_excise_.begin(), can_excise_.end(), false);
  
  std::vector<Lattice*> split_lats;

  int32 cur_t = NumFrames();
  while (cur_t != 0) {
    Backtrace this_backtrace = backtrace_[cur_t];
    int32 prev_t = this_backtrace.prev_frame;

    int32 seg_begin = prev_t, seg_end = cur_t;
    Lattice *new_lat = new Lattice();
    CreateOutputLattice(seg_begin, seg_end, new_lat);
    split_lats.push_back(new_lat);

    if (split_penalty_[cur_t] != 0)
      splits_ok = false; // we split where there was a penalty so we don't
                         //  expect equivalence.
    cur_t = prev_t;
  }
  KALDI_ASSERT(!split_lats.empty());
  std::reverse(split_lats.begin(), split_lats.end());
  for (size_t i = 1; i < split_lats.size(); i++) {
    // append split_lats[i] to split_lats[0], putting the
    // result in split_lats[0].
    Concat(split_lats[0], *(split_lats[i]));
  }
  Connect(split_lats[0]);
  KALDI_ASSERT(split_lats[0]->NumStates() > 0);
  

  if (!splits_ok) {
    KALDI_LOG << "Not self-testing because we split where there were "
              << "multiple paths.";
    
  } else {
    if (!(RandEquivalent(*(split_lats[0]), lat_, 5, 0.01,
         Rand(), 100))) {
      KALDI_WARN << "Lattices were not equivalent (self-test failed).";
      KALDI_LOG << "Original lattice was: ";
      WriteLattice(std::cerr, false, lat_);
      KALDI_LOG << "New lattice is:";
      WriteLattice(std::cerr, false, *(split_lats[0]));
      {
        Lattice best_path_orig;
        ShortestPath(lat_, &best_path_orig);
        KALDI_LOG << "Original best path was:";
        WriteLattice(std::cerr, false, best_path_orig);
      }
      {
        Lattice best_path_new;
        ShortestPath(*(split_lats[0]), &best_path_new);
        KALDI_LOG << "New best path was:";
        WriteLattice(std::cerr, false, best_path_new);
      }
    }
  }
  for (size_t i = 0; i < split_lats.size(); i++)
    delete split_lats[i];
}
*/



void SplitDiscriminativeExample(
    const SplitDiscriminativeExampleConfig &config,
    const TransitionModel &tmodel,
    const DiscriminativeNnetExample &eg,
    std::vector<DiscriminativeNnetExample> *egs_out,
    SplitExampleStats *stats_out) {
  DiscriminativeExampleSplitter splitter(config, tmodel, eg, egs_out);
  splitter.Split(stats_out);
}


void ExciseDiscriminativeExample(
    const SplitDiscriminativeExampleConfig &config,
    const TransitionModel &tmodel,
    const DiscriminativeNnetExample &eg,
    std::vector<DiscriminativeNnetExample> *egs_out,    
    SplitExampleStats *stats_out) {
  DiscriminativeExampleSplitter splitter(config, tmodel, eg, egs_out);
  splitter.Excise(stats_out);
}


void UpdateHash(
    const TransitionModel &tmodel,
    const DiscriminativeNnetExample &eg,
    std::string criterion,
    bool drop_frames,
    bool one_silence_class,
    Matrix<double> *hash,
    double *num_weight,
    double *den_weight,
    double *tot_t) {
  int32 feat_dim = eg.input_frames.NumCols(),
      left_context = eg.left_context,
      num_frames = eg.num_ali.size(),
      right_context = eg.input_frames.NumRows() - num_frames - left_context,
      context_width = left_context + 1 + right_context;
  *tot_t += num_frames;
  KALDI_ASSERT(right_context >= 0);
  KALDI_ASSERT(hash != NULL);
  if (hash->NumRows() == 0) {
    hash->Resize(tmodel.NumPdfs(), feat_dim);
  } else {
    KALDI_ASSERT(hash->NumRows() == tmodel.NumPdfs() &&
                 hash->NumCols() == feat_dim);
  }

  Posterior post;
  std::vector<int32> silence_phones; // we don't let the user specify this
                                     // because it's not necessary for testing
                                     // purposes -> leave it empty
  ExampleToPdfPost(tmodel, silence_phones, criterion, drop_frames,
                   one_silence_class, eg, &post);

  Vector<BaseFloat> avg_feat(feat_dim);
  
  for (int32 t = 0; t < num_frames; t++) {
    SubMatrix<BaseFloat> context_window(eg.input_frames,
                                        t, context_width,
                                        0, feat_dim);
    // set avg_feat to average over the context-window for this frame.
    avg_feat.AddRowSumMat(1.0 / context_width, context_window, 0.0);
    Vector<double> avg_feat_dbl(avg_feat);
    for (size_t i = 0; i < post[t].size(); i++) {
      int32 pdf_id = post[t][i].first;
      BaseFloat weight = post[t][i].second;
      hash->Row(pdf_id).AddVec(weight, avg_feat_dbl);
      if (weight > 0.0) *num_weight += weight;
      else *den_weight += -weight;
    }
  }
}


void ExampleToPdfPost(
    const TransitionModel &tmodel,
    const std::vector<int32> &silence_phones,    
    std::string criterion,
    bool drop_frames,
    bool one_silence_class,
    const DiscriminativeNnetExample &eg,
    Posterior *post) {
  KALDI_ASSERT(criterion == "mpfe" || criterion == "smbr" || criterion == "mmi");
  
  Lattice lat;
  ConvertLattice(eg.den_lat, &lat);
  TopSort(&lat);
  if (criterion == "mpfe" || criterion == "smbr") {
    Posterior tid_post;
    LatticeForwardBackwardMpeVariants(tmodel, silence_phones, lat, eg.num_ali,
                                      criterion, one_silence_class, &tid_post);
    
    ConvertPosteriorToPdfs(tmodel, tid_post, post);
  } else {
    bool convert_to_pdf_ids = true, cancel = true;
    LatticeForwardBackwardMmi(tmodel, lat, eg.num_ali,
                              drop_frames, convert_to_pdf_ids, cancel,
                              post);
  }
  ScalePosterior(eg.weight, post);
}


void SolvePackingProblem(BaseFloat max_cost,
                         const std::vector<BaseFloat> &costs,
                         std::vector<std::vector<size_t> > *groups) {
  groups->clear();
  std::vector<BaseFloat> group_costs;
  for (size_t i = 0; i < costs.size(); i++) {
    bool found_group = false;
    BaseFloat this_cost = costs[i];
    for (size_t j = 0; j < groups->size(); j++) {
      if (group_costs[j] + this_cost <= max_cost) {
        (*groups)[j].push_back(i);
        group_costs[j] += this_cost;
        found_group = true;
        break;
      }
    }
    if (!found_group) { // Put this object in a newly created group.
      groups->resize(groups->size() + 1);
      groups->back().push_back(i);
      group_costs.push_back(this_cost);
    }
  }
}

void AppendDiscriminativeExamples(
    const std::vector<const DiscriminativeNnetExample*> &input,
    DiscriminativeNnetExample *output) {
  KALDI_ASSERT(!input.empty());
  const DiscriminativeNnetExample &eg0 = *(input[0]);
  
  int32 dim = eg0.input_frames.NumCols() + eg0.spk_info.Dim(),
      left_context = eg0.left_context,
      num_frames = eg0.num_ali.size(),
      right_context = eg0.input_frames.NumRows() - num_frames - left_context;

  int32 tot_frames = eg0.input_frames.NumRows();  // total frames (appended,
                                                  // with context)
  for (size_t i = 1; i < input.size(); i++)
    tot_frames += input[i]->input_frames.NumRows();

  int32 arbitrary_tid = 1;  // arbitrary transition-id that we use to pad the
                            // num_ali and den_lat members between segments
                            // (since they're both the same, and the den-lat in
                            // those parts is linear, they contribute no
                            // derivative to the training).
  
  output->den_lat = eg0.den_lat;
  output->num_ali = eg0.num_ali;
  output->input_frames.Resize(tot_frames, dim, kUndefined);
  output->input_frames.Range(0, eg0.input_frames.NumRows(),
                             0, eg0.input_frames.NumCols()).CopyFromMat(eg0.input_frames);
  if (eg0.spk_info.Dim() != 0) {
    output->input_frames.Range(0, eg0.input_frames.NumRows(),
                               eg0.input_frames.NumCols(), eg0.spk_info.Dim()).
        CopyRowsFromVec(eg0.spk_info);
  }
  
  output->num_ali.reserve(tot_frames - left_context - right_context);
  output->weight = eg0.weight;
  output->left_context = eg0.left_context;
  output->spk_info.Resize(0);

  CompactLattice inter_segment_clat;
  int32 initial = inter_segment_clat.AddState(); // state 0.
  inter_segment_clat.SetStart(initial);
  
  std::vector<int32> inter_segment_ali(left_context + right_context);
  std::fill(inter_segment_ali.begin(), inter_segment_ali.end(), arbitrary_tid);

  CompactLatticeWeight final_weight = CompactLatticeWeight::One();
  final_weight.SetString(inter_segment_ali);
  inter_segment_clat.SetFinal(initial, final_weight);
  
  int32 feat_offset = eg0.input_frames.NumRows();
  
  for (size_t i = 1; i < input.size(); i++) {
    const DiscriminativeNnetExample &eg_i = *(input[i]);
        
    output->input_frames.Range(feat_offset, eg_i.input_frames.NumRows(),
                               0, eg_i.input_frames.NumCols()).CopyFromMat(
                                   eg_i.input_frames);
    if (eg_i.spk_info.Dim() != 0) {
      output->input_frames.Range(feat_offset, eg_i.input_frames.NumRows(),
                                 eg_i.input_frames.NumCols(),
                                 eg_i.spk_info.Dim()).CopyRowsFromVec(
                                     eg_i.spk_info);
      KALDI_ASSERT(eg_i.input_frames.NumCols() +
                   eg_i.spk_info.Dim() == dim);
    }
    
    output->num_ali.insert(output->num_ali.end(),
                           inter_segment_ali.begin(), inter_segment_ali.end());
    output->num_ali.insert(output->num_ali.end(),
                           eg_i.num_ali.begin(), eg_i.num_ali.end());
    Concat(&(output->den_lat), inter_segment_clat);
    Concat(&(output->den_lat), eg_i.den_lat);
    KALDI_ASSERT(output->weight == eg_i.weight);
    KALDI_ASSERT(output->left_context == eg_i.left_context);
    feat_offset += eg_i.input_frames.NumRows();
  }
  KALDI_ASSERT(feat_offset == tot_frames);
}
  
void CombineDiscriminativeExamples(
    int32 max_length,
    const std::vector<DiscriminativeNnetExample> &input,
    std::vector<DiscriminativeNnetExample> *output) {
  
  std::vector<BaseFloat> costs(input.size());
  for (size_t i = 0; i < input.size(); i++)
    costs[i] = static_cast<BaseFloat>(input[i].input_frames.NumRows());
  std::vector<std::vector<size_t> > groups;
  SolvePackingProblem(max_length,
                      costs,
                      &groups);
  output->clear();
  output->resize(groups.size());
  for (size_t i = 0; i < groups.size(); i++) {
    std::vector<const DiscriminativeNnetExample*> group_egs;
    for (size_t j = 0; j < groups[i].size(); j++) {
      size_t index = groups[i][j];
      group_egs.push_back(&(input[index]));
    }
    AppendDiscriminativeExamples(group_egs, &((*output)[i]));
  }
}



} // namespace nnet2
} // namespace kaldi
