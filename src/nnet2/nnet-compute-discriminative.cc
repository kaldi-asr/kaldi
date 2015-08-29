// nnet2/nnet-compute-discriminative.cc

// Copyright 2012-2013   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet2/nnet-compute-discriminative.h"
#include "hmm/posterior.h"
#include "lat/lattice-functions.h"

namespace kaldi {
namespace nnet2 {

/*
  This class does the forward and possibly backward computation for (typically)
  a whole utterance of contiguous features.  You'll instantiate one of
  these classes each time you want to do this computation.
*/
class NnetDiscriminativeUpdater {
 public:

  NnetDiscriminativeUpdater(const AmNnet &am_nnet,
                            const TransitionModel &tmodel,
                            const NnetDiscriminativeUpdateOptions &opts,
                            const DiscriminativeNnetExample &eg,
                            Nnet *nnet_to_update,
                            NnetDiscriminativeStats *stats);

  void Update() {
    Propagate();
    LatticeComputations();
    if (nnet_to_update_ != NULL)
      Backprop();
  }
  
  /// The forward-through-the-layers part of the computation.
  void Propagate();  

  /// Does the parts between Propagate() and Backprop(), that
  /// involve forward-backward over the lattice.
  void LatticeComputations();
  
  void Backprop();

  /// Assuming the lattice already has the correct scores in
  /// it, this function does the MPE or MMI forward-backward
  /// and puts the resulting discriminative posteriors (which
  /// may have positive or negative weight) into "post".
  /// It returns, for MPFE/SMBR, the objective function, or
  /// for MMI, the negative of the denominator-lattice log-likelihood.
  double GetDiscriminativePosteriors(Posterior *post);
  
  SubMatrix<BaseFloat> GetInputFeatures() const;
  
  CuMatrixBase<BaseFloat> &GetOutput() { return forward_data_.back(); }

  static inline Int32Pair MakePair(int32 first, int32 second) {
    Int32Pair ans;
    ans.first = first;
    ans.second = second;
    return ans;
  }
  
 private:
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;

  
  const AmNnet &am_nnet_;
  const TransitionModel &tmodel_;
  const NnetDiscriminativeUpdateOptions &opts_;
  const DiscriminativeNnetExample &eg_;
  Nnet *nnet_to_update_; // will equal am_nnet_.GetNnet(), in SGD case, or
                         // another Nnet, in gradient-computation case, or
                         // NULL if we just need the objective function.
  NnetDiscriminativeStats *stats_; // the objective function, etc.
  std::vector<ChunkInfo> chunk_info_out_; 
  // forward_data_[i] is the input of the i'th component and (if i > 0)
  // the output of the i-1'th component.
  std::vector<CuMatrix<BaseFloat> > forward_data_; 
  Lattice lat_; // we convert the CompactLattice in the eg, into Lattice form.
  CuMatrix<BaseFloat> backward_data_;
  std::vector<int32> silence_phones_; // derived from opts_.silence_phones_str
};



NnetDiscriminativeUpdater::NnetDiscriminativeUpdater(
    const AmNnet &am_nnet,
    const TransitionModel &tmodel,
    const NnetDiscriminativeUpdateOptions &opts,
    const DiscriminativeNnetExample &eg,
    Nnet *nnet_to_update,
    NnetDiscriminativeStats *stats):
    am_nnet_(am_nnet), tmodel_(tmodel), opts_(opts), eg_(eg),
    nnet_to_update_(nnet_to_update), stats_(stats) {
  if (!SplitStringToIntegers(opts_.silence_phones_str, ":", false,
                             &silence_phones_)) {
    KALDI_ERR << "Bad value for --silence-phones option: "
              << opts_.silence_phones_str;
  }
  const Nnet &nnet = am_nnet_.GetNnet();
  nnet.ComputeChunkInfo(eg_.input_frames.NumRows(), 1, &chunk_info_out_);
}



SubMatrix<BaseFloat> NnetDiscriminativeUpdater::GetInputFeatures() const {
  int32 num_frames_output = eg_.num_ali.size();
  int32 eg_left_context = eg_.left_context,
      eg_right_context = eg_.input_frames.NumRows() -
      num_frames_output - eg_left_context;
  KALDI_ASSERT(eg_right_context >= 0);
  const Nnet &nnet = am_nnet_.GetNnet();
  // Make sure the example has enough acoustic left and right
  // context... normally we'll use examples generated using the same model,
  // which will have the exact context, but we enable a mismatch in context as
  // long as it is more, not less.
  KALDI_ASSERT(eg_left_context >= nnet.LeftContext() &&
               eg_right_context >= nnet.RightContext());
  int32 offset = eg_left_context - nnet.LeftContext(),
      num_output_frames =
      num_frames_output + nnet.LeftContext() + nnet.RightContext();
  SubMatrix<BaseFloat> ans(eg_.input_frames, offset, num_output_frames,
                           0, eg_.input_frames.NumCols());
  return ans;
}

void NnetDiscriminativeUpdater::Propagate() {
  const Nnet &nnet = am_nnet_.GetNnet();
  forward_data_.resize(nnet.NumComponents() + 1);
  
  SubMatrix<BaseFloat> input_feats = GetInputFeatures();
  int32 spk_dim = eg_.spk_info.Dim();
  if (spk_dim == 0) {
    forward_data_[0] = input_feats;
  } else {
    forward_data_[0].Resize(input_feats.NumRows(),
                            input_feats.NumCols() + eg_.spk_info.Dim());
    forward_data_[0].Range(0, input_feats.NumRows(),
                           0, input_feats.NumCols()).CopyFromMat(input_feats);
    forward_data_[0].Range(0, input_feats.NumRows(),
                           input_feats.NumCols(), spk_dim).CopyRowsFromVec(
                               eg_.spk_info);
  }

  for (int32 c = 0; c < nnet.NumComponents(); c++) {
    const Component &component = nnet.GetComponent(c);
    CuMatrix<BaseFloat> &input = forward_data_[c],
        &output = forward_data_[c+1];
    component.Propagate(chunk_info_out_[c] , chunk_info_out_[c+1], input, &output);
    const Component *prev_component = (c == 0 ? NULL :
                                       &(nnet.GetComponent(c-1)));
    bool will_do_backprop = (nnet_to_update_ != NULL),
        keep_last_output = will_do_backprop &&
        ((c>0 && prev_component->BackpropNeedsOutput()) ||
         component.BackpropNeedsInput());
    if (!keep_last_output)
      forward_data_[c].Resize(0, 0); // We won't need this data; save memory.
  }
}



void NnetDiscriminativeUpdater::LatticeComputations() {
  ConvertLattice(eg_.den_lat, &lat_); // convert to Lattice.
  TopSort(&lat_); // Topologically sort (required by forward-backward algorithms)

  if (opts_.criterion == "mmi" && opts_.boost != 0.0) {
    BaseFloat max_silence_error = 0.0;
    LatticeBoost(tmodel_, eg_.num_ali, silence_phones_,
                 opts_.boost, max_silence_error, &lat_);
  }
  
  int32 num_frames = static_cast<int32>(eg_.num_ali.size());

  stats_->tot_t += num_frames;
  stats_->tot_t_weighted += num_frames * eg_.weight;
  
  const VectorBase<BaseFloat> &priors = am_nnet_.Priors();
  const CuMatrix<BaseFloat> &posteriors = forward_data_.back();

  KALDI_ASSERT(posteriors.NumRows() == num_frames);
  int32 num_pdfs = posteriors.NumCols();
  KALDI_ASSERT(num_pdfs == priors.Dim());
  
  // We need to look up the posteriors of some pdf-ids in the matrix
  // "posteriors".  Rather than looking them all up using operator (), which is
  // very slow because each lookup involves a separate CUDA call with
  // communication over PciExpress, we look them up all at once using
  // CuMatrix::Lookup().
  // Note: regardless of the criterion, we evaluate the likelihoods in
  // the numerator alignment.  Even though they may be irrelevant to
  // the optimization, they will affect the value of the objective function.
  
  std::vector<Int32Pair> requested_indexes;
  BaseFloat wiggle_room = 1.3; // value not critical.. it's just 'reserve'
  requested_indexes.reserve(num_frames + wiggle_room * lat_.NumStates());

  if (opts_.criterion == "mmi") { // need numerator probabilities...
    for (int32 t = 0; t < num_frames; t++) {
      int32 tid = eg_.num_ali[t], pdf_id = tmodel_.TransitionIdToPdf(tid);
      KALDI_ASSERT(pdf_id >= 0 && pdf_id < num_pdfs);
      requested_indexes.push_back(MakePair(t, pdf_id));
    }
  }

  std::vector<int32> state_times;
  int32 T = LatticeStateTimes(lat_, &state_times);
  KALDI_ASSERT(T == num_frames);
  
  StateId num_states = lat_.NumStates();
  for (StateId s = 0; s < num_states; s++) {
    StateId t = state_times[s];
    for (fst::ArcIterator<Lattice> aiter(lat_, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) { // input-side has transition-ids, output-side empty
        int32 tid = arc.ilabel, pdf_id = tmodel_.TransitionIdToPdf(tid);
        requested_indexes.push_back(MakePair(t, pdf_id));
      }
    }
  }

  std::vector<BaseFloat> answers;
  CuArray<Int32Pair> cu_requested_indexes(requested_indexes);
  answers.resize(requested_indexes.size());
  posteriors.Lookup(cu_requested_indexes, &(answers[0]));

  int32 num_floored = 0;

  BaseFloat floor_val = 1.0e-20; // floor for posteriors.
  size_t index;

  // Replace "answers" with the vector of scaled log-probs.  If this step takes
  // too much time, we can look at other ways to do it, using the CUDA card.
  for (index = 0; index < answers.size(); index++) {
    BaseFloat post = answers[index];
    if (post < floor_val) {
      post = floor_val;
      num_floored++;
    }
    int32 pdf_id = requested_indexes[index].second;
    BaseFloat pseudo_loglike = Log(post / priors(pdf_id)) * opts_.acoustic_scale;
    KALDI_ASSERT(!KALDI_ISINF(pseudo_loglike) && !KALDI_ISNAN(pseudo_loglike));
    answers[index] = pseudo_loglike;
  }
  if (num_floored > 0) {
    KALDI_WARN << "Floored " << num_floored << " probabilities from nnet.";
  }
  
  index = 0;
  
  if (opts_.criterion == "mmi") {
    double tot_num_like = 0.0;
    for (; index < eg_.num_ali.size(); index++)
      tot_num_like += answers[index];
    stats_->tot_num_objf += eg_.weight * tot_num_like;
  }

  // Now put the (scaled) acoustic log-likelihoods in the lattice.
  for (StateId s = 0; s < num_states; s++) {
    for (fst::MutableArcIterator<Lattice> aiter(&lat_, s);
         !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel != 0) { // input-side has transition-ids, output-side empty
        arc.weight.SetValue2(-answers[index]);
        index++;
        aiter.SetValue(arc);
      }
    }
    LatticeWeight final = lat_.Final(s);
    if (final != LatticeWeight::Zero()) {
      final.SetValue2(0.0); // make sure no acoustic term in final-prob.
      lat_.SetFinal(s, final);
    }
  }
  KALDI_ASSERT(index == answers.size());
  
  // Get the MPE or MMI posteriors.
  Posterior post;
  stats_->tot_den_objf += eg_.weight * GetDiscriminativePosteriors(&post);

  ScalePosterior(eg_.weight, &post);

  double tot_num_post = 0.0, tot_den_post = 0.0;
  std::vector<MatrixElement<BaseFloat> > sv_labels;
  sv_labels.reserve(answers.size());
  for (int32 t = 0; t < post.size(); t++) {
    for (int32 i = 0; i < post[t].size(); i++) {
      int32 pdf_id = post[t][i].first;
      BaseFloat weight = post[t][i].second;
      if (weight > 0.0) { tot_num_post += weight; }
      else { tot_den_post -= weight; }
      MatrixElement<BaseFloat> elem = {t, pdf_id, weight};
      sv_labels.push_back(elem);
    }
  }
  stats_->tot_num_count += tot_num_post;
  int32 num_components = am_nnet_.GetNnet().NumComponents();
  const CuMatrix<BaseFloat> &output(forward_data_[num_components]);
  backward_data_.Resize(output.NumRows(), output.NumCols()); // zeroes it.
  
  { // We don't actually need tot_objf and tot_weight; we have already
    // computed the objective function.
    BaseFloat tot_objf, tot_weight;
    backward_data_.CompObjfAndDeriv(sv_labels, output, &tot_objf, &tot_weight);
    // Now backward_data_ will contan the derivative at the output.
    // Our work here is done..
  }
}


double NnetDiscriminativeUpdater::GetDiscriminativePosteriors(Posterior *post) {
  if (opts_.criterion == "mpfe" || opts_.criterion == "smbr") {
    Posterior tid_post;
    double ans;
    ans = LatticeForwardBackwardMpeVariants(tmodel_, silence_phones_, lat_,
                                            eg_.num_ali, opts_.criterion,
                                            opts_.one_silence_class,
                                            &tid_post);
    ConvertPosteriorToPdfs(tmodel_, tid_post, post);
    return ans; // returns the objective function.
  } else {
    KALDI_ASSERT(opts_.criterion == "mmi");
    bool convert_to_pdfs = true, cancel = true;
    // we'll return the denominator-lattice forward backward likelihood,
    // which is one term in the objective function.
    return LatticeForwardBackwardMmi(tmodel_, lat_, eg_.num_ali,
                                     opts_.drop_frames, convert_to_pdfs,
                                     cancel, post);
  }
}



void NnetDiscriminativeUpdater::Backprop() {
  const Nnet &nnet = am_nnet_.GetNnet();
  for (int32 c = nnet.NumComponents() - 1; c >= 0; c--) {
    const Component &component = nnet.GetComponent(c);
    Component *component_to_update = &(nnet_to_update_->GetComponent(c));
    const CuMatrix<BaseFloat>  &input = forward_data_[c],
                            &output = forward_data_[c+1],
                      &output_deriv = backward_data_;
    CuMatrix<BaseFloat> input_deriv;
    component.Backprop(chunk_info_out_[c], chunk_info_out_[c+1], input, output, output_deriv,
                       component_to_update, &input_deriv);
    backward_data_.Swap(&input_deriv); // backward_data_ = input_deriv.
  }
}


void NnetDiscriminativeUpdate(const AmNnet &am_nnet,
                              const TransitionModel &tmodel,
                              const NnetDiscriminativeUpdateOptions &opts,
                              const DiscriminativeNnetExample &eg,
                              Nnet *nnet_to_update,
                              NnetDiscriminativeStats *stats) {
  NnetDiscriminativeUpdater updater(am_nnet, tmodel, opts, eg,
                                    nnet_to_update, stats);
  updater.Update();
}

void NnetDiscriminativeStats::Add(const NnetDiscriminativeStats &other) {
  tot_t += other.tot_t;
  tot_t_weighted += other.tot_t_weighted;
  tot_num_count += other.tot_num_count;
  tot_num_objf += other.tot_num_objf;
  tot_den_objf += other.tot_den_objf;
}

void NnetDiscriminativeStats::Print(std::string criterion) {
  KALDI_ASSERT(criterion == "mmi" || criterion == "smbr" ||
               criterion == "mpfe");

  double avg_post_per_frame = tot_num_count / tot_t_weighted;
  KALDI_LOG << "Number of frames is " << tot_t
            << " (weighted: " << tot_t_weighted
            << "), average (num or den) posterior per frame is "
            << avg_post_per_frame;
  
  if (criterion == "mmi") {
    double num_objf = tot_num_objf / tot_t_weighted,
        den_objf = tot_den_objf / tot_t_weighted,
        objf = num_objf - den_objf;
    KALDI_LOG << "MMI objective function is " << num_objf << " - "
              << den_objf << " = " << objf << " per frame, over "
              << tot_t_weighted << " frames.";
  } else if (criterion == "mpfe") {
    double objf = tot_den_objf / tot_t_weighted; // this contains the actual
                                                 // summed objf
    KALDI_LOG << "MPFE objective function is " << objf
              << " per frame, over " << tot_t_weighted << " frames.";
  } else {
    double objf = tot_den_objf / tot_t_weighted; // this contains the actual
                                                 // summed objf
    KALDI_LOG << "SMBR objective function is " << objf
              << " per frame, over " << tot_t_weighted << " frames.";
  }
}


} // namespace nnet2
} // namespace kaldi
