// nnet3/discriminative-training.cc

// Copyright      2012-2015    Johns Hopkins University (author: Daniel Povey)
// Copyright      2014-2015    Vimal Manohar

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

#include "nnet3/discriminative-training.h"
#include "lat/lattice-functions.h"
#include "cudamatrix/cu-matrix.h"

namespace kaldi {
namespace discriminative {

DiscriminativeObjectiveInfo::DiscriminativeObjectiveInfo() {
  std::memset(this, 0, sizeof(*this));
}

DiscriminativeObjectiveInfo::DiscriminativeObjectiveInfo(int32 num_pdfs) :
  accumulate_gradients(false),
  accumulate_output(false),
  num_pdfs(num_pdfs) {
  gradients.Resize(num_pdfs);
  output.Resize(num_pdfs);
  Reset();
}

// Constructor from config structure
DiscriminativeObjectiveInfo::DiscriminativeObjectiveInfo(
    const DiscriminativeOptions &opts) :
  accumulate_gradients(opts.accumulate_gradients),
  accumulate_output(opts.accumulate_output),
  num_pdfs(opts.num_pdfs) {
  gradients.Resize(opts.num_pdfs);
  output.Resize(opts.num_pdfs);
  Reset();
}

// Reset statistics
void DiscriminativeObjectiveInfo::Reset() {
  gradients.SetZero();
  output.SetZero();

  tot_t = 0.0;
  tot_t_weighted = 0.0;
  tot_objf = 0.0;
  tot_num_count = 0.0;
  tot_den_count = 0.0;
  tot_num_objf = 0.0;
  tot_l2_term = 0.0;
}

void DiscriminativeObjectiveInfo::Configure(const DiscriminativeOptions &opts) {
  accumulate_gradients = opts.accumulate_gradients;
  accumulate_output = opts.accumulate_output;
  num_pdfs = opts.num_pdfs;
  gradients.Resize(opts.num_pdfs);
  output.Resize(opts.num_pdfs);
}

// This class is responsible for the forward-backward of the
// 'supervision' lattices and computation of the objective function
// and gradients.
//
// note: the supervision.weight is ignored by this class, you have to apply
// it externally.
class DiscriminativeComputation {
  typedef Lattice::Arc Arc;
  typedef Arc::StateId StateId;

 public:
  // Initialize the objcect.  Note: we expect the 'nnet_output' to have the
  // same number of rows as supervision.num_frames * supervision.num_sequences,
  // and the same number of columns as tmodel.NumPdfs(); but the
  // ordering of the rows of 'nnet_output' is not the same as the ordering of
  // frames in paths in the 'supervision' object (which has all frames of the
  // 1st sequence first, then the 2nd sequence, and so on).  Instead, the
  // frames in 'nnet_output' are ordered as: first the first frame of each
  // sequence, then the second frame of each sequence, and so on.
  // This is done to be similar to the setup in 'chain' training
  // even though this does not offer any computational advantages here
  // as in the 'chain' case.
  DiscriminativeComputation(const DiscriminativeOptions &opts,
      const TransitionModel &tmodel,
      const CuVectorBase<BaseFloat> &log_priors,
      const DiscriminativeSupervision &supervision,
      const CuMatrixBase<BaseFloat> &nnet_output,
      DiscriminativeObjectiveInfo *stats,
      CuMatrixBase<BaseFloat> *nnet_output_deriv,
      CuMatrixBase<BaseFloat> *xent_output_deriv);

  // Does the forward-backward computation and add the derivative of the
  // w.r.t. the nnet output (log-prob) times supervision_.weight times
  // deriv_weight to 'nnet_output_deriv'.
  void Compute();

 private:
  const DiscriminativeOptions &opts_;
  const TransitionModel &tmodel_;

  // Vector of log-priors of pdfs.
  // This can be a size zero vector e.g. for 'chain' model
  const CuVectorBase<BaseFloat> &log_priors_;

  const DiscriminativeSupervision &supervision_;

  // The neural net output.
  const CuMatrixBase<BaseFloat> &nnet_output_;

  // Training stats including accumulated objective function, gradient
  // and total weight. Optionally the nnet_output and gradients per pdf can be
  // accumulated for debugging purposes.
  DiscriminativeObjectiveInfo *stats_;

  // If non-NULL, derivative w.r.t. to nnet_output is written here.
  CuMatrixBase<BaseFloat> *nnet_output_deriv_;

  // If non-NULL, then the xent objective derivative
  // (which equals a posterior from the numerator forward-backward, scaled by
  // the supervision weight) is written to here.
  // This will be used in the cross-entropy regularization code.
  CuMatrixBase<BaseFloat> *xent_output_deriv_;

  // Denominator lattice.
  Lattice den_lat_;

  // List of silence phones. Useful to treat silence phones
  // differently in computing SMBR / MPFE objectives.
  std::vector<int32> silence_phones_;

  // The function that actually computes the objective and gradients
  double ComputeObjfAndDeriv(Posterior *post, Posterior *xent_post);

  // This function looks up the nnet output the pdf-ids in the
  // denominator lattice and the alignment in the case of "mmi" objective
  // using the CuMatrix::Lookup() and stores them in "answers"
  void LookupNnetOutput(std::vector<Int32Pair> *requested_indexes,
                        std::vector<BaseFloat> *answers) const ;

  // Converts the answers looked up by LookupNnetOutput function into
  // log-likelihoods scaled by acoustic scale.
  void ConvertAnswersToLogLike(
      const std::vector<Int32Pair>& requested_indexes,
      std::vector<BaseFloat> *answers) const;

  // Does acoustic rescoring of lattice to put the negative (scaled) acoustic
  // log-likelihoods in the arcs of the lattice. Returns the number of
  // indexes of log-likelihoods read from the "answers" vector.
  static size_t LatticeAcousticRescore(const std::vector<BaseFloat> &answers,
                                size_t index,
                                Lattice *lat);

  // Process the derivative stored as posteriors into CuMatrix.
  // Optionally accumulate numerator and denominator posteriors.
  void ProcessPosteriors(const Posterior &post,
                         CuMatrixBase<BaseFloat> *output_deriv_temp,
                         double *tot_num_post = NULL,
                         double *tot_den_post = NULL) const;

  static inline Int32Pair MakePair(int32 first, int32 second) {
    Int32Pair ans;
    ans.first = first;
    ans.second = second;
    return ans;
  }
};

DiscriminativeComputation::DiscriminativeComputation(
                            const DiscriminativeOptions &opts,
                            const TransitionModel &tmodel,
                            const CuVectorBase<BaseFloat> &log_priors,
                            const DiscriminativeSupervision &supervision,
                            const CuMatrixBase<BaseFloat> &nnet_output,
                            DiscriminativeObjectiveInfo *stats,
                            CuMatrixBase<BaseFloat> *nnet_output_deriv,
                            CuMatrixBase<BaseFloat> *xent_output_deriv)
  : opts_(opts), tmodel_(tmodel), log_priors_(log_priors),
  supervision_(supervision), nnet_output_(nnet_output),
  stats_(stats),
  nnet_output_deriv_(nnet_output_deriv),
  xent_output_deriv_(xent_output_deriv) {

  den_lat_ = supervision.den_lat;
  TopSort(&den_lat_);

  if (!SplitStringToIntegers(opts_.silence_phones_str, ":", false,
                             &silence_phones_)) {
    KALDI_ERR << "Bad value for --silence-phones option: "
              << opts_.silence_phones_str;
  }
}

void DiscriminativeComputation::LookupNnetOutput(
    std::vector<Int32Pair> *requested_indexes,
    std::vector<BaseFloat> *answers) const {
  BaseFloat wiggle_room = 1.3; // value not critical.. it's just 'reserve'

  int32 num_frames = supervision_.frames_per_sequence * supervision_.num_sequences;
  int32 num_pdfs = tmodel_.NumPdfs();

  int32 num_reserve = wiggle_room * den_lat_.NumStates();

  if (opts_.criterion == "mmi") {
    // For looking up the posteriors corresponding to the pdfs in the alignment
    num_reserve += num_frames;
  }

  requested_indexes->reserve(num_reserve);

  // Denominator probabilities to look up from denominator lattice
  std::vector<int32> state_times;
  int32 T = LatticeStateTimes(den_lat_, &state_times);
  KALDI_ASSERT(T == num_frames);

  StateId num_states = den_lat_.NumStates();
  for (StateId s = 0; s < num_states; s++) {
    int32 t = state_times[s];
    int32 seq = t / supervision_.frames_per_sequence,
          idx = t % supervision_.frames_per_sequence;

    for (fst::ArcIterator<Lattice> aiter(den_lat_, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) { // input-side has transition-ids, output-side empty
        int32 tid = arc.ilabel, pdf_id = tmodel_.TransitionIdToPdf(tid);
        // The ordering of the indexes is similar to that in chain models
        requested_indexes->push_back(MakePair(idx * supervision_.num_sequences + seq, pdf_id));
      }
    }
  }

  if (opts_.criterion == "mmi") {
    // Numerator probabilities to look up from alignment
    for (int32 t = 0; t < num_frames; t++) {
      int32 seq = t / supervision_.frames_per_sequence,
            idx = t % supervision_.frames_per_sequence;
      int32 tid = supervision_.num_ali[t],
                  pdf_id = tmodel_.TransitionIdToPdf(tid);
      KALDI_ASSERT(pdf_id >= 0 && pdf_id < num_pdfs);
      requested_indexes->push_back(MakePair(idx * supervision_.num_sequences + seq, pdf_id));
    }
  }

  CuArray<Int32Pair> cu_requested_indexes(*requested_indexes);
  answers->resize(requested_indexes->size());
  nnet_output_.Lookup(cu_requested_indexes, &((*answers)[0]));
  // requested_indexes now contain (t, j) pair and answers contains the
  // neural network output, which is log p(j|x(t)) for CE models
}

void DiscriminativeComputation::ConvertAnswersToLogLike(
    const std::vector<Int32Pair>& requested_indexes,
    std::vector<BaseFloat> *answers) const {
  int32 num_floored = 0;

  BaseFloat floor_val = -20 * kaldi::Log(10.0); // floor for posteriors.
  size_t index;

  Vector<BaseFloat> log_priors(log_priors_);

  // Replace "answers" with the vector of scaled log-probs.  If this step takes
  // too much time, we can look at other ways to do it, using the CUDA card.
  for (index = 0; index < answers->size(); index++) {
    BaseFloat log_post = (*answers)[index];
    if (log_post < floor_val) {
      // TODO: this might not be required for 'chain' models
      log_post = floor_val;
      num_floored++;
    }

    if (log_priors_.Dim() > 0) {
      int32 pdf_id = requested_indexes[index].second;
      KALDI_ASSERT(log_post <= 0 && log_priors(pdf_id) <= 0);
      BaseFloat pseudo_loglike = (log_post - log_priors(pdf_id))
                                  * opts_.acoustic_scale;
      KALDI_ASSERT(!KALDI_ISINF(pseudo_loglike) && !KALDI_ISNAN(pseudo_loglike));
      (*answers)[index] = pseudo_loglike;
    } else {
      (*answers)[index] = log_post * opts_.acoustic_scale;
    }
  }

  if (num_floored > 0) {
    KALDI_WARN << "Floored " << num_floored << " probabilities from nnet.";
  }
}

size_t DiscriminativeComputation::LatticeAcousticRescore(
    const std::vector<BaseFloat> &answers,
    size_t index, Lattice *lat) {
  int32 num_states = lat->NumStates();

  for (StateId s = 0; s < num_states; s++) {
    for (fst::MutableArcIterator<Lattice> aiter(lat, s);
         !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel != 0) { // input-side has transition-ids, output-side empty
        arc.weight.SetValue2(-answers[index]);
        index++;
        aiter.SetValue(arc);
      }
    }
    LatticeWeight final = lat->Final(s);
    if (final != LatticeWeight::Zero()) {
      final.SetValue2(0.0); // make sure no acoustic term in final-prob.
      lat->SetFinal(s, final);
    }
  }

  // Number of indexes of log-likes used to rescore lattice
  return index;
}

void DiscriminativeComputation::ProcessPosteriors(
                                const Posterior &post,
                                CuMatrixBase<BaseFloat> *output_deriv_temp,
                                double *tot_num_post,
                                double *tot_den_post) const {
  std::vector<Int32Pair> deriv_indexes;
  std::vector<BaseFloat> deriv_data;
  for (size_t t = 0; t < post.size(); t++) {
    for (size_t j = 0; j < post[t].size(); j++) {
      int32 seq = t / supervision_.frames_per_sequence,
            idx = t % supervision_.frames_per_sequence;
      int32 pdf_id = post[t][j].first;

      // Same ordering as for 'chain' models
      deriv_indexes.push_back(MakePair(idx * supervision_.num_sequences + seq, pdf_id));

      BaseFloat weight = post[t][j].second;
      if (tot_num_post && weight > 0.0) *tot_num_post += weight;
      if (tot_den_post && weight < 0.0) *tot_den_post -= weight;
      deriv_data.push_back(weight);
    }
  }
  CuArray<Int32Pair> cu_deriv_indexes(deriv_indexes);
  output_deriv_temp->AddElements(supervision_.weight, cu_deriv_indexes,
                                 deriv_data.data());
}

void DiscriminativeComputation::Compute() {
  if (opts_.criterion == "mmi" && opts_.boost != 0.0) {
    BaseFloat max_silence_error = 0.0;
    LatticeBoost(tmodel_, supervision_.num_ali, silence_phones_,
                 opts_.boost, max_silence_error, &den_lat_);
  }

  int32 num_frames = supervision_.frames_per_sequence * supervision_.num_sequences;

  int32 num_pdfs = nnet_output_.NumCols();
  KALDI_ASSERT(log_priors_.Dim() == 0 || num_pdfs == log_priors_.Dim());

  // We need to look up the nnet output for some pdf-ids.
  // Rather than looking them all up using operator (), which is
  // very slow because each lookup involves a separate CUDA call with
  // communication over PciExpress, we look them up all at once using
  // CuMatrix::Lookup().
  std::vector<BaseFloat> answers;
  std::vector<Int32Pair> requested_indexes;

  LookupNnetOutput(&requested_indexes, &answers);

  ConvertAnswersToLogLike(requested_indexes, &answers);

  size_t index = 0;

  // Now put the negative (scaled) acoustic log-likelihoods in the lattice.
  index = LatticeAcousticRescore(answers, index, &den_lat_);
  // index is now the number of indexes of log-likes used to rescore lattice.
  // This is required to further lookup answers for computing "mmi"
  // numerator score.

  // Get statistics for this minibatch
  DiscriminativeObjectiveInfo this_stats;
  if (stats_) {
    this_stats = *stats_;
    this_stats.Reset();
  }

  // Look up numerator probabilities corresponding to alignment
  if (opts_.criterion == "mmi") {
    double tot_num_like = 0.0;
    KALDI_ASSERT(index + supervision_.num_ali.size() == answers.size());
    for (size_t this_index = 0; this_index < supervision_.num_ali.size(); this_index++) {
      tot_num_like += answers[index + this_index];
    }
    this_stats.tot_num_objf += supervision_.weight * tot_num_like;
    index += supervision_.num_ali.size();
  }

  KALDI_ASSERT(index == answers.size());

  if (nnet_output_deriv_) {
    nnet_output_deriv_->SetZero();
    KALDI_ASSERT(nnet_output_deriv_->NumRows() == nnet_output_.NumRows() &&
        nnet_output_deriv_->NumCols() == nnet_output_.NumCols());
  }

  if (xent_output_deriv_) {
    xent_output_deriv_->SetZero();
    KALDI_ASSERT(xent_output_deriv_->NumRows() == nnet_output_.NumRows() &&
        xent_output_deriv_->NumCols() == nnet_output_.NumCols());
  }

  Posterior post;
  Posterior xent_post;
  double objf = ComputeObjfAndDeriv(&post,
                (xent_output_deriv_ ? &xent_post : NULL));

  this_stats.tot_objf += supervision_.weight * objf;

  KALDI_ASSERT(nnet_output_.NumRows() == post.size());

  CuMatrix<BaseFloat> output_deriv;

  CuMatrixBase<BaseFloat> *output_deriv_temp;

  if (nnet_output_deriv_)
    output_deriv_temp = nnet_output_deriv_;
  else {
    // This is for accumulating the statistics
    output_deriv.Resize(nnet_output_.NumRows(), nnet_output_.NumCols());
    output_deriv_temp = &output_deriv;
  }

  double tot_num_post = 0.0, tot_den_post = 0.0;
  {
    ProcessPosteriors(post, output_deriv_temp,
                             &tot_num_post, &tot_den_post);
  }

  if (xent_output_deriv_) {
    ProcessPosteriors(xent_post, xent_output_deriv_, NULL, NULL);
  }

  this_stats.tot_den_count += tot_den_post;
  this_stats.tot_num_count += tot_num_post;

  if (this_stats.AccumulateGradients())
    (this_stats.gradients).AddRowSumMat(1.0, CuMatrix<double>(*output_deriv_temp));

  if (this_stats.AccumulateOutput()) {
    CuMatrix<double> temp(nnet_output_);
    temp.ApplyExp();
    (this_stats.output).AddRowSumMat(1.0, temp);
  }

  this_stats.tot_t = num_frames;
  this_stats.tot_t_weighted = num_frames * supervision_.weight;

  if (!(this_stats.TotalObjf(opts_.criterion) ==
        this_stats.TotalObjf(opts_.criterion))) {
    // inf or NaN detected
    if (nnet_output_deriv_)
      nnet_output_deriv_->SetZero();
    BaseFloat default_objf = -10;
    KALDI_WARN << "Objective function is "
               << this_stats.TotalObjf(opts_.criterion)
               << ", setting to " << default_objf << " per frame.";
    this_stats.tot_objf = default_objf * this_stats.tot_t_weighted;
  }

  if (GetVerboseLevel() >= 2) {
    if (GetVerboseLevel() >= 3) {
      this_stats.PrintAll(opts_.criterion);
    } else
      this_stats.Print(opts_.criterion);
  }

  // This code helps us see how big the derivatives are, on average,
  // for different frames of the sequences.  As expected, they are
  // smaller towards the edges of the sequences (due to the penalization
  // of 'incorrect' pdf-ids.
  if (nnet_output_deriv_ && GetVerboseLevel() >= 1) {
    int32 tot_frames = nnet_output_deriv_->NumRows(),
 frames_per_sequence = supervision_.frames_per_sequence,
       num_sequences = supervision_.num_sequences;
    CuVector<BaseFloat> row_products(tot_frames);
    row_products.AddDiagMat2(1.0, *nnet_output_deriv_, kNoTrans, 0.0);
    Vector<BaseFloat> row_products_cpu(row_products);
    Vector<BaseFloat> row_products_per_frame(frames_per_sequence);
    for (int32 i = 0; i < tot_frames; i++)
      row_products_per_frame(i / num_sequences) += row_products_cpu(i);
    KALDI_LOG << "Derivs per frame are " << row_products_per_frame;
  }

  if (opts_.l2_regularize != 0.0) {
    // compute the l2 penalty term and its derivative
    BaseFloat scale = supervision_.weight * opts_.l2_regularize;
    this_stats.tot_l2_term += -0.5 * scale * TraceMatMat(nnet_output_, nnet_output_, kTrans);
    if (nnet_output_deriv_)
      nnet_output_deriv_->AddMat(-1.0 * scale, nnet_output_);
  }

  if (stats_)
    stats_->Add(this_stats);

}

double DiscriminativeComputation::ComputeObjfAndDeriv(Posterior *post,
                                                      Posterior *xent_post) {

  if (xent_post) {
    Posterior tid_post;
    // Compute posterior from the numerator alignment
    AlignmentToPosterior(supervision_.num_ali, &tid_post);
    ConvertPosteriorToPdfs(tmodel_, tid_post, xent_post);
  }

  if (opts_.criterion == "mpfe" || opts_.criterion == "smbr") {
    Posterior tid_post;
    double ans = LatticeForwardBackwardMpeVariants(tmodel_, silence_phones_,
        den_lat_,
        supervision_.num_ali, opts_.criterion,
        opts_.one_silence_class,
        &tid_post);
    ConvertPosteriorToPdfs(tmodel_, tid_post, post);
    return ans;
  } else if (opts_.criterion == "mmi") {
    bool convert_to_pdfs = true, cancel = true;
    // we'll return the denominator-lattice forward backward likelihood,
    // which is one term in the objective function.
    return (LatticeForwardBackwardMmi(tmodel_, den_lat_, supervision_.num_ali,
                                      opts_.drop_frames, convert_to_pdfs,
                                      cancel, post));
  } else {
    KALDI_ERR << "Unknown criterion " << opts_.criterion;
  }

  return 0;
}


void ComputeDiscriminativeObjfAndDeriv(const DiscriminativeOptions &opts,
                                       const TransitionModel &tmodel,
                                       const CuVectorBase<BaseFloat> &log_priors,
                                       const DiscriminativeSupervision &supervision,
                                       const CuMatrixBase<BaseFloat> &nnet_output,
                                       DiscriminativeObjectiveInfo *stats,
                                       CuMatrixBase<BaseFloat> *nnet_output_deriv,
                                       CuMatrixBase<BaseFloat> *xent_output_deriv) {
  DiscriminativeComputation computation(opts, tmodel, log_priors, supervision,
                                        nnet_output, stats,
                                        nnet_output_deriv, xent_output_deriv);
  computation.Compute();
}

void DiscriminativeObjectiveInfo::Add(const DiscriminativeObjectiveInfo &other) {
  tot_t += other.tot_t;
  tot_t_weighted += other.tot_t_weighted;
  tot_objf += other.tot_objf;             // Actually tot_den_objf for mmi
  tot_num_count += other.tot_num_count;
  tot_den_count += other.tot_den_count;
  tot_num_objf += other.tot_num_objf;     // Only for mmi
  tot_l2_term += other.tot_l2_term;

  if (AccumulateGradients()) {
    gradients.AddVec(1.0, other.gradients);
  }
  if (AccumulateOutput()) {
    output.AddVec(1.0, other.output);
  }
}

void DiscriminativeObjectiveInfo::Print(const std::string &criterion,
                                        bool print_avg_gradients,
                                        bool print_avg_output) const {
  if (criterion == "mmi") {
    double num_objf = tot_num_objf / tot_t_weighted,
           den_objf = tot_objf / tot_t_weighted;
    double objf = num_objf - den_objf;

    double avg_post_per_frame = tot_num_count / tot_t_weighted;

    KALDI_LOG << "Number of frames is " << tot_t
              << " (weighted: " << tot_t_weighted
              << "), average (num or den) posterior per frame is "
              << avg_post_per_frame;
    KALDI_LOG << "MMI objective function is " << num_objf << " - "
              << den_objf << " = " << objf << " per frame, over "
              << tot_t_weighted << " frames.";
  } else if (criterion == "mpfe") {
    double avg_gradients = (tot_num_count + tot_den_count) / tot_t_weighted;
    double objf = tot_objf / tot_t_weighted;
    KALDI_LOG << "Average modulus of MPFE gradients is " << avg_gradients
              << " per frame, over "
              << tot_t_weighted << " frames";
    KALDI_LOG << "MPFE objective function is " << objf
              << " per frame, over " << tot_t_weighted << " frames.";
  } else if (criterion == "smbr") {
    double avg_gradients = (tot_num_count + tot_den_count) / tot_t_weighted;
    double objf = tot_objf / tot_t_weighted;
    KALDI_LOG << "Average modulus of SMBR gradients is " << avg_gradients
              << " per frame, over "
              << tot_t_weighted << " frames";
    KALDI_LOG << "SMBR objective function is " << objf
              << " per frame, over " << tot_t_weighted << " frames.";
  }

  if (AccumulateGradients()) {
    Vector<double> temp(gradients);
    temp.Scale(1.0/tot_t_weighted);
    if (print_avg_gradients) {
      KALDI_LOG << "Vector of average gradients wrt output activations is: \n" << temp;
    } else {
      KALDI_VLOG(4) << "Vector of average gradients wrt output activations is: \n" << temp;
    }
  }
  if (AccumulateOutput()) {
    Vector<double> temp(output);
    temp.Scale(1.0/tot_t_weighted);
    if (print_avg_output) {
      KALDI_LOG << "Average DNN output is: \n" << temp;
    } else {
      KALDI_VLOG(4) << "Average DNN output is: \n" << temp;
    }
  }
}

void DiscriminativeObjectiveInfo::PrintAvgGradientForPdf(int32 pdf_id) const {
  if (pdf_id < gradients.Dim() && pdf_id >= 0) {
    KALDI_LOG << "Average gradient wrt output activations of pdf " << pdf_id
              << " is " << gradients(pdf_id) / tot_t_weighted
              << " per frame, over "
              << tot_t_weighted << " frames";
  }
}



}  // namespace discriminative
}  // namespace kaldi

