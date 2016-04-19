// nnet3/discriminative-training.h

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


#ifndef KALDI_NNET3_DISCRIMINATIVE_TRAINING_H_
#define KALDI_NNET3_DISCRIMINATIVE_TRAINING_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "hmm/transition-model.h"
#include "nnet3/discriminative-supervision.h"
#include "lat/lattice-functions.h"
#include "cudamatrix/cu-matrix-lib.h"

namespace kaldi {
namespace discriminative {

/* Options for discriminative training
 *
 * Legend:
 * mmi - Maximum Mutual Information
 * mpfe - Minimum Phone Frame Error
 * smbr - State Minimum Bayes Risk
 *
 */
struct DiscriminativeOptions {
  std::string criterion; // one of {"mmi", "mpfe", "smbr"}
                         // If the criterion does not match the supervision
                         // object, the derivatives may not be very accurate
  BaseFloat acoustic_scale; // e.g. 0.1
  bool drop_frames; // for MMI, true if we ignore frames where alignment
                    // pdf-id is not in the lattice.
  bool one_silence_class;  // Affects MPFE and SMBR objectives 
  BaseFloat boost; // for MMI, boosting factor (would be Boosted MMI)... e.g. 0.1.
  
  std::string silence_phones_str; // colon-separated list of integer ids of silence phones,
                                  // for MPFE and SMBR objectives

  // Cross-entropy regularization constant.  (e.g. try 0.1).  If nonzero,
  // the network is expected to have an output named 'output-xent', which
  // should have a softmax as its final nonlinearity.
  BaseFloat xent_regularize;

  // l2 regularization constant on the 'chain' output; the actual term added to
  // the objf will be -0.5 times this constant times the squared l2 norm.
  // (squared so it's additive across the dimensions).  e.g. try 0.0005.
  BaseFloat l2_regularize;
  
  // Options for debugging discriminative training
  
  // Accumulates gradients wrt nnet outputs
  bool accumulate_gradients;  
  
  // Accumulates nnet output
  bool accumulate_output;     
  
  // Applicable for debugging discriminative training when accumulate_gradients
  // or accumulate_output is true 
  int32 num_pdfs;             

  DiscriminativeOptions(): criterion("smbr"), 
                           acoustic_scale(0.1),
                           drop_frames(false),
                           one_silence_class(false),
                           boost(0.0), 
                           xent_regularize(0.0), 
                           l2_regularize(0.0),
                           accumulate_gradients(false), 
                           accumulate_output(false),
                           num_pdfs(0) { }

  void Register(OptionsItf *opts) {
    opts->Register("criterion", &criterion, "Criterion, 'mmi'|'mpfe'|'smbr', "
                   "determines the objective function to use.  Should match "
                   "option used when we created the examples.");
    opts->Register("acoustic-scale", &acoustic_scale, "Weighting factor to "
                   "apply to acoustic likelihoods.");
    opts->Register("drop-frames", &drop_frames, "For MMI, if true we drop frames "
                   "with no overlap of num and den pdf-ids");
    opts->Register("boost", &boost, "Boosting factor for boosted MMI (e.g. 0.1)");
    opts->Register("one-silence-class", &one_silence_class, "If true, newer "
                   "behavior which will tend to reduce insertions "
                   "when using MPFE or SMBR objective");
    opts->Register("silence-phones", &silence_phones_str,
                   "For MPFE or SMBR objectives, colon-separated list of "
                   "integer ids of silence phones, e.g. 1:2:3");
    opts->Register("l2-regularize", &l2_regularize, "l2 regularization "
                   "constant for 'chain' output "
                   "of the neural net.");
    opts->Register("xent-regularize", &xent_regularize, "Cross-entropy "
                   "regularization constant for sequence training.  If "
                   "nonzero, the network is expected to have an output "
                   "named 'output-xent', which should have a softmax as "
                   "its final nonlinearity.");
    opts->Register("accumulate-gradients", &accumulate_gradients,
                   "Accumulate gradients wrt nnet output "
                   "for debugging discriminative training");
    opts->Register("accumulate-output", &accumulate_output,
                   "Accumulate nnet output "
                   "for debugging discriminative training");
    opts->Register("num-pdfs", &num_pdfs,
                   "Number of pdfs; "
                   "applicable when accumulate-output or accumulate-gradients "
                   "is true for discriminative training");
  }
};

struct DiscriminativeObjectiveInfo {
  double tot_t;          // total number of frames
  double tot_t_weighted; // total number of frames times weight.
  double tot_objf;      // for 'mmi', the (weighted) denominator likelihood; for
                        // everything else, the objective function.
  double tot_num_count; // total count of numerator posterior 
  double tot_den_count; // total count of denominator posterior 
  double tot_num_objf;  // for 'mmi', the (weighted) numerator likelihood; for
                        // everything else 0
  
  double tot_l2_term;   // l2 regularization objective
  // l2 regularization constant on the 'chain' output; the actual term added to
  // the objf will be -0.5 times this constant times the squared l2 norm.
  // (squared so it's additive across the dimensions).  e.g. try 0.0005.

  // Options for debugging discriminative training
  
  // Accumulates gradients wrt nnet outputs
  bool accumulate_gradients;  
  
  // Accumulates nnet output
  bool accumulate_output;     
  
  // Applicable for debugging discriminative training when accumulate_gradients
  // or accumulate_output is true 
  int32 num_pdfs;             

  // Used to accumulates gradients wrt nnet outputs
  // when accumulate_gradients is true
  CuVector<double> gradients;
  // Used to accumulates output when accumulate_output is true
  CuVector<double> output;

  // Print statistics for the criterion
  void Print(const std::string &criterion, 
             bool print_avg_gradients = false, 
             bool print_avg_output = false) const;

  // Print all accumulated statistics for debugging
  void PrintAll(const std::string &criterion) const {
    Print(criterion, true, true);
  }

  // Print the gradient wrt nnet output accumulated for a pdf
  void PrintAvgGradientForPdf(int32 pdf_id) const;

  // Add stats from another object
  void Add(const DiscriminativeObjectiveInfo &other);

  // Returns the objective function value for the criterion
  inline double TotalObjf(const std::string &criterion) const {
    if (criterion == "mmi") return (tot_num_objf - tot_objf);
    return tot_objf;
  }

  // Returns true if accumulate_gradients is true 
  // and the gradients vector has been resized to store the 
  // accumulated gradients
  inline bool AccumulateGradients() const {
    return accumulate_gradients && gradients.Dim() > 0;
  }

  // Returns true if accumulate_output is true 
  // and the output vector has been resized to store the 
  // accumulated nnet output 
  inline bool AccumulateOutput() const {
    return accumulate_output && output.Dim() > 0;
  }

  // Empty constructor
  DiscriminativeObjectiveInfo();

  // Constructor preparing to gradients or output to be accumulated
  DiscriminativeObjectiveInfo(int32 num_pdfs);

  // Constructor from config options
  DiscriminativeObjectiveInfo(const DiscriminativeOptions &opts);
  
  // Reset statistics
  void Reset();
  
  void Configure(const DiscriminativeOptions &opts);
};

/**
   This function does forward-backward on the numerator and denominator 
   lattices and computes derivates wrt to the output for the specified 
   objective function.

   @param [in] opts        Struct containing options
   @param [in] tmodel       Transition model
   @param [in] log_priors   Vector of log-priors for pdfs
   @param [in] supervision  The supervision object, containing the numerator
                            and denominator paths. The denominator is 
                            always a lattice. The numerator is an alignment.
   @param [in] nnet_output  The output of the neural net; dimension must equal
                          ((supervision.num_sequences * supervision.frames_per_sequence) by
                            tmodel.NumPdfs()).

   @param [out] stats       Statistics accumulated during training such as 
                            the objective function and the total weight.
   @param [out] xent_output_deriv  If non-NULL, then the xent objective derivative
                           (which equals a posterior from the numerator forward-backward,
                           scaled by the supervision weight) is written to here.  This will
                           be used in the cross-entropy regularization code.  
*/
void ComputeDiscriminativeObjfAndDeriv(
    const DiscriminativeOptions &opts,
    const TransitionModel &tmodel,
    const CuVectorBase<BaseFloat> &log_priors,
    const DiscriminativeSupervision &supervision,
    const CuMatrixBase<BaseFloat> &nnet_output,
    DiscriminativeObjectiveInfo *stats,
    CuMatrixBase<BaseFloat> *nnet_output_deriv,
    CuMatrixBase<BaseFloat> *xent_output_deriv);

}  // namespace discriminative
}  // namespace kaldi

#endif  // KALDI_NNET3_DISCRIMINATIVE_TRAINING_H_


