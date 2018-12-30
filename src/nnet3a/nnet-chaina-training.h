// nnet3a/nnet-chaina-training.h

// Copyright    2015-2018  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_CHAINA_TRAINING_H_
#define KALDI_NNET3_NNET_CHAINA_TRAINING_H_

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-chain-example.h"
#include "nnet3/nnet-training.h"
#include "nnet3/am-nnet-simple.h"
#include "chain/chain-training.h"
#include "chain/chain-den-graph.h"
#include "adapt/differentiable-transform-itf.h"

namespace kaldi {
namespace nnet3 {

struct NnetChainaTrainingOptions {
  NnetTrainerOptions nnet_config;
  chain::ChainTrainingOptions chain_config;
  bool apply_deriv_weights;
  BaseFloat unadapted_top_weight;
  BaseFloat unadapted_bottom_weight;
  int32 bottom_subsampling_factor;
  bool keep_embedding_context;
  bool bottom_model_test_mode;
  bool top_model_test_mode;

  NnetChainaTrainingOptions():
      apply_deriv_weights(true),
      unadapted_top_weight(1.0),
      unadapted_bottom_weight(0.5),
      bottom_subsampling_factor(1),
      keep_embedding_context(true),
      bottom_model_test_mode(false),
      top_model_test_mode(false) { }

  void Register(OptionsItf *opts) {
    nnet_config.Register(opts);
    chain_config.Register(opts);
    opts->Register("apply-deriv-weights", &apply_deriv_weights,
                   "If true, apply the per-frame derivative weights stored with "
                   "the example");
    opts->Register("unadapted-top-weight", &unadapted_top_weight,
                   "Scale used for the step sizes and max-change values when "
                   "training the top nnet and evaluating the unadapted output. "
                   "Affects how strongly the top nnets are trained by the "
                   "unadapted embeddings.  The scale on the adapted branch is "
                   "implicitly 1.0, but all these numbers also get multiplied "
                   "by language-specific weights obtained from the egs.");
    opts->Register("unadapted-bottom-weight", &unadapted_bottom_weight,
                   "Scale that is applied to the derivatives arising from the "
                   "unadapted branch of the top nnets, when training the bottom "
                   "nnet.   Affects how much we prioritize the unadapted "
                   "features for bottom nnet training.");
    opts->Register("bottom-subsampling-factor", &bottom_subsampling_factor,
                   "Determines the frequency at which we subsample the "
                   "embeddings from the bottom nnet.  Implicitly, the "
                   "subsampling factor in the top nnet is the overall "
                   "--frame-subsampling-factor (determined when we dumped "
                   "the egs) divided by this value.");
    opts->Register("keep-embedding-context", &keep_embedding_context,
                   "If true, we compute as much left/right context of the "
                   "embedding vectors (the output of the bottom nnet) as is "
                   "possible given the provided input features in the eg. "
                   "You'll generally only want this to be true "
                   "if the top network is recurrent or otherwise has "
                   "optional dependencies (for example: if it uses "
                   "StatisticsExtractionComponent, IfDefined(), Failover(), "
                   "etc.).");
    opts->Register("bottom-model-test-mode", &bottom_model_test_mode,
                   "Set this to true to disable training of the bottom nnet, "
                   "to use test-mode for any batch-norm or dropout"
                   "components in it, and to disable the accumulation of "
                   "statistics for the bottom model (to keep the batchnorm "
                   "stats frozen).  Setting this to false can be used to "
                   "evaluate train or valid probs.");
    opts->Register("top-model-test-mode", &top_model_test_mode,
                   "Set this to true to disable training of the top nnet, "
                   "to use test-mode for any batch-norm or dropout"
                   "components in it, and to disable the accumulation of "
                   "statistics for the top model (to keep the batchnorm "
                   "stats frozen).  Setting this to false can be used to "
                   "evaluate train or valid probs.");
  }
  void Check() {
    KALDI_ASSERT(unadapted_top_weight > 0.0 &&
                 unadapted_bottom_weight >= 0.0 &&
                 bottom_subsampling_factor > 0);
  }
};


/**
   This class, intended to mostly be accessed by NnetChainaTrainer, handles the
   logic of reading the models and their corresponding denominator FSTs from
   disk, and of writing out the corresponding (raw) trained models when
   this iteration of training has finished.

   The reason this is not entirely trivial is that we want to make it easy to
   support the multilingual case.  In this case there is one 'bottom' model (the
   embedding extractor) but there may be multiple 'top' models, each with their
   associated transition model and denominator FST, containing their own
   langauge name.  We use a directory to organize these.
 */
class NnetChainaModels {
 public:
  /**
     Constructor to which you pass the model directory and the den-fst
     directory.  The directory structure is:
       <model_dir>/bottom.raw
     should exist, and then for each language name (e.g. "english"), the following
     files should exist:
       <model_dir>/english.mdl <den_fst_dir>/english.fst <transform_dir>/english.ada
     There is no requirement that all these directories be distinct.

     In practice, the language name will be either "default", in the
     typical (monolingual) setup, or it might be arbitrary strings
     representing languages such as "english", "french", and so on.
     In general the language can be any string containing ASCII letters, numbers
     or underscores.

     The models and denominator FSTs will only be read when they are actually
     required, so languages that are not used by a particular job (e.g. because
     they were not represented in the egs) will not actually be read.


         @param [in] zero_components stats...  The --zero-component-stats option
                     from NnetChainaTrainingOptions::nnet_config.  Note: if
                     bottom_model_test_mode is true, we won't zero the stats on
                     the bottom model regardless of this value.
         @param [in] bottom_model_test_mode   If true, the bottom model will not be
                     trained (should be set to the same-named option from
                     NnetChainaTrainingOptions).  It's needed to know
                     whether to write the bottom model in WriteRawModels(),
                     and whether to zero the component stats, set batch-norm
                     test mode, and collapse the model.
         @param [in] top_model_test_mode   If true, the top model will not be
                     trained (should be set to the same-named option from
                     NnetChainaTrainingOptions).  It's needed to know
                     whether to write the top models in WriteRawModels(),
                     and whether to zero the component stats, set batch-norm
                     test mode, and collapse the model.
         @param  [in] model_dir  Directory where we'll find bottom.raw, and
                      <lang>.mdl for each language <lang> present in the egs
                      (the <lang> will be worked out from the key name from
                      "...?lang=xxx" in the key when reading the egs,
                      see ParseFromQueryString() in nnet-chain-utils.h.
         @param [in] den_fst_ir  Directory where we'll find the denominator
                      FST <lang>.fst for each language <lang> present in
                      the egs.
         @param [in] transform_dir  Directory where we'll find the
                      transforms (of type DifferentiableTransformItf),
                      as files <lang>.ada for each language <lang> present
                      in the egs.
   */
  NnetChainaModels(bool zero_component_stats,
                   bool bottom_model_test_mode,
                   bool top_model_test_mode,
                   const std::string &model_dir,
                   const std::string &den_fst_dir,
                   const std::string &transform_dir);

  Nnet* GetBottomNnet();

  /**
     Returns the AmNnetSimple object corresponding to a given language
     name (e.g. "default", "english", "french").  Note: the model
     file <model_dir>/<language_name>.mdl will contain a TransitionModel and an
     AmNnetSimple object
   */
  AmNnetSimple *GetNnetForLang(const std::string &language_name);

  TransitionModel *GetTransitionModelForLang(
      const std::string &language_name);


  fst::StdVectorFst *GetDenFstForLang(const std::string &language_name);

  // This convenience function returns the Nnet object in the
  // AmNnetSimple object returned by 'GetNnetForLang'.
  Nnet *GetRawNnetForLang(const std::string &language_name);

  differentiable_transform::DifferentiableTransformMapped *GetTransformForLang(
      const std::string &language_name);

  // Writes the files
  //  <model_out_dir>/bottom.<job_id>.raw
  // and, for each language <lang> that we accessed,
  //  <model_out_dir>/<lang>.<job_id>.raw
  void WriteRawModels(const std::string &model_out_dir,
                      bool binary,
                      int32 job_id);

  ~NnetChainaModels();
 private:
  // This function sets "pathname" to the string:
  // <dir>/<name>.<suffix>
  void GetPathname(const std::string &dir,
                   const std::string &name,
                   const std::string &suffix,
                   std::string *pathname);

  // This version of GetPathname() sets "pathname" to the string:
  // <dir>/<name>.<job_id>.<suffix>
  void GetPathname(const std::string &dir,
                   const std::string &name,
                   int32 job_id,
                   const std::string &suffix,
                   std::string *pathname);

  // struct LanguageInfo contains the data that is stored per language.
  struct LanguageInfo {
    // am_nnet comes from <model_dir>/<language_name>.mdl, which also
    // stores a TransitionModel.
    TransitionModel trans_model;
    AmNnetSimple am_nnet;
    // den_fst comes from <den_fst_dir>/<language_name>.fst
    fst::StdVectorFst den_fst;
    // transform comes from <transform_dir>/<language_name>.ada
    differentiable_transform::DifferentiableTransformMapped transform;
  };


  // get the LanguageInfo* for this language, creating it (and reading its
  // contents from disk) if it does not already exist.
  LanguageInfo *GetInfoForLang(const std::string &lang);

  // True if we are going to call ZeroComponentStats() on models when they are
  // read.
  bool zero_component_stats_;
  // A copy of the "bottom-model-test-mode" option in NnetChainaTrainingOptions.
  bool bottom_model_test_mode_;
  // A copy of the "top-model-test-mode" option in NnetChainaTrainingOptions.
  bool top_model_test_mode_;
  // Directory where models are located.
  std::string model_dir_;
  // Directory where denominator FSTs are located.
  std::string den_fst_dir_;
  // Directory where transforms (type: DifferentiableTransformMapped) are located.
  std::string transform_dir_;

  // This corresponds to <model_dir>/bottom.raw.
  Nnet bottom_nnet_;
  // The left and right context of bottom_nnet_.
  int32 bottom_nnet_left_context_;
  int32 bottom_nnet_right_context_;


  std::unordered_map<std::string, LanguageInfo*, StringHasher> lang_info_;
};


/**
   This object, which has a similar function to NnetChainTrainer, trains the
   'top' model for a single language and (optionally) outputs the derivatives
   required to obtain the 'bottom' model.
 */
class NnetChainaTopTrainer {
 public:
  /**
     Constructor.
      @param [in] lang_name  The name of the language this corresponds to
                             (needed for diagnostics).   E.g. "default",
                             "english".
      @param [in] config     Options class
      @param [in] den_fst    The denominator FST for this language
      @param [in] transform  The transform object which will be used to produce adapted
                             features after the first pass of training.
      @param [in,out] nnet   The neural net we are training.  Expected to have
                             outputs called "output-si" (speaker-independent
                             output), "output", "output-si-xent", "output-xent",
                             and an input called "input".  This class does not
                             take ownership of the pointer, but it will modify
                             its parameters (and stored statistics) during
                             training.
   */
  NnetChainaTopTrainer(
      const std::string &lang_name,
      const NnetChainaTrainingOptions &config,
      const fst::StdVectorFst &den_fst,
      const differentiable_transform::DifferentiableTransformMapped &transform,
      Nnet *nnet);

  /**  Train on one minibatch.
          @param [in] input  The input (unadapted) features, most likely the embeddings
                    that are the output of the 'bottom' nnet.  Assumed to form a
                    regular grid with the 't' value having higher stride, so the
                    first 'num_sequences' rows would correspond to the
                    lowest-numbered frames for all sequences, and so on.
          @param [in] num_sequences The number of sequences/chunks represented
                    in 'input' (a.k.a. the minibatch size).  Actually this must
                    be equal to supervision.num_sequences, but it's easier for
                    reasons of clarity and documentation to repeat it here.
          @param [in] num_spk  The total number of speakers.  Must be >1, and must divide
                     num_sequences.   The number of sequences per speaker
                     must be the same for all speakers (it will equal num_sequences / num_spk),
                     and the sequences for a speaker must be consecutively numbered.
          @param [in] first_input_t The 't' value corresponding to the first
                     input frame (will normally be a negative number,
                     corresponding to the left context we are giving to the
                     'top' model, since we renumber to ensure that the sequences
                     have 't' values starting from 0).  The 't' values at the
                     input will be consecutive, and the number of frames per
                     sequence will equal input.NumRows() / num_sequences.  Note:
                     if the embeddings are computed at a lower frame rate than
                     the original features, we renumber things to make the
                     embeddings consecutive.
          @param [in] top_subsampling_factor  The subsampling factor of the top network
                     (which will equal the frame subsampling factor implicit in the original
                     egs that we read, divided by bottom_subsampling_factor).  E.g. this
                     might frequently be 1 or 3.  The frames at the output of the 'top'
                     nnet are evaluated for 't' values that are multiples of
                     'top_subsampling_factor', starting from t=0.
          @param [in] deriv_weights  Per-frame weights that will be applied to the derivatives
                     w.r.t. the objective function.  Dimension is expected to be either
                     input.NumRows(), or zero (in which case it is treated the same as a
                     vector containing all ones).
          @param [in] supervision  The chain supervision object representing the objective
                     function at the output.  Its num_sequences must equal the
                     num_sequences passed into this function as a separate argument.
          @param [in] model_training_scale  A scale we'll apply to the parameter changes
                     and max-change values when taking any step.  This will be
                     referred to elsewhere as top_weight, or "tw" when present in
                     keys of egs in scp files; we'll have a separately specifiable
                     weight for the bottom nnet.  If this is zero, we won't be training
                     the top model on this eg at all.
          @param [out] input_deriv  If non-NULL, the derivative of the objective function
                     w.r.t. the input features will be written to here (this function
                     will set it using Swap(), so you don't need to correctly size it).
          @return   Returns true if it successfully trained on this minbiatch,
                    false on error (e.g. if a NaN was generated, which should
                    not really happen).
  */
  bool Train(const CuMatrixBase<BaseFloat> &input,
             int32 num_sequences,
             int32 num_spk,
             int32 first_input_t,
             int32 top_subsampling_factor,
             const VectorBase<BaseFloat> &deriv_weights,
             const chain::Supervision &supervision,
             BaseFloat model_training_scale,
             CuMatrix<BaseFloat> *input_deriv = NULL);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  // Calls kaldi::nnet3::ConsolidateMemory() on nnet_ and delta_nnet_; we do
  // this after the first minibatch of training, to reduce fragmentation.
  void ConsolidateMemory();

  ~NnetChainaTopTrainer();
 private:

  // We use this as an index with which to look up computations, kind of like a
  // lookaside buffer; it avoids creating a much larger structure with large
  // vectors of Indexes in it.
  struct ComputationStructure {
    bool adapted;
    bool train_model;
    bool need_input_deriv;
    int32 num_sequences;
    int32 frames_per_sequence_in;
    int32 frames_per_sequence_out;
    int32 first_input_t;
    int32 top_subsampling_factor;
    inline bool operator == (const ComputationStructure &other) const {
      return adapted == other.adapted &&
          train_model == other.train_model &&
          need_input_deriv == other.need_input_deriv &&
          num_sequences == other.num_sequences &&
          frames_per_sequence_in == other.frames_per_sequence_in &&
          frames_per_sequence_out == other.frames_per_sequence_out &&
          first_input_t == other.first_input_t &&
          top_subsampling_factor == other.top_subsampling_factor;
    };
    ComputationStructure (const ComputationStructure &other) = default;
    ComputationStructure &operator = (
        const ComputationStructure &other) = default;
    /**
       Constructor.
       @param [in] adapted  True if we want the outputs from "output" and
                   "output-xent", and false if we want the outputs from
                    "output-si" and "output-si-xent".
       @param [in] train_model   True if we will be training the acoustic
                   model with this example.
       @param [in] need_input_deriv  True if we need the derivative w.r.t.
                     the features that are the input to this computation.
       @param [in] num_sequences  The number of sequences in this minibatch
                     (a.k.a. the minibatch size).
       @param [in] frames_per_sequence_in  The number of frames for each sequence
                    of input features.  They are assumed to be consecutively
                    numbered.
       @param [in] frames_per_sequence_out  The 'frames_per_sequence' in
                    the ChainSupervision object, i.e. the length of the
                    output sequences of the computation.
       @param [in] first_input_t  The first 't' value in the input
                    sequence; will normally be negative (corresponding to
                    the negative of the number of frames of left context).
       @param [in] top_subsampling_factor  Frame subsampling factor at the
                    output; e.g., 3 would mean we are evaluating the output
                    at frames t=0, t=3, and so on.
    */
    ComputationStructure(bool adapted,
                         bool train_model,
                         bool need_input_deriv,
                         int32 num_sequences,
                         int32 frames_per_sequence_in,
                         int32 frames_per_sequence_out,
                         int32 first_input_t,
                         int32 top_subsampling_factor);
  };
  struct ComputationHasher {
    inline size_t operator() (const ComputationStructure &s) const {
      return (s.adapted ? 33 : 0) +
          (s.train_model ? 333 : 0) +
          size_t(s.num_sequences) +
          10  * size_t(s.frames_per_sequence_in) +
          100 * size_t(s.frames_per_sequence_out) +
          1000 * size_t(s.first_input_t) +
          10000 * size_t(s.top_subsampling_factor);
    }
  };

  // This is a faster lookup mechanism for the computation than
  // is provided by the compiler's inherent caching.
  std::unordered_map<ComputationStructure,
                     std::shared_ptr<const NnetComputation>,
                     ComputationHasher> computation_map_;

  // This wraps the call to the compiler.  See constructor
  // of struct ComputationStructure for more documentation.
  std::shared_ptr<const NnetComputation> GetComputation(
      const ComputationStructure &s);


  /**
    This does the training on the unadapted branch ("si" / speaker-independent)
    of the neural net.
      @param [in] input    The input features, as supplied to Train().  Order
                          of rows is: the first frame of all sequences; the
                          second frame of all sequences; and so on.
      @param [in] computation  The computation corresponding to the unadapted
                               branch of the nnet.
      @param [in] supervision   The chain supervision object.  The nnet output
                               dimensions are worked out from this, as well as
                               using this object to compute the objective function.
      @param [in] model_training_scale  A scale we'll apply to the parameter
                        changes and max-change values when taking any step.
                        This will be the product of the top_weight ("tw") from
                        the key in the egs, with the value of the
                        --unadapted-top-weight option.  If this is zero, we
                        won't be training the top model on this eg at all.
      @param [in] deriv_weights  Weights to be applied to the derivatives for the
                               corresponding frames of the output (order is:
                               first frame for all sequences; second frame for
                               all sequences, etc.).  May be stored with the
                               egs.  If this is the empty vector or
                               --apply-deriv-weights=false, they won't be
                               appplied.
      @param [out] posterior    The posteriors from the numerator forward-backward
                               on the adaptation model will be written to here.
                               The number of frames will be the number of frames in
                               the output sequence (supervision.frames_per_sequence),
                               and the order is: all sequences' frame 0; then all
                               sequences' frame 1; and so on.
      @param [out] input_deriv   Derivative w.r.t. the input features; this will
                               be set via Swap(), if it is not NULL.  Any weight to
                               (be applied e.g. opts_.unadapted_bottom_weight),
                               should be applied by the caller.
      @return  Returns true if the training went through successfully
            (it should very rarely return false, e.g. if a NaN was generated).
  */
  bool TrainUnadapted(const CuMatrixBase<BaseFloat> &input,
                      const NnetComputation &computation,
                      const chain::Supervision &supervision,
                      BaseFloat model_training_scale,
                      const CuVectorBase<BaseFloat> &deriv_weights,
                      Posterior *posterior,
                      CuMatrix<BaseFloat> *input_deriv);

  /**
     Converts the format of the posterior from how it is at the output of the
     network to how it is at the input (i.e. in the embedding space).
     Basically, this will consist of padding with empty posteriors for the
     "context frames", and possibly upsampling the posteriors (by just repeating
     each one for, say, 3 frames, if top_subsampling_factor == 3).  The
     rule we'll use is: copy the posterior from the output frame that
     is closest in numbering, rounding down in case of ties (i.e., for even
     subsampling factor).

        @param [in] post_at_output  The posterior that needs to be padded,
                      consisting of 'num_sequences' sequences, each with 't'
                      values starting at zero, at multiples of
                      'top_subsampling_factor', and with number of 't' values
                      determined by: num_frames_out = post_at_output.size() /
                      num_sequences.  The 't' has the larger stride than the
                      minibatch index 'n', so it's: frame t=0 of all sequences,
                      then frame t=1*top_subsampling_factor of all sequences,
                      and so on.
        @param [in] num_sequences  The number of sequences/chunks
        @param [in] first_input_t  The first 't' value at the input, for which
                      we need a posterior for (note: negative 't' values will
                      get zero posterior).  Implicitly, first_output_t = 0.
                      The number of input frames is worked out as
                      post_at_input->size() / num_sequences; the 't' values
                      at the input are assumed to be consecutive.
        @param [in] top_subsampling_factor  The number of frames with which
                      't' values at the output are separated.
        @param [in] pdf_map  This is either the empty vector (meaning:
                     the DifferentiableTransform object deals with pdf-ids
                     directly), or it is a map from pdf-ids to cluster-ids.
                     This would actually be obtained from build-tree-two-level
                     after building a two-level tree, and it would be stored
                     in the .ada object.  The actual class labels that
                     the DifferentiableTransform object deals with, will
                     be the values stored in 'pfd_map' (i.e. these cluster-ids).
        @param [in] num_classes  Provided for checking purposes only: the
                     number of classes that the DifferentiableTransform object
                     expects.  If pdf_map is empty we expect this to be the
                     same as the number of pdf-ids (and the ints in
                     post_at_output to be in the range [0, num_classes - 1]).
                     If pdf_map is nonempty, we expect this to be the same
                     as the maximum element in pdf_map, plus one.
        @param [out] post_at_input  The posterior after padding and possibly
                      subsampling.  Should have the correct size but its
                      elements are expected to be empty at entry.  Like
                      post_at_output, the 't' has the larger stride than
                      the minibatch-index 'n'.

  */
  void ConvertPosterior(const Posterior &post_at_output,
                        int32 num_sequences,
                        int32 first_input_t,
                        int32 top_subsampling_factor,
                        const std::vector<int32> &pdf_map,
                        int32 num_classes,
                        Posterior *post_at_input);

  /**
     Does the adapted pass of training.
         @param [in] computation  The adapted version of the
                     computation (this one uses the outputs
                     "output" and "output-xent" instead of
                     "output-si" and "output-si-xent".
         @param [in] supervision  The chain supervision
                     object, containing information derived
                     from the numerator lattices.
         @param [in] model_training_scale  A scale we'll apply to the parameter changes
                     and max-change values when taking any step.  This will be
                     referred to elsewhere as top_weight, or "tw" when present in
                     keys of egs in scp files; we'll have a separately specifiable
                     weight for the bottom nnet.  If this is zero, we won't be training
                     the top model on this eg at all.
         @param [in] deriv_weights  Weights to be applied to the derivatives for the
                     corresponding frames of the output (order is:
                     first frame for all sequences; second frame for
                     all sequences, etc.).  May be stored with the
                     egs.  If this is the empty vector or
                     --apply-deriv-weights=false, they won't be
                     appplied.
         @param [in] input  The adapted input features.  Provided as a non-const
                     pointer because it is consumed destructively (via Swap()).
         @param [in,out] input_deriv  If non-NULL, the
                     feature derivative w.r.t. the [speaker-adapted] input
                     features will be written to this location.  It's
                     done via Swap(), so it doesn't have to be correctly
                     sized on entry.
         @return
   */
  bool TrainAdapted(const NnetComputation &computation,
                    const chain::Supervision &supervision,
                    BaseFloat model_training_scale,
                    const CuVectorBase<BaseFloat> &deriv_weights,
                    CuMatrix<BaseFloat> *input,
                    CuMatrix<BaseFloat> *input_deriv);

  // This function increments num_minibatches_processed_, but before
  // doing so, if it notices that it is zero it makes certain calls
  // to ConsolidateMemory()
  void IncrementNumMinibatches();

  std::string lang_name_;

  const NnetChainaTrainingOptions &opts_;
  chain::DenominatorGraph den_graph_;
  const differentiable_transform::DifferentiableTransformMapped &transform_;
  CachingOptimizingCompiler compiler_;


  Nnet *nnet_;
  Nnet *delta_nnet_;  // stores the change to the parameters on each training
                      // iteration.

  // These objects keep track of the objective-function values for the 4
  // outputs.  We have the regular output (sequence objective) and the 'xent'
  // output for cross-entropy regularization, and there are speaker independent
  // (si) versions of those outputs also.
  ObjectiveFunctionInfo output_si_objf_;
  ObjectiveFunctionInfo output_si_xent_objf_;
  ObjectiveFunctionInfo output_objf_;
  ObjectiveFunctionInfo output_xent_objf_;

  // Number of minibatches processed.  Note: we actually train the nnet twice
  // per minibatch, because there are the speaker-independent and
  // speaker-dependent passes.
  int32 num_minibatches_processed_;

  // stats for max-change (for speaker-independent phases of training)
  MaxChangeStats max_change_stats_si_;
  // stats for max-change (for speaker-adapted phases of training)
  MaxChangeStats max_change_stats_;
};



/**
   This object, which has a similar function to NnetChainTrainer, takes care of
   evaluating and possibly training the 'bottom' model.
*/
class NnetChainaBottomTrainer {
 public:
  /**
     Constructor.
      @param [in] opts    Options class.  This class maintains a reference to it,
                          so don't delete it.
      @param [in,out]  nnet   The neural net we are training.  Expected (for now)
                            to have an input called 'input' (corresponding to
                            the original input features and an output called
                            'output' (corresponding to the embeddings).
   */
  NnetChainaBottomTrainer(const NnetChainaTrainingOptions &opts,
                          Nnet *nnet);

  /**  Train on one minibatch.
          @param [in] num_sequences The number of sequences/chunks represented
                    in 'input' (a.k.a. the minibatch size).
          @param [in] first_input_t  The  't' value corresponding to the first input
                     frame (will normally be a negative number).  The 't' values at
                     the input will be consecutive, and the number of frames per sequence
                     will equal input.NumRows() / num_sequences.  Note: if the embeddings
                     are computed at a lower frame rate than the original features, we
                     renumber things to make the embeddings consecutive.
               (Note: bottom_subsampling_factor was passed in in the constructor).
          @param [in] first_output_t  The  't' value corresponding to the first output
                     frame (will normally be a negative number, corresponding to the left
                     context we are giving to the 'top' model, since we assume that the
                     sequences have 't' values starting from 0).  The 't' values at
                     the output will be separated by the 'bottom_subsampling_factor'
                     which was given to the constructor.  (We'll renumber them
                     by dividing them by 'bottom_subsampling_factor' before giving
                     them to the 'top' network.
          @param [in]  frames_per_sequence_out  The  number of output frames per sequence.
                     This is determined by the context of the top and bottom nnets
                     and the "keep_embedding_context" config value.
          @param [in] train_model   True if we'll be training the bottom model
                     for this eg.  If this is false, a backward pass will not be.
                     needed, and this function will return NULL
          @param [in] input  The input features, most likely raw MFCC or filterbank
                     features.   A pointer, since it is consumed destructively
                     (via 'swap').
          @param [out] output   The output will be written to here.  Does not have
                     to be correctly sized (we'll copy using Swap()).
          @return   Returns the NnetComputer object that we did the computation with,
                    if train_model == true (otherwise, returns NULL).
                    The user should either pass this into Backward(), or delete it.
  */
  NnetComputer* Forward(int32 num_sequences,
                        int32 first_input_t,
                        int32 first_output_t,
                        int32 frames_per_sequence_out,
                        bool train_model,
                        CuMatrix<BaseFloat> *input,
                        CuMatrix<BaseFloat> *output);


  /**
      Does the backward pass, which will do model training.  This should only be
      called if the bottom nnet needs to be trained.
         @param [in] model_training_scale  A scale we'll apply to the parameter changes
                     and max-change values when taking the step..  This will be
                     referred to elsewhere as bottom_weight, or "bw" when present in
                     keys of egs in scp files; we'll have a separately specifiable
                     weight for the top nnet.  If this is zero, we won't be training
                     the top model on this eg at all (and we'll expect 'false' to
                     have been passed in for the 'train_model' arg on the corresponding
                     call to Forward()).
         @param [in] computer   The computer object returned from the
                    forward pass.  This function takes ownership of it and
                    will delete it when done with it.
         @param [in] output_deriv  The derivative w.r.t. the output of
                    the forward pass.  It is consumed destructively
                    by this function.

   */
  void Backward(BaseFloat model_training_scale,
                NnetComputer *computer,
                CuMatrix<BaseFloat> *output_deriv);

  // Prints the max-change stats for the bottom nnet.
  void PrintTotalStats() const;

  // Calls kaldi::nnet3::ConsolidateMemory() on nnet_ and delta_nnet_; we do
  // this after the first minibatch of training, to reduce fragmentation.
  void ConsolidateMemory();

  ~NnetChainaBottomTrainer();
 private:

  // We use this as an index with which to look up computations, kind of like a
  // lookaside buffer; it avoids creating a much larger structure with large
  // vectors of Indexes in it.
  struct ComputationStructure {
    bool train_model;
    int32 num_sequences;
    int32 frames_per_sequence_in;
    int32 frames_per_sequence_out;
    int32 first_input_t;
    int32 first_output_t;
    inline bool operator == (const ComputationStructure &other) const {
      return train_model == other.train_model &&
          num_sequences == other.num_sequences &&
          frames_per_sequence_in == other.frames_per_sequence_in &&
          frames_per_sequence_out == other.frames_per_sequence_out &&
          first_input_t == other.first_input_t &&
          first_output_t == other.first_output_t;
  };
    ComputationStructure (const ComputationStructure &other) = default;
    ComputationStructure &operator = (
        const ComputationStructure &other) = default;
    /**
       Constructor.
       @param [in] train_model  True if we are going to train the bottom model.
       @param [in] need_input_deriv  True if we need the derivative w.r.t.
                     the features that are the input to this computation.
       @param [in] num_sequences  The number of sequences in this minibatch
                     (a.k.a. the minibatch size).
       @param [in] frames_per_sequence_in  The number of frames for each sequence
                    of input features.  They are assumed to be consecutively
                    numbered.
       @param [in] frames_per_sequence_out  The 'frames_per_sequence' in
                    the ChainSupervision object, i.e. the length of the
                    output sequences of the computation.
       @param [in] first_input_t  The first 't' value in the input
                    sequence; will normally be negative (corresponding to
                    the negative of the number of frames of left context).
    */
    ComputationStructure(bool train_model,
                         int32 num_sequences,
                         int32 frames_per_sequence_in,
                         int32 frames_per_sequence_out,
                         int32 first_input_t,
                         int32 first_output_t);
  };
  struct ComputationHasher {
    inline size_t operator() (const ComputationStructure &s) const {
      return size_t(s.num_sequences) +
          10  * size_t(s.frames_per_sequence_in) +
          100 * size_t(s.frames_per_sequence_out) +
          1000 * size_t(s.first_input_t) +
          10000 * size_t(s.first_output_t);
    }
  };

  // This is a faster lookup mechanism for the computation than
  // is provided by the compiler's inherent caching.
  std::unordered_map<ComputationStructure,
                     std::shared_ptr<const NnetComputation>,
                     ComputationHasher> computation_map_;

  // This wraps the call to the compiler.  See constructor
  // of struct ComputationStructure for more documentation.
  std::shared_ptr<const NnetComputation> GetComputation(
      const ComputationStructure &s);



  /**
     Converts the format of the posterior from how it is at the output of the
     network to how it is at the input (i.e. in the embedding space).
     Basically, this will consist of padding with empty posteriors for the
     "context frames", and possibly upsampling the posteriors (by just repeating
     each one for, say, 3 frames, if top_subsampling_factor == 3).

     The number of frames per sequence at the output will equal
     post_at_output.size() / num_sequences, and the number of frames per
     sequence at the input will equal post_at_inptu->size() / num_sequences
     (note: this means 'post_at_input is expected to be appropriately sized
     when this function is called).
  */
  void ConvertPosterior(const Posterior &post_at_output,
                        int32 num_sequences,
                        int32 first_input_t,
                        int32 top_subsampling_factor,
                        Posterior *post_at_input);

  const NnetChainaTrainingOptions opts_;

  Nnet *nnet_;
  Nnet *delta_nnet_;  // stores the change to the parameters on each training
                      // iteration.

  CachingOptimizingCompiler compiler_;

  // Number of minibatches processed.
  int32 num_minibatches_processed_;

  // stats for max-change
  MaxChangeStats max_change_stats_;
};



/**
   This class is for single-threaded training of neural nets using the 'chain'
   model and our adaptation framework
*/
class NnetChainaTrainer {
 public:
  /**
     Constructor
        @param [in] config  Options class
        @param [in] models  Object that provides access to the models and
                         denominator FSTs, indexed as appropriate by language-id.
   */
  NnetChainaTrainer(const NnetChainaTrainingOptions &config,
                    NnetChainaModels *models);

  /* Train on one minibatch.
           @param [in] key  The key the example had in the archive.  This is
                        used to work out the language name.
            @param [in] eg  The example we are training on.  It is expected
                        to have an input named 'input' (the features) and an
                         output named 'output' (containing the chain supervision
                         object).  We'll make use of the chunks_per_spk member
                         of the NnetChainSupervision object, which is not used
                         outside the 'chaina' framework.
  */
  void Train(const std::string &key,
             const NnetChainExample &eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  // Prints out the max-change stats (if nonzero): the percentage of time that
  // per-component max-change and global max-change were enforced.
  void PrintMaxChangeStats() const;

  ~NnetChainaTrainer();

 private:

  void GetContextInfo(const std::string &lang,
                      int32 *bottom_left_context,
                      int32 *bottom_right_context,
                      int32 *top_left_context,
                      int32 *top_right_context);


  NnetChainaTopTrainer *GetTopTrainerForLang(const std::string &lang);


  const NnetChainaTrainingOptions &opts_;
  // pointer to object owned outside this class.
  NnetChainaModels *models_;

  // left and right context of bottom model.
  int32 bottom_left_context_;
  int32 bottom_right_context_;

  NnetChainaBottomTrainer bottom_trainer_;
  // map from language name (e.g. "default", "english", "french") to
  // the object that trains the corresponding 'top' nnet.
  std::unordered_map<std::string, NnetChainaTopTrainer*,
                     StringHasher> top_trainers_;
};


} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_CHAINA_TRAINING_H_
