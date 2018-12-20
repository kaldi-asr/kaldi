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
  BaseFloat unadapted_deriv_scale;
  BaseFloat unadapted_backprop_scale;
  bool train_bottom_nnet;  // True if we will be training the bottom nnet.
  bool train_top_nnet;    // True if we will be training the top nnet.
  int32 bottom_subsampling_factor;
  bool keep_embedding_context;

  NnetChainaTrainingOptions():
      apply_deriv_weights(true),
      unadapted_deriv_scale(0.5),
      unadapted_backprop_scale(1.0),
      train_bottom_nnet(true),
      train_top_nnet(true),
      bottom_subsampling_factor(1),
      keep_embedding_context(true) { }

  void Register(OptionsItf *opts) {
    nnet_config.Register(opts);
    chain_config.Register(opts);
    opts->Register("train-bottom-nnet", &train_bottom_nnet,
                   "Set this to false to disable training of bottom nnet.");
    opts->Register("train-top-nnet", &train_top_nnet,
                   "Set this to false to disable training of top nnet.");
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
    opts->Register("apply-deriv-weights", &apply_deriv_weights,
                   "If true, apply the per-frame derivative weights stored with "
                   "the example");
    opts->Register("unadapted-deriv-scale", &unadapted_deriv_scale,
                   "Scale on the derivatives (and max-change values, for the top "
                   "nnet) for the unadapted branches of the nnets (at the outputs "
                   "output-si and output-si-xent.  Affects how strongly the nnets "
                   "are trained by the unadapted embeddings.  Note: this also "
                   "affects the derivatives given to the bottom nnet.  The scale "
                   "on the adapted branch is implicitly 1.0.");
    opts->Register("unadapted-backprop-scale", &unadapted_backprop_scale,
                   "Scale that is applied to the derivatives arising from the "
                   "unadapted branch of the top nnets, when backpropagating "
                   "to the embeddings.  Affects how much we prioritize the "
                   "unadapted features.  Note: this is effectively multiplied by "
                   "unadapted-deriv-scale; unadapted-deriv-scale also affects "
                   "training of the top nnet.");
  }
  void Check() {
    KALDI_ASSERT(unadapted_deriv_scale > 0.0 &&
                 unadapted_backprop_scale >= 0.0);
    // TODO: add more checks?
  }

};


/**
   This struct, intended mostly to be accessed by NnetChainaTrainer, handles the
   logic of reading the models and their corresponding denominator FSTs from
   disk, and of writing out the corresponding (raw) trained models when
   this iteration of training has finished.

   The reason this is not entirely trivial is that we want to make it easy
   to support the multilingual case.  In this case there is one 'bottom'
   model (the embedding extractor) but there may be multiple 'top' models,
   each with their associated transition model and denominator FST, and their
   own name.  We use a directory to organize these.
 */
class NnetChainaModels {
 public:
  /**
     Constructor to which you pass the model directory and the den-fst
     directory.  The directory structure is:
       <model_dir>/bottom.raw
     should exist, and then for each language name "lang", the following
     files should exist:
       <model_dir>/lang.mdl <den_fst_dir>/lang.fst <transform_dir>/lang.ada

     In practice, the language name will be either "default", in the
     typical (monolingual) setup, or it might be arbitrary strings
     representing languages such as "english", "french" (in

     In general the language can be any string containing ASCII letters, numbers
     or underscores, and it will be a suffix of the key in the egs that we are
     reading, separated from them by a "-".  E.g. if the key is
     "143213423-1234123432_10-english", the language would be "english".
     The models and denominator FSTs will only be read when they are
     actually required.
   */
  NnetChainaModels(const std::string &model_dir,
                   const std::string &den_fst_dir,
                   const std::string &transform_dir);

  Nnet* GetBottomNnet();

  int32 BottomNnetLeftContext() const;
  int32 BottomNnetRightContext() const;

  /**
     Returns the AmNnetSimple object corresponding to a given language
     name (e.g. "default", "english", "french").  Note: the model
     file <model_dir>/<language_name>.mdl will contain a TransitionModel and an
     AmNnetSimple object
   */
  AmNnetSimple *GetNnetForLang(const std::string &language_name);


  const TransitionModel *GetTransitionModelForLang(
      const std::string &language_name);


  fst::StdVectorFst *GetDenFstForLang(const std::string &language_name);

  // This convenience function returns the Nnet object in the
  // AmNnetSimple object returned by 'GetNnetForLang'.
  Nnet *GetRawNnetForLang(const std::string &language_name);

  differentiable_transform::DifferentiableTransform *GetTransformForLang(
      const std::string &language_name);


  // Writes to 'langs' a vector (in no particular order) of the
  // names of all the languages that have been loaded (this will depend
  // on whether they were represented in the egs).  This might
  // be [ "default" ], or it might be [ "english", "french" ], for
  // example.
  void ListAllLanguages(std::vector<std::string> *langs);

  // Writes the files
  //  <model_out_dir>/bottom.<job_id>.raw
  // and, for each language <lang> that we accessed,
  //  <model_out_dir>/<lang>.<job_id>.raw
  void WriteRawModels(const std::string &model_out_dir,
                      int32 job_id);

  ~NnetChainaModels();
 private:
  // Directory where models are located.
  std::string model_dir_;
  // Directory where denominator FSTs are located.
  std::string den_fst_dir_;
  // Directory where transforms (type: DifferentiableTransform) are located.
  std::string transform_dir_;

  // This corresponds to <model_dir>/bottom.raw.
  Nnet bottom_nnet_;
  // The left and right context of bottom_nnet_.
  int32 bottom_nnet_left_context_;
  int32 bottom_nnet_right_context_;

  // Data that is loaded per language.

  struct LanguageInfo {
    // trans_model and am_nnet come from <model_dir>/<language_name>.mdl
    TransitionModel trans_model;
    AmNnetSimple am_nnet;
    // den_fst comes from <den_fst_dir>/<language_name>.fst
    fst::StdVectorFst den_fst;
    // trans comes from <transform_dir>/<language_name>.ada
    differentiable_transform::DifferentiableTransform *trans;
  };

  std::unordered_map<std::string, LanguageInfo*> lang_info_;

};

/**
   steps of training:

   for a minibatch:
     work out the language
     work out how many chunks per speaker
     work out the context and how many frames of embeddings are
     needed.

     See whether we need backprop and model update for the two
      passes of training.
     Make the 3 computations.



    We need

 */


/**
   This object, which has a similar function to NnetChainTrainer, trains the
   'top' model for a single language and (optionally) outputs the derivatives
   required to obtain the 'bottom' model.
 */
class NnetChainaTopTrainer {
 public:
  /**
     Constructor.
      @param [in] lang_name  The name of the language this corresponds to (for diagnostics).
                             E.g. "default", "english", etc.
      @param [in] config   Options class
      @param [in] train_top_model   True if we are training the 'top' model... this is one
                           configuration value that's outside 'config', that we need.
      @param [in] den_fst   The denominator FST for this language
      @param [in] transform  The transform object which will be used to produce adapted
                             features after the first pass of training.
      @param [in] compiler   A pointer to the compiler we are to use (we make it
                             owned externally for easier caching).
      @param [in,out]  nnet   The neural net we are training.  Expected to have outputs
                             called "output-si" (speaker-independent output), "output",
                             "output-si-xent", "output-xent", and an input called
                             "input".  This class does not take ownership of the pointer.
   */
  NnetChainaTopTrainer(const std::string &lang_name,
                       const NnetChainaTrainingOptions &config,
                       const fst::StdVectorFst &den_fst,
                       const differentiable_transform::DifferentiableTransform &transform,
                       CachingOptimizingCompiler *compiler,
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
                    reasons of clarity and documentation repeat it here.
          @param [in] num_spk  The total number of speakers.  Must be >1, and must divide
                     num_sequences.   The number of sequences per speaker
                     must be the same for all speakers (it will equal num_sequences / num_spk),
                     and the sequences for a speaker must be consecutively numbered.
          @param [in] first_input_t  The  't' value corresponding to the first input
                     frame (will normally be a negative number, corresponding to the left
                     context we are giving to the 'top' model, since we assume that the
                     sequences have 't' values starting from 0).  The 't' values at
                     the input will be consecutive, and the number of frames per sequence
                     will equal input.NumRows() / num_sequences.  Note: if the embeddings
                     are computed at a lower frame rate than the original features, we
                     renumber things to make the embeddings consecutive.
          @param [in] top_subsampling_factor  The subsampling factor of the top network
                     (which will equal the frame subsampling factor implicit in the original
                     egs that we read, divided by bottom_subsampling_factor).  E.g. this
                     might frequently be 1 or 3.  The frames at the output of the 'top'
                     nnet are evaluated for 't' values that are multiples of
                     'top_subsampling_factor', starting from t=0.
          @param [in] supervision  The chain supervision object representing the objective
                     function at the output.  Its num_sequences must equal the
                     num_sequences passed into this function separately.
          @param [out] input_deriv  If non-NULL, the derivative of the objective function
                     w.r.t. the input features will be written to here (this function assumes
                     that its value is zero on entry).
          @return   Returns true if it successfully trained on this minbiatch, false
                    on error (e.g. if a NaN was generated, which should not really happen).
  */
  bool Train(const CuMatrixBase<BaseFloat> &input,
             int32 num_sequences,
             int32 num_spk,
             int32 first_input_t,
             int32 top_subsampling_factor,
             const VectorBase<BaseFloat> &deriv_weights,
             const chain::Supervision &supervision,
             CuMatrixBase<BaseFloat> *input_deriv = NULL);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  // Prints out the max-change stats (if nonzero): the percentage of time that
  // per-component max-change and global max-change were enforced.
  void PrintMaxChangeStats() const;

  ~NnetChainaTopTrainer();
 private:

  // We use this as an index with which to look up computations, kind of like a
  // lookaside buffer; it avoids creating a much larger structure with large
  // vectors of Indexes in it.
  struct ComputationStructure {
    bool adapted;
    bool need_input_deriv;
    int32 num_sequences;
    int32 frames_per_sequence_in;
    int32 frames_per_sequence_out;
    int32 first_input_t;
    int32 top_subsampling_factor;
    inline bool operator == (const ComputationStructure &other) const {
      return adapted == other.adapted &&
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
                         bool need_input_deriv,
                         int32 num_sequences,
                         int32 frames_per_sequence_in,
                         int32 frames_per_sequence_out,
                         int32 first_input_t,
                         int32 top_subsampling_factor);
  };
  struct ComputationHasher {
    inline size_t operator() (const ComputationStructure &s) const {
      return size_t(s.num_sequences) +
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
                               be added to, if it is not NULL.  This function
                               applies the scale opts_.unadapted_backprop_weight
                               after adding this derivative to it.  (The scale
                               opts_.unadapted_backprop_scale is implicitly
                               included already as we already scaled the objf
                               derivatives).
      @return  Returns true if the training went through successfully
            (it should very rarely return false, e.g. if a NaN was generated).
  */
  bool TrainUnadapted(const CuMatrixBase<BaseFloat> &input,
                      const NnetComputation &computation,
                      const chain::Supervision &supervision,
                      const CuVectorBase<BaseFloat> &deriv_weights,
                      Posterior *posterior,
                      CuMatrixBase<BaseFloat> *input_deriv);

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

  /**
     Does the adapted pass of training.
         @param [in] input   The adapted input features.
         @param [in] computation  The adapted version of the
                     computation (this one uses the outputs
                     "output" and "output-xent" instead of
                     "output-si" and "output-si-xent".
         @param [in] supervision  The chain supervision
                     object, containing information derived
                     from the numerator lattices.
         @param [in] deriv_weights  Weights to be applied to the derivatives for the
                     corresponding frames of the output (order is:
                     first frame for all sequences; second frame for
                     all sequences, etc.).  May be stored with the
                     egs.  If this is the empty vector or
                     --apply-deriv-weights=false, they won't be
                     appplied.
         @param [in,out] input_deriv  If non-NULL, the
                     feature derivative w.r.t. the [speaker-adapted] input
                     features will be *added* to this location.
         @return
   */
  bool TrainAdapted(const CuMatrixBase<BaseFloat> &input,
                    const NnetComputation &computation,
                    const chain::Supervision &supervision,
                    const CuVectorBase<BaseFloat> &deriv_weights,
                    CuMatrixBase<BaseFloat> *input_deriv);


  void ProcessOutputs(const NnetChainExample &eg,
                      NnetComputer *computer);

  std::string lang_name_;

  const NnetChainaTrainingOptions &opts_;
  chain::DenominatorGraph den_graph_;
  const differentiable_transform::DifferentiableTransform &transform_;
  // This is a pointer to a compiler owned outside this class (we had to
  // implement it like this to enable computation caching to work with a single
  // option).
  CachingOptimizingCompiler *compiler_;


  Nnet *nnet_;
  Nnet *delta_nnet_;  // Only used if momentum != 0.0 or max-param-change !=
                      // 0.0.  nnet representing accumulated parameter-change
                      // (we'd call this gradient_nnet_, but due to
                      // natural-gradient update, it's better to consider it as
                      // a delta-parameter nnet.


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

  // stats for max-change (for speaker-independent model).
  std::vector<int32> num_max_change_per_component_applied_si_;
  int32 num_max_change_global_applied_si_;
  // stats for max-change (for speaker-dependent model).
  std::vector<int32> num_max_change_per_component_applied_;
  int32 num_max_change_global_applied_;
};



/**
   This object, which has a similar function to NnetChainTrainer, takes care of
   evaluating and possibly training the 'bottom' model.
*/
class NnetChainaBottomTrainer {
 public:
  /**
     Constructor.
      @param [in] nnet_config   Options class
      @param [in] train_bottom_model   True if we are training the 'bottom' model
                            (otherwise this class just does the computation without
                            any backprop).
      @param [in] bottom_subsampling_factor   The factor by which we subsample
                            frames at the output of the 'bottom' nnet.  E.g. if
                            this is 3, then the output frames in each sequence
                            would be numbered t=0, t=3, and so on.
      @param [in,out]  nnet   The neural net we are training.  Expected (for now)
                            to have an input called 'input' (corresponding to
                            the original input features and an output called
                            'output' (corresponding to the embeddings).
   */
  NnetChainaBottomTrainer(const NnetTrainerOptions &nnet_config,
                          int32 bottom_subsampling_factor,
                          bool train_bottom_model,
                          CachingOptimizingCompiler *compiler,
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
          @param [in] input  The input features, most likely raw MFCC or filterbank
                     features.   A pointer, since it is consumed destructively
                     (via 'swap').
          @param [out] output   The output will be written to here.
          @return   Returns the NnetComputer object that we did the computation with;
                    the user should either pass this into Backward(), or delete it.
  */
  NnetComputer* Forward(int32 num_sequences,
                        int32 first_input_t,
                        int32 first_output_t,
                        int32 frames_per_sequence_out,
                        CuMatrix<BaseFloat> *input,
                        CuMatrix<BaseFloat> *output);


  /**
      Does the backward pass, which will do model training.  This will only be
      called if the bottom nnet needs to be trained (otherwise the caller will
      delete the 'computer' object.
         @param [in] computer   The computer object returned from the
                    forward pass.  This function takes ownership of it and
                    will delete it when done with it.
         @param [in] output_deriv  The derivative w.r.t. the output of
                    the forward pass.  It is consumed destructively
                    by this function.

   */
  void Backward(NnetComputer *computer,
                CuMatrix<BaseFloat> *output_deriv);


  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  // Prints out the max-change stats (if nonzero): the percentage of time that
  // per-component max-change and global max-change were enforced.
  void PrintMaxChangeStats() const;

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

  /**
     Does the adapted pass of training.
         @param [in] input   The adapted input features.
         @param [in] computation  The adapted version of the
                     computation (this one uses the outputs
                     "output" and "output-xent" instead of
                     "output-si" and "output-si-xent".
         @param [in] supervision  The chain supervision
                     object, containing information derived
                     from the numerator lattices.
         @param [in,out] input_deriv  If non-NULL, the
                     feature derivative w.r.t. the [speaker-adapted] input
                     features will be *added* to this location.
   */
  void TrainAdapted(const CuMatrixBase<BaseFloat> &input,
                    const NnetComputation &computation,
                    const chain::Supervision &supervision,
                    const VectorBase<BaseFloat> &deriv_weights,
                    CuMatrixBase<BaseFloat> *input_deriv);


  void ProcessOutputs(const NnetChainExample &eg,
                      NnetComputer *computer);

  std::string lang_name_;

  const NnetChainaTrainingOptions opts_;
  bool train_top_model_;
  chain::DenominatorGraph den_graph_;
  const differentiable_transform::DifferentiableTransform &transform_;

  Nnet *nnet_;
  Nnet *delta_nnet_;  // Only used if momentum != 0.0 or max-param-change !=
                      // 0.0.  nnet representing accumulated parameter-change
                      // (we'd call this gradient_nnet_, but due to
                      // natural-gradient update, it's better to consider it as
                      // a delta-parameter nnet.

  // This is a pointer to a compiler owned outside this class (we had to
  // implement it like this to enable computation caching to work with a single
  // option).
  CachingOptimizingCompiler *compiler_;

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

  // stats for max-change (for speaker-independent model).
  std::vector<int32> num_max_change_per_component_applied_si_;
  int32 num_max_change_global_applied_si_;
  // stats for max-change (for speaker-dependent model).
  std::vector<int32> num_max_change_per_component_applied_;
  int32 num_max_change_global_applied_;
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

  // train on one minibatch.
  void Train(const NnetChainExample &eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  // Prints out the max-change stats (if nonzero): the percentage of time that
  // per-component max-change and global max-change were enforced.
  void PrintMaxChangeStats() const;

  ~NnetChainaTrainer();
 private:


  const NnetChainaTrainingOptions &opts_;
  NnetChainaModels *models_;
  // This 'compiler' object is shared by bottom_trainer and the objects
  // stores in top_trainers_.  Storing it here is helpful to simplify writing and
  // reading of computation caches.
  CachingOptimizingCompiler compiler_;

  NnetChainaBottomTrainer *bottom_trainer_;
  // map from language name (e.g. "default", "english", "french") to
  // the object that trains the corresponding 'top' nnet.
  std::unordered_map<std::string, NnetChainaTopTrainer*,
                     StringHasher> top_trainers_;
};


} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_CHAINA_TRAINING_H_
