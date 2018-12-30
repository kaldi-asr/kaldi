// adapt/differentiable-transform-itf.h

// Copyright      2018  Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_TRANSFORM_DIFFERENTIABLE_TRANSFORM_ITF_H_
#define KALDI_TRANSFORM_DIFFERENTIABLE_TRANSFORM_ITF_H_

#include <vector>
#include "base/kaldi-common.h"
#include "matrix/kaldi-matrix.h"
#include "cudamatrix/cu-matrix.h"
#include "util/text-utils.h"
#include "hmm/posterior.h"


namespace kaldi {
namespace differentiable_transform {

class MinibatchInfoItf {
 public:
  virtual ~MinibatchInfoItf() { }
};


class SpeakerStatsItf {
 public:
  // Does any estimation that is required-- you call this after accumulating
  // stats and before calling TestingForward().  You'll normally want to
  // override this, unless your object requires no estimation.
  virtual void Estimate() { }

  virtual ~SpeakerStatsItf() { }
};



/**
   This class is for speaker-dependent feature-space transformations --
   principally various varieties of fMLLR, including mean-only, diagonal and
   block-diagonal versions -- which are intended for placement in the bottleneck
   of a neural net.  So code-wise, we'd have: bottom neural net, then transform,
   then top neural net.  The transform is designed to be differentiable, i.e. it
   can be used during training to propagate derivatives from the top neural net
   down to the bottom neural net.  The reason this is non-trivial (i.e. why it's
   not just a matrix multiplication) is that the value of the transform itself
   depends on the features, and also on the speaker-independent statistics for
   each class (i.e. the mean and variance), which also depend on the features
   sicne we estimate them from the same minibatch.
   You can view this as an extension of things like BatchNorm, except the
   interface is more complicated because there is a dependence on the per-frame
   class labels.

   The class labels we'll use here will probably be derived from some kind of
   minimal tree, with hundreds instead of thousands of states.  Part of the
   reason for using a smaller number of states is that, to make the thing
   properly differentiable during training, we need to use a small enough number
   of states that we can obtain a reasonable estimate for the mean and (spherical)
   variance of a Gaussian for each one in training time.   Anyway, as you can see in
   http://isl.anthropomatik.kit.edu/pdf/Nguyen2017.pdf, it's generally better
   for this kind of thing to use "simple target models" for adaptation rather than
   very complex models.

   Note: for training utterances we'll generally get the class labels used for
   adatpation in a supervised manner, either by aligning a previous system like
   a GMM system, or-- more likely-- from the (soft) posteriors of the the
   numerator graphs.  In test time, we'll usually be getting these class labels
   from some kind of unsupervised process.

   Because we tend to train neural nets on fairly small fixed-size chunks
   (e.g. 1.5 seconds), and transforms like fMLLR don't tend to work very well
   until you have about 5 seconds of data, we will usually be arranging those
   chunks into groups where all members of the group come from the same
   speaker.  So, for instance, instead of 128 totally separate chunks, we might
   have 4 chunks per speaker and 32 speakers.

   The basic pattern of usage of class DifferentiableTransform is this:

     - Initialize the object prior to training, e.g. with InitFromConfig().

     - Use this object to jointly train the 'bottom' (feature-extracting) and
       'top' (ASR) network.  This involves functions TrainingForward() and
       TrainingBackward() of this object; the posteriors used for that might be
       dumped with the 'egs' (e.g. come from a GMM system), or might be derived
       from the alignment of the numerator lattices in chain training.  Any
       class means that must be estimated, would be estimated on each minibatch
       (we'll try to keep the minibatches as large as possible, and may use
       tricks like using bigger minibatch sizes for the bottom
       (feature-extracting) network and smaller ones for the top one, to save
       memory.  At this stage, this object will most likely only contain
       configuration information and not any kind of data-dependent statistics.

     - Use some reasonable-sized subset of training data to accumulate more
       reliable statistics for the target model using Accumulate() followed
       by Estimate().  If NumFinalIterations() is more than one you may need
       do this in a short loop.

     - In test time, for each speaker you'll:
       - call GetEmptySpeakerStats() to get an object to store adaptation statistics
         for your speaker.
       - Obtain some class-level posteriors somehow (could come from an initial
         decoding pass on all the data, or from the final decoding pass on the
         part of the data you've seen up till now).  Use these to call
         TestingAccumulate() to accumulate speaker stats.
       - Call TestingForward() with the speaker-stats object to get
         adapted features.


 */
class DifferentiableTransform {
 public:

  /// Return the dimension of the features this operates on.
  virtual int32 Dim() const = 0;

  /// Return the number of classes in the model used for adaptation.  These
  /// will probably correspond to the leaves of a small tree, so they would
  /// be pdf-ids.  This model only keeps track of the number of classes,
  /// it does not contain any information about what they mean.  The
  /// integers in the objects of type Posterior provided to this class
  /// are expected to contain numbers from 0 to NumClasses() - 1.
  int32 NumClasses() const { return num_classes_; }


  /// This can be used to change the number of classes.  It would normally be
  /// used, if at all, after the model is trained and prior to calling
  /// Accumulate(), in case you want to use a more detailed model (e.g. the
  /// normal-size tree instead of the small one that we use during training).
  /// Child classes may want to override this, in case they need to do
  /// something more than just set this variable.
  virtual void SetNumClasses(int32 num_classes) { num_classes_ = num_classes; }

  /**
     This is the function you call in training time, for the forward
     pass; it adapts the features.  By "training time" here, we
     assume you are training the 'bottom' neural net, that produces
     the features in 'input'; if you were not training it, it would
     be the same as test time as far as this function is concerned.

     @param [in] input  The original, un-adapted features; these
              will typically be output by a neural net, the 'bottom' net in our
              terminology.  This will correspond to a whole minibatch,
              consisting of multiple speakers and multiple sequences (chunks)
              per speaker.  Caution: in the input and
              output features, and the posteriors, the 't' has the larger
              stride than the minibatch-index 'n', so the order is:
              first frame of all sequences; then the second frame of
              all sequences; and so on.  This is the default order in
              nnet3; see operator < of nnet3::Index.
     @param [in] num_chunks   The number of individual sequences
              (e.g., chunks of speech) represented in 'input'.
              input.NumRows() will equal num_sequences times the number
              of time frames.
     @param [in] num_spk  The number of speakers.  Must be greater than one, and
             must divide num_chunks.  The number of chunks per speaker
             must be the same for all speakers (it will equal num_chunks / num_spk),
             and the chunks for a speaker must be consecutively numbered.
     @param [in] posteriors  (note: this is a vector of vector of
             pair<int32,BaseFloat>).  This provides, in 'soft-count'
             form, the class supervision information that is used for the
             adaptation.  posteriors.size() will be equal to input.NumRows(),
             and the ordering of its elements is the same as the ordering
             of the rows of input (i.e. the 't' has the larger stride).
             There is no assumption that the posteriors sum to one;
             this allows you to do things like silence weighting.  But
             the posteriors are expected to be nonnegative.
     @param [out] output  The adapted output.  This matrix should have the
             same dimensions as 'input'.  It does not have to be free of
             NaNs when you call this function.
     @return  This function returns either NULL or an object of type
             DifferentiableTransform*, which is expected to later be given
             to the function TrainingBackward().  It will store
             any information that needs to be remembered for the backward
             phase.
   */
  virtual MinibatchInfoItf* TrainingForward(
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      CuMatrixBase<BaseFloat> *output) const = 0;

  /**
     This does the backpropagation, during the training pass.

     @param [in] input   The original input (pre-transform) features that
                       were given to TrainingForward().
     @param [in] output_deriv  The derivative of the objective function
                       (that we are backpropagating) w.r.t. the output.
     @param [in] num_chunks,num_spk,posteriors
                       See TrainingForward() for information
                       about these arguments; they should be the same
                       values.
     @param [in] minibatch_info  The pointer returned by the corresponding
                      call to TrainingForward() (may be NULL).  This function
                      takes ownership of the pointer.  If for some reason the
                      backward pass was not done, the caller will likely
                      want to delete it themselves.
     @param [in,out] input_deriv  The derivative at the input, i.e.
                      dF/d(input), where F is the function we are
                      evaluating.  Must have the same dimension as
                      'input'.  The derivative is *added* to here.
                      This is useful because generally we will also
                      be training (perhaps with less weight) on
                      the unadapted features, in order to prevent them
                      from deviating too far from the adapted ones
                      and to allow the same model to be used for the
                      first pass.
   */
  virtual void TrainingBackward(
      const CuMatrixBase<BaseFloat> &input,
      const CuMatrixBase<BaseFloat> &output_deriv,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      MinibatchInfoItf *minibatch_info,
      CuMatrixBase<BaseFloat> *input_deriv) const = 0;


  /**
     Returns the number of times you have to (call Accumulate() on a subset
     of data, then call Estimate())
   */
  virtual int32 NumFinalIterations() = 0;

  /**
     This will typically be called sequentially, minibatch by minibatch,
     for a subset of training data, after training the neural nets,
     followed by a call to Estimate().  Accumulate() stores statistics
     that are used by Estimate().  This process is analogous to
     computing the final stats in BatchNorm, in preparation for testing.
     In practice it will be doing things like computing per-class means
     and variances.

        @param [in] final_iter  An iteration number in the range
                 [0, NumFinalIterations()].  In many cases there will
                 be only one iteration so this will just be zero.

     The input parameters are the same as the same-named parameters to
     TrainingForward(); please refer to the documentation there.
   */
  virtual void Accumulate(
      int32 final_iter,
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors) = 0;

  // To be called after repeated calls to Accumulate(), does any estimation that
  // is required in training time (normally per-speaker means and possibly
  // variances.
  //      @param [in] final_iter  An iteration number in the range
  //               [0, NumFinalIterations()].  In many cases there will
  //               be only one iteration so this will just be zero.
  virtual void Estimate(int32 final_iter) = 0;

  // Returns an object representing sufficient statistics for estimating a
  // speaker-dependent transform.  This object will initially have zero
  // counts in its statistics.  It will represent the stats for a single
  // speaker.
  virtual SpeakerStatsItf *GetEmptySpeakerStats() const = 0;


  // Accumulate statistics for a segment of test data, storing them in the
  // object 'speaker_stats'.  There is no assumption that the soft-counts in
  // 'posteriors' are positive; this allows you to change your mind about the
  // traceback, in test-time, by subtracting the stats that you no longer want
  // to use.
  virtual void TestingAccumulate(
      const MatrixBase<BaseFloat> &input,
      const SubPosterior &posteriors,
      SpeakerStatsItf *speaker_stats) const = 0;

  // Applies the transformation implied by the statistics in 'speaker_stats' to
  // 'input', storing in the result in 'output'.  You must have done any estimation
  // procedure that is required first, by calling Estimate() on the speaker-stats
  // object.  'output' may contain NaN's at entry.
  virtual void TestingForward(
      const MatrixBase<BaseFloat> &input,
      const SpeakerStatsItf &speaker_stats,
      MatrixBase<BaseFloat> *output) const = 0;

  // TestingForwardBatch() combines GetEmptySpeakerStats(), TestingAccumulate() and
  // TestingForward().  It has a default implementation.   It is a convenience
  // function that may be useful during training under some circumstances, e.g.
  // when you want to train only the top network.
  virtual void TestingForwardBatch(
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      CuMatrixBase<BaseFloat> *output) const;

  // Copies transform (deep copy).
  virtual DifferentiableTransform* Copy() const = 0;

  // Return the type of this transform.  E.g. "NoOpTransform".
  virtual std::string Type() const = 0;

  /*
    Initialize this object from the config line at position 'cur_pos' of the
    vector 'config_lines'.  This function may end up reading more lines than
    one, if this is a transform type that contains other transforms.

        @param [in]     cur_pos  The starting position in config_lines; required
                            to be in the range [0, config_lines->size() - 1].
                            The Type() of this object must match the first token
                            (function FirstToken()) of that ConfigLine.
        @param [in,out] config_lines   Config lines to be read.  It's non-const
                            because the process of reading them has effects on
                            the lines themselves (the ConfigLine object keeps
                            track of which configuration values have been read).
        @return        Returns the next position to be read.  Will be in the range
                       [cur_pos + 1, config_lines->size()]; if it's equal to
                       config_lines->size(), it means we're done.
   */
  virtual int32 InitFromConfig(int32 cur_pos,
                               std::vector<ConfigLine> *config_lines) = 0;

  // Returns a new transform of the given type e.g. "NoOpTransform"
  // or NULL if no such component type exists.  If angle brackets are
  // present, e.g. "<FmllrTransform>", this function will detect and
  // remove them.
  static DifferentiableTransform *NewTransformOfType(const std::string &type);

  // Reads a differentiable transform from a config file (this function parses
  // the file and reads a single DifferentiableTransform object from it).  Note:
  // since DifferentiableTransform objects can contain others, the file may
  // contain many lines.  Throws exception if it did not succeed-- including
  // if the config file had junk at the end that was not parsed.
  static DifferentiableTransform *ReadFromConfig(std::istream &is,
                                                 int32 num_classes);



  // Write transform to stream
  virtual void Write(std::ostream &os, bool binary) const = 0;

  // Reads transform from stream (normally you would previously have created
  // the transform object of the correct type using ReadNew().
  virtual void Read(std::istream &is, bool binary) = 0;

  // Read transform from stream (works out its type).  Dies on error.
  // This will be used when reading in objects that have been written with
  // the Write() function, since you won't know the type of the object
  // beforehand.
  static DifferentiableTransform* ReadNew(std::istream &is, bool binary);

  DifferentiableTransform(): num_classes_(-1) { }

  DifferentiableTransform(const DifferentiableTransform &other):
      num_classes_(other.num_classes_) { }

  virtual ~DifferentiableTransform() { }
 protected:
  int32 num_classes_;
};


/**
   struct DifferentiableTransformMapped is just a holder of an object of type
   DifferentiableTransform and a vector<int32> representing a map from
   pdf-ids to classes.

   This map (if present) will be obtained from the binary build-tree-two-level,
   and will map from tree leaves to a smaller number of classes (e.g. 200), so
   that we can reasonably estimate the class means from a single minibatch
   during training.  The contents of 'pdf_map' should be in the range [0,
   transform->NumClases() - 1].

 */
struct DifferentiableTransformMapped {
  DifferentiableTransform *transform;
  std::vector<int32> pdf_map;

  // This function returns pdf_map.size() if pdf_map is nonempty; otherwise
  // it returns transform->NumClasses().
  int32 NumPdfs() const;

  void Read(std::istream &is, bool binary);

  void Write(std::ostream &os, bool binary) const;

  // Check that the dimensions are consistent, i.e. pdf_map.empty() or
  // transform->NumClasses() == max-element-in-pdf_map + 1.
  void Check() const;

  DifferentiableTransformMapped(): transform(NULL)  {}

  ~DifferentiableTransformMapped() { delete transform; }

};


} // namespace differentiable_transform
} // namespace kaldi

#endif  // KALDI_TRANSFORM_DIFFERENTIABLE_TRANSFORM_H_
