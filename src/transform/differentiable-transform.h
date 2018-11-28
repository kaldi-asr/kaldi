// transform/differentiable-transform.h

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


#ifndef KALDI_TRANSFORM_DIFFERENTIABLE_TRANSFORM_H_
#define KALDI_TRANSFORM_DIFFERENTIABLE_TRANSFORM_H_

#include <vector>

#include "base/kaldi-common.h"
#include "util/kaldi-table.h"
#include "util/kaldi-holder.h"

namespace kaldi {


namespace differentiable_transform {

class MinibatchInfoItf {
 public:

  virtual ~MinibatchInfoItf() { }
};


class SpeakerStatsItf {

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
   each class (i.e. the mean and variance), which also depends on the features.
   You can view this as an extension of things like BatchNorm, except the
   interface is more complicated because there is a dependence on the per-frame
   class labels.

   The class labels we'll use here will probably be derived from some kind of
   minimal tree, with hundreds instead of thousands of states.  Part of the
   reason for using a smaller number of states is that, to make the thing
   properly differentiable during training, we need to use a small enough number
   of states that we can obtain a reasonable estimate for the mean and variance
   of a Gaussian for each one in training time.   Anyway, see
   http://isl.anthropomatik.kit.edu/pdf/Nguyen2017.pdf, it's generally better
   for this kind of thing to use "simple target models" for adaptation.

   Note: for training utterances we'll generally get the class labels used for
   adatpation in a supervised manner, either by aligning a previous system like
   a GMM system, or from the (soft) posteriors of the the numerator graphs.  In
   test time, we'll usually be getting these class labels from some kind of
   unsupervised process.

   Because we tend to train neural nets on fairly small fixed-size chunks
   (e.g. 1.5 seconds), and transforms like fMLLR don't tend to work very well
   until you have about 5 seconds of data, we will usually be arranging those
   chunks into groups where all members of the group comes from the same
   speaker.
 */
class DifferentiableTransform {
 public:

  /// Return the dimension of the input and output features.
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
              per speaker.  Caution: the order of both the input and
              output features, and the posteriors, does not consist of blocks,
              one per sequence, but rather blocks, one per time frame, so the
              sequences are intercalated.
     @param [in] num_chunks   The number of individual sequences
              (e.g., chunks of speech) represented in 'input'.
              input.NumRows() will equal num_sequences times the number
              of time frames.
     @param [in] num_spk  The number of speakers.  Must be greater than one, and
             must divide num_chunks.  The number of chunks per speaker
             (num_chunks / num_spk) must be the same for all speakers, and the
             chunks for a speaker must be consecutive.
     @param [in] posteriors  (note: this is a vector of vector of
             pair<int32,BaseFloat>).  This provides, in 'soft-count'
             form, the class supervision information that is used for the
             adaptation.  posteriors.size() will be equal to input.NumRows(),
             and the ordering of its elements is the same as the ordering
             of the rows of input, i.e. the sequences are intercalated.
             There is no assumption that the posteriors sum to one;
             this allows you to do things like silence weighting.
     @param [out] output  The adapted output.  This matrix should have the
            same dimensions as 'input'.
     @return  This function returns either NULL or an object of type
             DifferentiableTransformItf*, which is expected to be given
             to the function TrainingBackward().  It will store
             any information that will be needed in the backprop phase.
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
     @param [in] minibatch_info  The object returned by the corresponding
                      call to TrainingForward().  The caller
                      will likely want to delete that object after
                      calling this function
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
      const MinibatchInfoItf &minibatch_info,
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

  // To be called after repeated alls to Accumulate(), does any estimation that
  // is required in training time (normally per-speaker means and possibly
  // variances.
  virtual void Estimate(int32 final_iter) = 0;

  // Returns an object representing sufficient statistics for estimating a
  // speaker-dependent transform.  This object will initially have zero
  // counts in its statistics.  It will represent the stats for a single
  // speaker.
  virtual SpeakerStatsItf *GetEmptySpeakerStats() = 0;


  // Accumulate statistics for a segment of test data, storing them in the
  // object 'speaker_stats'.  There is no assumption that the soft-counts in
  // 'posteriors' are positive; this allows you to change your mind about the
  // traceback, in test-time, by subtracting the stats that you no longer want
  // to use.
  virtual void TestingAccumulate(
      const MatrixBase<BaseFloat> &input,
      const Posterior &posteriors,
      SpeakerStatsItf *speaker_stats) const = 0;

  // Applies the transformation implied by the statistics in 'speaker_stats' to
  // 'input', storing in the result in 'output'.  It will do any estimation
  // procedure that is required first, if applicable.
  virtual void TestingForward(
      const MatrixBase<BaseFloat> &input,
      const SpeakerStatsItf &speaker_stats,
      MatrixBase<BaseFloat> *output) const = 0;


  // Read transform from stream (works out its type).  Dies on error.
  static DifferentiableTransform* ReadNew(std::istream &is, bool binary);

  // Copies transform (deep copy).
  virtual DifferentiableTransform* Copy() const = 0;

  // Returns a new transform of the given type e.g. "MeanNormalize",
  // or NULL if no such component type exists.
  static DifferentiableTransform *NewTransformOfType(const std::string &type);

  // Write transform to stream
  virtual void Write(std::ostream &os, bool binary) const = 0;

  // Reads transform from stream (normally you would previously have created
  // the transform object of the correct type using ReadNew().
  virtual void Read(std::istream &is, bool binary) = 0;

 protected:
  int32 num_classes_;


};


/**
   This is a version of the transform class that does nothing.  It's potentially
   useful for situations where you want to apply speaker normalization to some
   dimensions of the feature vector but not to others.
 */
class NoOpTransform: public DifferentiableTransform {
 public:

  int32 Dim() const override { return dim_; }
  int32 NumClasses() const override { return num_classes_; }
  MinibatchInfoItf* TrainingForward(
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      CuMatrixBase<BaseFloat> *output) const override {
    output->CopyFromMat(input);
    return NULL;
  }
  virtual void TrainingBackward(
      const CuMatrixBase<BaseFloat> &input,
      const CuMatrixBase<BaseFloat> &output_deriv,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      const MinibatchInfoItf &minibatch_info,
      CuMatrixBase<BaseFloat> *input_deriv) const override {
    input_deriv->AddMat(1.0, output_deriv);
  }

  virtual int32 NumFinalIterations() { return 0; }

  void Accumulate(
      int32 final_iter,
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors) override { }



  SpeakerStatsItf *GetEmptySpeakerStats() override { return NULL; }

  void TestingAccumulate(
      const MatrixBase<BaseFloat> &input,
      const Posterior &posteriors,
      SpeakerStatsItf *speaker_stats) const override { }
  void TestingForward(
      const MatrixBase<BaseFloat> &input,
      const SpeakerStatsItf &speaker_stats,
      MatrixBase<BaseFloat> *output) override {
    output->CopyFromMat(input);
  }

  void Estimate(int32 final_iter) override { }

  NoOpTransform(const NoOpTransform &other):
      dim_(other.dim_), num_classes_(other.num_classes_) { }

  DifferentiableTransform* Copy() const override {
    return new NoOpTransform(*this);
  }

  void Write(std::ostream &os, bool binary) const override;

  void Read(std::istream &is, bool binary) override;

 private:
  int32 dim_;
  int32 num_classes_;
};


/**
   This is a version of the transform class that does a sequence of other
   transforms, specified by other instances of the DifferentiableTransform
   interface.
 */
class SequenceTransform: public DifferentiableTransform {
 public:

  int32 Dim() const override;
  int32 SetNumClasses() const override;

  MinibatchInfoItf* TrainingForward(
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      CuMatrixBase<BaseFloat> *output) const override;
  virtual void TrainingBackward(
      const CuMatrixBase<BaseFloat> &input,
      const CuMatrixBase<BaseFloat> &output_deriv,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      const MinibatchInfoItf &minibatch_info,
      CuMatrixBase<BaseFloat> *input_deriv) const override;

  virtual int32 NumFinalIterations();

  void Accumulate(
      int32 final_iter,
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors) override;

  SpeakerStatsItf *GetEmptySpeakerStats() override;

  void TestingAccumulate(
      const MatrixBase<BaseFloat> &input,
      const Posterior &posteriors,
      SpeakerStatsItf *speaker_stats) const override;

  virtual void TestingForward(
      const MatrixBase<BaseFloat> &input,
      const SpeakerStatsItf &speaker_stats,
      MatrixBase<BaseFloat> *output) override;

  void Estimate(int32 final_iter) override;

  SequenceTransform(const SequenceTransform &other);

  DifferentiableTransform* Copy() const override {
    return new SequenceTransform(*this);
  }

  void Write(std::ostream &os, bool binary) const override;

  void Read(std::istream &is, bool binary) override;

 private:
  std::vector<DifferentiableTransform*> transforms_;
};


/**
   This is a version of the transform class that consists of a number of
   other transforms, appended dimension-wise-- e.g. this could be used to
   implement block-diagonal fMLLR, or a structure where some dimensions are
   adapted and some are not.
 */
class AppendTransform: public DifferentiableTransform {
 public:

  int32 Dim() const override;
  int32 SetNumClasses() const override;

  MinibatchInfoItf* TrainingForward(
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      CuMatrixBase<BaseFloat> *output) const override;
  virtual void TrainingBackward(
      const CuMatrixBase<BaseFloat> &input,
      const CuMatrixBase<BaseFloat> &output_deriv,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      const MinibatchInfoItf &minibatch_info,
      CuMatrixBase<BaseFloat> *input_deriv) const override;

  virtual int32 NumFinalIterations();

  void Accumulate(
      int32 final_iter,
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors) override;

  virtual void TestingForward(
      const MatrixBase<BaseFloat> &input,
      const SpeakerStatsItf &speaker_stats,
      MatrixBase<BaseFloat> *output) override;

  void Estimate(int32 final_iter) override;

  AppendTransform(const AppendTransform &other);

  DifferentiableTransform* Copy() const override {
    return new AppendTransform(*this);
  }

  void Write(std::ostream &os, bool binary) const override;

  void Read(std::istream &is, bool binary) override;

 private:
  std::vector<DifferentiableTransform*> transforms_;
};



/**
   This is a version of the transform class that appends over sub-ranges
   of dimensions, so that, for instance, you can implement a block-diagonal
   transform or a setup where some dimensions are transformed and some are
   not.
*/
class AppendTransform: public DifferentiableTransform {
  int32 Dim() const override;
  int32 NumClasses() const override;
  MinibatchInfoItf* TrainingForward(
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      CuMatrixBase<BaseFloat> *output) const override {
    output->CopyFromMat(input);
    return NULL;
  }
  virtual void TrainingBackward(
      const CuMatrixBase<BaseFloat> &input,
      const CuMatrixBase<BaseFloat> &output_deriv,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      const MinibatchInfoItf &minibatch_info,
      CuMatrixBase<BaseFloat> *input_deriv) const override;

  void Accumulate(
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors) override;

  void Estimate() override { }

  AppendTransform(const AppendTransform &other);

  DifferentiableTransform* Copy() const override;

  void Write(std::ostream &os, bool binary) const override;

  void Read(std::istream &is, bool binary) override;
 private:
  std::vector<DifferentiableTransform*> transforms_;
};


/**
   This version of the transform class does a mean normalization: adding an
   offset to its input so that the difference (per speaker) of the transformed
   class means from the speaker-independent class means is minimized.
   This is like a mean-only fMLLR with fixed (say, unit) covariance model.
 */
class SimpleMeanTransform: public DifferentiableTransform {
 public:
  int32 Dim() const override;
  int32 NumClasses() const override;
  MinibatchInfoItf* TrainingForward(
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      CuMatrixBase<BaseFloat> *output) const override {
    output->CopyFromMat(input);
    return NULL;
  }
  virtual void TrainingBackward(
      const CuMatrixBase<BaseFloat> &input,
      const CuMatrixBase<BaseFloat> &output_deriv,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      const MinibatchInfoItf &minibatch_info,
      CuMatrixBase<BaseFloat> *input_deriv) const override;

  void Accumulate(
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors) override;

  virtual void TestingForward(
      const MatrixBase<BaseFloat> &input,
      const SpeakerStatsItf &speaker_stats,
      MatrixBase<BaseFloat> *output) override;


  void Estimate() override { }

  AppendTransform(const AppendTransform &other);

  DifferentiableTransform* Copy() const override;

  void Write(std::ostream &os, bool binary) const override;

  void Read(std::istream &is, bool binary) override;
 private:

  // OK: how to compute stats
  class MinibatchInfo: public MinibatchInfoItf {

    // Stores the total weights, per frame, that correspond to the Posteriors
    // supplied to TrainingForward().
    CuVector<BaseFloat> frame_weights;

    // The total of frame_weights.
    BaseFloat total_weight;
  };

  // dim_ is the feature dimension
  int32 dim_;

  // The class-dependent means.  Dimension is num_classes_ by dim_.
  // Note: these will not be set up during training, they will only
  // be set up after calling Accumulate() and Estimate(), which happens
  // in test time.
  CuMatrix<BaseFloat> means_;

  // mean_stats_ and count_ are used in Accumulate() to accumulate
  // statistics to adapt the mean.
  CuMatrix<double> mean_stats_;
  double count_;

};


/**
   Notes on the math behind differentiable fMLLR transform.

 */

class FmllrTransform: public DifferentiableTransform {
 public:
  int32 Dim() const override;
  int32 NumClasses() const override;
  MinibatchInfoItf* TrainingForward(
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      CuMatrixBase<BaseFloat> *output) const override;
  virtual void TrainingBackward(
      const CuMatrixBase<BaseFloat> &input,
      const CuMatrixBase<BaseFloat> &output_deriv,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors,
      const MinibatchInfoItf &minibatch_info,
      CuMatrixBase<BaseFloat> *input_deriv) const override;
  void Accumulate(
      int32 final_iter,
      const CuMatrixBase<BaseFloat> &input,
      int32 num_chunks,
      int32 num_spk,
      const Posterior &posteriors) override;

  SpeakerStatsItf *GetEmptySpeakerStats() override;

  void TestingAccumulate(
      const MatrixBase<BaseFloat> &input,
      const Posterior &posteriors,
      SpeakerStatsItf *speaker_stats) const override;

  virtual void TestingForward(
      const MatrixBase<BaseFloat> &input,
      const SpeakerStatsItf &speaker_stats,
      MatrixBase<BaseFloat> *output) override;

  void Estimate(int32 final_iter) override { }

  FmllrTransform(const FmllrTransform &other);

  DifferentiableTransform* Copy() const override;

  void Write(std::ostream &os, bool binary) const override;

  void Read(std::istream &is, bool binary) override;
 private:

  // OK: how to compute stats
  class MinibatchInfo: public MinibatchInfoItf {

    // Stores the total weights, per frame, that correspond to the Posteriors
    // supplied to TrainingForward().  frame_weights.Dim() equals
    // input.NumRows().
    CuVector<BaseFloat> frame_weights;

    // The total of frame_weights per speaker.
    CuVector<BaseFloat> frame_weights;

    BaseFloat total_weight;
  };

  class SpeakerStats: public SpeakerStatsItf {

  };

  // dim_ is the feature dimension
  int32 dim_;

  // The class-dependent means.  Dimension is num_classes_ by dim_.
  // Note: these will not be set up during training, they will only
  // be set up after calling Accumulate() and Estimate(), which happens
  // in test time.
  CuMatrix<BaseFloat> means_;

  // mean_stats_ and count_ are used in Accumulate() to accumulate
  // statistics to adapt the mean.
  CuMatrix<double> mean_stats_;
  double count_;

};


} // namespace differentiable_transform
} // namespace kaldi

#endif  // KALDI_TRANSFORM_DIFFERENTIABLE_TRANSFORM_H_
