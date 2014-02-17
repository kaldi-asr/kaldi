// dec-wrap/dec-wrap-feat-input.h

// Copyright 2012 Cisco Systems (author: Matthias Paulik)
//           2012-2013  Vassil Panayotov
//           2013 Johns Hopkins University (author: Daniel Povey)
//           2013 Ondrej Platek, UFAL MFF UK

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
#ifndef KALDI_DEC_WRAP_DEC_WRAP_FEAT_INPUT_H_
#define KALDI_DEC_WRAP_DEC_WRAP_FEAT_INPUT_H_

#include "feat/feature-functions.h"
#include "dec-wrap/dec-wrap-audio-source.h"


namespace kaldi {


// Interface specification COPY -> because OnlFeatInputItf
// requires portaudio to be installed through various includes
// It should be easy to remove the portaudio dependancy by splitting
// the code into multiple files.
class OnlFeatInputItf {
 public:
  // Produces feature vectors in some way.
  // The features may be e.g. extracted from an audio samples, received and/or
  // transformed from another OnlFeatInput class etc.
  //
  // "output" - a matrix to store the extracted feature vectors in its rows.
  //            The number of rows (NumRows()) of "output" when the function is
  //            called, is treated as a hint of how many frames the user wants,
  //            but this function does not promise to produce exactly that many:
  //            it may be slightly more, less, or even zero, on a given call.
  virtual MatrixIndexT Compute(Matrix<BaseFloat> *output) = 0;

  virtual int32 Dim() const = 0; // Return the output dimension of these features.

  virtual void Reset() =0;

  virtual ~OnlFeatInputItf() {}

};

/*********************************************************************
 *                    OnlFeatureMatrix                           *
 *********************************************************************/
struct OnlFeatureMatrixOptions {
  int32 batch_size; // number of frames to request each time.
  OnlFeatureMatrixOptions(): batch_size(27) { }
  void Register(OptionsItf *po) {
    po->Register("batch-size", &batch_size,
                 "Number of feature vectors processed without interruption");
  }
};

// The class OnlFeatureMatrix wraps something of type
// OnlFeatInputItf in a manner that is convenient for
// a Decodable type to consume.
class OnlFeatureMatrix {
 public:
  OnlFeatureMatrix(const OnlFeatureMatrixOptions &opts,
                      OnlFeatInputItf *input):
      opts_(opts), input_(input), feat_dim_(input->Dim()),
      feat_loaded_(0) { }

  bool IsValidFrame (int32 frame);

  int32 Dim() const { return feat_dim_; }

  // GetFrame() will die if it's not a valid frame; you have to
  // call IsValidFrame() for this frame, to see whether it
  // is valid.
  SubVector<BaseFloat> GetFrame(int32 frame);

  void Reset();

 private:
  /// Called when we need more features.
  MatrixIndexT GetNextFeatures();

  const OnlFeatureMatrixOptions opts_;
  OnlFeatInputItf *input_;
  int32 feat_dim_;
  Matrix<BaseFloat> feat_matrix_;
  Matrix<BaseFloat> feat_matrix_old_;
  int32 feat_loaded_; // the # of features totally loaded
};

/*********************************************************************
 *                          OnlFeInput                           *
 *********************************************************************/
// Implementation, that is meant to be used to read samples from an
// OnlAudioSource and to extract MFCC/PLP features in the usual way
template <class E>
class OnlFeInput : public OnlFeatInputItf {
 public:
  // "au_src" - OnlAudioSourceItf object
  // "fe" - object implementing MFCC/PLP feature extraction
  // "frame_size" - frame extraction window size in audio samples
  // "frame_shift" - feature frame width in audio samples
  OnlFeInput(OnlAudioSourceItf *au_src, E *fe,
                const int32 frame_size, const int32 frame_shift);

  virtual int32 Dim() const { return extractor_->Dim(); }

  virtual MatrixIndexT Compute(Matrix<BaseFloat> *output);

  virtual void Reset();

 private:
  OnlAudioSourceItf *source_; // audio source
  E *extractor_; // the actual feature extractor used
  const int32 frame_size_;
  const int32 frame_shift_;
  Vector<BaseFloat> wave_remainder_; // the samples remained from the previous
                                     // feature batch

  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlFeInput);
};

template<class E>
OnlFeInput<E>::OnlFeInput(OnlAudioSourceItf *au_src, E *fe,
                                   int32 frame_size, int32 frame_shift)
    : source_(au_src), extractor_(fe),
      frame_size_(frame_size), frame_shift_(frame_shift) {}

template<class E> MatrixIndexT
OnlFeInput<E>::Compute(Matrix<BaseFloat> *output) {
  MatrixIndexT nvec = output->NumRows(); // the number of output vectors
  if (nvec <= 0) {
    KALDI_WARN << "No feature vectors requested?!";
    return 0;
  }

  // Prepare the input audio samples
  int32 samples_req = frame_size_ + (nvec - 1) * frame_shift_;
  Vector<BaseFloat> read_samples(samples_req);

  MatrixIndexT read = source_->Read(&read_samples);

  if (read == 0) {
    KALDI_VLOG(4) << "Read nothing from audio source";
    return 0;
  }

  SubVector<BaseFloat> actually_read(read_samples, 0, read);

  Vector<BaseFloat> all_samples(wave_remainder_.Dim() + actually_read.Dim());
  all_samples.Range(0, wave_remainder_.Dim()).CopyFromVec(wave_remainder_);
  all_samples.Range(wave_remainder_.Dim(), actually_read.Dim()).CopyFromVec(actually_read);

  // Extract the features
  if (all_samples.Dim() < frame_size_) {
    wave_remainder_ = all_samples;
    KALDI_VLOG(2) << "Read some " << read
                  << " but not enough for " << read_samples.Dim();
    return 0;
  } else {
    BaseFloat vtln_warp_local = 1.0;
    extractor_->Compute(all_samples, vtln_warp_local, output, &wave_remainder_);
    KALDI_VLOG(2) << "Read all data " << read
                  << ", still reamining " << wave_remainder_.Dim();
    return output->NumRows();
  }
}

template<class E>
void OnlFeInput<E>::Reset() {
  wave_remainder_.Resize(0);
}


/*********************************************************************
 *                          OnlLdaInput                          *
 *********************************************************************/
// Splices the input features and applies a transformation matrix.
// Note: the transformation matrix will usually be a linear transformation
// [output-dim x input-dim] but we accept an affine transformation too.
class OnlLdaInput: public OnlFeatInputItf {
 public:
  OnlLdaInput(OnlFeatInputItf *input,
                 const Matrix<BaseFloat> &transform,
                 int32 left_context,
                 int32 right_context);

  virtual MatrixIndexT Compute(Matrix<BaseFloat> *output);

  virtual int32 Dim() const { return linear_transform_.NumRows(); }

  virtual void Reset();

 private:
  // The static function SpliceFeats splices together the features and
  // puts them together in a matrix, so that each row of "output" contains
  // a contiguous window of size "context_window" of input frames.  The dimension
  // of "output" will be feats.NumRows() - context_window + 1 by
  // feats.NumCols() * context_window.  The input features are
  // treated as if the frames of input1, input2 and input3 have been appended
  // together before applying the main operation.
  static void SpliceFrames(const MatrixBase<BaseFloat> &input1,
                           const MatrixBase<BaseFloat> &input2,
                           const MatrixBase<BaseFloat> &input3,
                           int32 context_window,
                           Matrix<BaseFloat> *output);

  void TransformToOutput(const MatrixBase<BaseFloat> &spliced_feats,
                         Matrix<BaseFloat> *output);
  void ComputeNextRemainder(const MatrixBase<BaseFloat> &input);

  OnlFeatInputItf *input_; // underlying/inferior input object
  const int32 input_dim_; // dimension of the feature vectors before xform
  const int32 left_context_;
  const int32 right_context_;
  Matrix<BaseFloat> linear_transform_; // transform matrix (linear part only)
  Vector<BaseFloat> offset_; // Offset, if present; else empty.
  Matrix<BaseFloat> remainder_; // The last few frames of the input, that may
  // be needed for context purposes.

  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlLdaInput);
};

/*********************************************************************
 *                         OnlDeltaInput                         *
 *********************************************************************/
// Does the time-derivative computation (e.g., adding deltas and delta-deltas).
// This is standard in more "old-fashioned" feature extraction.  Like an online
// version of the function ComputeDeltas in feat/feature-functions.h, where the
// struct DeltaFeaturesOptions is also defined.
class OnlDeltaInput: public OnlFeatInputItf {
 public:
  OnlDeltaInput(const DeltaFeaturesOptions &delta_opts,
                   OnlFeatInputItf *input);

  virtual MatrixIndexT Compute(Matrix<BaseFloat> *output);

  virtual int32 Dim() const { return input_dim_ * (opts_.order + 1); }

  virtual void Reset();

 private:
  // The static function AppendFrames appends together the three input matrices,
  // some of which may be empty.
  static void AppendFrames(const MatrixBase<BaseFloat> &input1,
                           const MatrixBase<BaseFloat> &input2,
                           const MatrixBase<BaseFloat> &input3,
                           Matrix<BaseFloat> *output);

  // Context() is the number of frames on each side of a given frame,
  // that we need for context.
  int32 Context() const { return opts_.order * opts_.window; }

  // Does the delta computation.  Here, "output" will be resized to dimension
  // (input.NumRows() - Context() * 2) by (input.NumCols() * opts_.order)
  // "remainder" will be the last Context() rows of "input".
  void DeltaComputation(const MatrixBase<BaseFloat> &input,
                        Matrix<BaseFloat> *output,
                        Matrix<BaseFloat> *remainder) const;

  OnlFeatInputItf *input_; // underlying/inferior input object
  DeltaFeaturesOptions opts_;
  const int32 input_dim_;
  Matrix<BaseFloat> remainder_; // The last few frames of the input, that may
  // be needed for context purposes.

  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlDeltaInput);
};


} // namespace kaldi

#endif // KALDI_DEC_WRAP_DEC_WRAP_FEAT_INPUT_H_
