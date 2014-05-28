// online/online-feat-input.h

// Copyright 2012 Cisco Systems (author: Matthias Paulik)
//           2012-2013  Vassil Panayotov
//           2013 Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_ONLINE_ONLINE_FEAT_INPUT_H_
#define KALDI_ONLINE_ONLINE_FEAT_INPUT_H_

#if !defined(_MSC_VER)
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif

#include "online-audio-source.h"
#include "feat/feature-functions.h"

namespace kaldi {

// Interface specification
class OnlineFeatInputItf {
 public:
  // Produces feature vectors in some way.
  // The features may be e.g. extracted from an audio samples, received and/or
  // transformed from another OnlineFeatInput class etc.
  //
  // "output" - a matrix to store the extracted feature vectors in its rows.
  //            The number of rows (NumRows()) of "output" when the function is
  //            called, is treated as a hint of how many frames the user wants,
  //            but this function does not promise to produce exactly that many:
  //            it may be slightly more, less, or even zero, on a given call.
  //            Zero frames may be returned because we timed out or because
  //            we're at the beginning of the file and some buffering is going on.
  //            In that case you should try again.  The function will return "false"
  //            when it knows the stream is finished, but if it returns nothing
  //            several times in a row you may want to terminate processing the
  //            stream.
  //
  // Note: similar to the OnlineAudioInput::Read(), Compute() previously
  //       had a second argument - "timeout". Again we decided against including
  //       this parameter in the interface specification. Instead we are
  //       considering time out handling to be implementation detail, and if needed
  //       it should be configured, through the descendant class' constructor,
  //       or by other means.
  //       For consistency, we recommend 'timeout' values greater than zero
  //       to mean that Compute() should not block for more than that number
  //       of milliseconds, and to return whatever data it has, when the timeout
  //       period is exceeded.
  //
  // Returns "false" if we know the underlying data source has no more data, and
  // true if there may be more data.
  virtual bool Compute(Matrix<BaseFloat> *output) = 0;

  virtual int32 Dim() const = 0; // Return the output dimension of these features.
  
  virtual ~OnlineFeatInputItf() {}
};


// Acts as a proxy to an underlying OnlineFeatInput.
// Applies cepstral mean normalization
class OnlineCmnInput: public OnlineFeatInputItf {
 public:
  // "input" - the underlying(unnormalized) feature source
  // "cmn_window" - the count of the preceding vectors over which the average is
  //                calculated
  // "min_window" - the minimum count of frames for which it will compute the
  //                mean, at the start of the file.  Adds latency but only at the
  //                start
  OnlineCmnInput(OnlineFeatInputItf *input, int32 cmn_window, int32 min_window)
      : input_(input), cmn_window_(cmn_window), min_window_(min_window),
        history_(cmn_window + 1, input->Dim()), t_in_(0), t_out_(0),
        sum_(input->Dim()) { KALDI_ASSERT(cmn_window >= min_window && min_window > 0); }
  
  virtual bool Compute(Matrix<BaseFloat> *output);

  virtual int32 Dim() const { return input_->Dim(); }

 private:
  virtual bool ComputeInternal(Matrix<BaseFloat> *output);

  
  OnlineFeatInputItf *input_;
  const int32 cmn_window_; // > 0
  const int32 min_window_; // > 0, < cmn_window_.
  Matrix<BaseFloat> history_; // circular-buffer history, of dim (cmn_window_ +
                              // 1, feat-dim).  The + 1 is to serve as a place
                              // for the frame we're about to normalize.

  void AcceptFrame(const VectorBase<BaseFloat> &input); // Accept the next frame
                                                        // of input (read into the
                                                        // history buffer).
  void OutputFrame(VectorBase<BaseFloat> *output); // Output the next frame.
  
  int32 NumOutputFrames(int32 num_new_frames,
                        bool more_data) const; // Tells the caller, assuming
  // we get given "num_new_frames" of input (and given knowledge of whether
  // there is more data coming), how many frames would we be able to
  // output?
  
  
  int64 t_in_; // Time-counter for what we've obtained from the input.
  int64 t_out_; // Time-counter for what we've written to the output.
  
  Vector<double> sum_; // Sum of the frames from t_out_ - HistoryLength(t_out_),
                       // to t_out_ - 1.
  
  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineCmnInput);
};


class OnlineCacheInput : public OnlineFeatInputItf {
 public:
  OnlineCacheInput(OnlineFeatInputItf *input): input_(input) { }
  
  // The Compute function just forwards to the previous member of the
  // chain, except that we locally accumulate the result, and
  // GetCachedData() will return the entire input up to the current time.
  virtual bool Compute(Matrix<BaseFloat> *output);

  void GetCachedData(Matrix<BaseFloat> *output);
  
  int32 Dim() const { return input_->Dim(); }
  
  void Deallocate();
    
  virtual ~OnlineCacheInput() { Deallocate(); }
  
 private:
  OnlineFeatInputItf *input_;
  // data_ is a list of all the outputs we produced in successive
  // calls to Compute().  The memory is owned here.
  std::vector<Matrix<BaseFloat>* > data_;
};


#if !defined(_MSC_VER)

// Accepts features over an UDP socket
// The current implementation doesn't support the "timeout" -
// the server is waiting for data indefinetily long time.
class OnlineUdpInput : public OnlineFeatInputItf {
 public:
  OnlineUdpInput(int32 port, int32 feature_dim);

  virtual bool Compute(Matrix<BaseFloat> *output);

  virtual int32 Dim() const { return feature_dim_; }

  const sockaddr_in& client_addr() const { return client_addr_; }

  const int32 descriptor() const { return sock_desc_; }
  
 private:
  int32 feature_dim_;
  // various BSD sockets-related data structures
  int32 sock_desc_; // socket descriptor
  sockaddr_in server_addr_;
  sockaddr_in client_addr_;
};

#endif


// Splices the input features and applies a transformation matrix.
// Note: the transformation matrix will usually be a linear transformation
// [output-dim x input-dim] but we accept an affine transformation too.
class OnlineLdaInput: public OnlineFeatInputItf {
 public:
  OnlineLdaInput(OnlineFeatInputItf *input,
                 const Matrix<BaseFloat> &transform,
                 int32 left_context,
                 int32 right_context);

  virtual bool Compute(Matrix<BaseFloat> *output);

  virtual int32 Dim() const { return linear_transform_.NumRows(); }

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
  
  OnlineFeatInputItf *input_; // underlying/inferior input object
  const int32 input_dim_; // dimension of the feature vectors before xform
  const int32 left_context_;
  const int32 right_context_;
  Matrix<BaseFloat> linear_transform_; // transform matrix (linear part only)
  Vector<BaseFloat> offset_; // Offset, if present; else empty.
  Matrix<BaseFloat> remainder_; // The last few frames of the input, that may
  // be needed for context purposes.
  
  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineLdaInput);
};


// Does the time-derivative computation (e.g., adding deltas and delta-deltas).
// This is standard in more "old-fashioned" feature extraction.  Like an online
// version of the function ComputeDeltas in feat/feature-functions.h, where the
// class DeltaFeaturesOptions is also defined.
class OnlineDeltaInput: public OnlineFeatInputItf {
 public:
  OnlineDeltaInput(const DeltaFeaturesOptions &delta_opts,
                   OnlineFeatInputItf *input);
  
  virtual bool Compute(Matrix<BaseFloat> *output);

  virtual int32 Dim() const { return input_dim_ * (opts_.order + 1); }
  
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
  
  OnlineFeatInputItf *input_; // underlying/inferior input object
  DeltaFeaturesOptions opts_;
  const int32 input_dim_;
  Matrix<BaseFloat> remainder_; // The last few frames of the input, that may
  // be needed for context purposes.
  
  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineDeltaInput);
};

// Implementation, that is meant to be used to read samples from an
// OnlineAudioSource and to extract MFCC/PLP features in the usual way
template <class E>
class OnlineFeInput : public OnlineFeatInputItf {
 public:
  // "au_src" - OnlineAudioSourceItf object
  // "fe" - object implementing MFCC/PLP feature extraction
  // "frame_size" - frame extraction window size in audio samples
  // "frame_shift" - feature frame width in audio samples
  OnlineFeInput(OnlineAudioSourceItf *au_src, E *fe,
                const int32 frame_size, const int32 frame_shift);

  virtual int32 Dim() const { return extractor_->Dim(); }
  
  virtual bool Compute(Matrix<BaseFloat> *output);

 private:
  OnlineAudioSourceItf *source_; // audio source
  E *extractor_; // the actual feature extractor used
  const int32 frame_size_;
  const int32 frame_shift_;
  Vector<BaseFloat> wave_; // the samples to be passed for extraction
  Vector<BaseFloat> wave_remainder_; // the samples remained from the previous
                                     // feature batch

  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineFeInput);
};

template<class E>
OnlineFeInput<E>::OnlineFeInput(OnlineAudioSourceItf *au_src, E *fe,
                                   int32 frame_size, int32 frame_shift)
    : source_(au_src), extractor_(fe),
      frame_size_(frame_size), frame_shift_(frame_shift) {}

template<class E> bool
OnlineFeInput<E>::Compute(Matrix<BaseFloat> *output) {
  MatrixIndexT nvec = output->NumRows(); // the number of output vectors
  if (nvec <= 0) {
    KALDI_WARN << "No feature vectors requested?!";
    return true;
  }

  // Prepare the input audio samples
  int32 samples_req = frame_size_ + (nvec - 1) * frame_shift_;
  Vector<BaseFloat> read_samples(samples_req);

  bool ans = source_->Read(&read_samples);
  
  Vector<BaseFloat> all_samples(wave_remainder_.Dim() + read_samples.Dim());
  all_samples.Range(0, wave_remainder_.Dim()).CopyFromVec(wave_remainder_);
  all_samples.Range(wave_remainder_.Dim(), read_samples.Dim()).
      CopyFromVec(read_samples);
  
  // Extract the features
  if (all_samples.Dim() >= frame_size_) {
    extractor_->Compute(all_samples, 1.0, output, &wave_remainder_);
  } else {
    output->Resize(0, 0);
    wave_remainder_ = all_samples;
  }
  
  return ans;
}

struct OnlineFeatureMatrixOptions {
  int32 batch_size; // number of frames to request each time.
  int32 num_tries; // number of tries of getting no output and timing out,
                   // before we give up.
  OnlineFeatureMatrixOptions(): batch_size(27),
                                num_tries(5) { }
  void Register(OptionsItf *po) {
    po->Register("batch-size", &batch_size,
                 "Number of feature vectors processed w/o interruption");
    po->Register("num-tries", &num_tries,
                 "Number of successive repetitions of timeout before we "
                 "terminate stream");
  }
};

// The class OnlineFeatureMatrix wraps something of type
// OnlineFeatInputItf in a manner that is convenient for
// a Decodable type to consume.
class OnlineFeatureMatrix {
 public:
  OnlineFeatureMatrix(const OnlineFeatureMatrixOptions &opts,
                      OnlineFeatInputItf *input):
      opts_(opts), input_(input), feat_dim_(input->Dim()),
      feat_offset_(0), finished_(false) { }
  
  bool IsValidFrame (int32 frame); 

  int32 Dim() const { return feat_dim_; }

  // GetFrame() will die if it's not a valid frame; you have to
  // call IsValidFrame() for this frame, to see whether it
  // is valid.
  SubVector<BaseFloat> GetFrame(int32 frame);

  bool Good(); // returns true if we have at least one frame.
 private:
  void GetNextFeatures(); // called when we need more features.  Guarantees
  // to get at least one more frame, or set finished_ = true.
  
  const OnlineFeatureMatrixOptions opts_;
  OnlineFeatInputItf *input_;
  int32 feat_dim_;
  Matrix<BaseFloat> feat_matrix_;
  int32 feat_offset_; // the offset of the first frame in the current batch
  bool finished_; // True if there are no more frames to be got from the input.
};


} // namespace kaldi

#endif // KALDI_ONLINE_ONLINE_FEAT_INPUT_H_
