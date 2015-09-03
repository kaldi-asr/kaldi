// online/online-feat-input.cc

// Copyright 2012 Cisco Systems (author: Matthias Paulik)

//   Modifications to the original contribution by Cisco Systems made by:
//   Vassil Panayotov
//   Johns Hopkins University (author: Daniel Povey)

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

#include "online-feat-input.h"

namespace kaldi {


// This is a wrapper for ComputeInternal.  It behaves exactly the
// same as ComputeInternal, except that at the start of the file,
// ComputeInternal may return empty output multiple times in a row.
// This function prevents those initial non-productive calls, which
// may otherwise confuse decoder code into thinking there is
// a problem with the stream (too many timeouts), and cause it to fail.
bool OnlineCmnInput::Compute(Matrix<BaseFloat> *output) {
  
  int32 orig_nr = output->NumRows(), orig_nc = output->NumCols();
  int32 initial_t_in = t_in_;
  bool ans;
  while ((ans = ComputeInternal(output))) {
    if (output->NumRows() == 0 &&
        t_in_ != initial_t_in) {
      // we produced no output but added to our internal buffer.
      // Call ComputeInternal again.
      initial_t_in = t_in_;
      output->Resize(orig_nr, orig_nc); // make the same request.
    } else {
      return ans;
    }
  }
  return ans;
  // ans = false.  If ComputeInternal returned false,
  // it means we are done, so no point calling it again.
}


int32 OnlineCmnInput::NumOutputFrames(int32 num_new_frames,
                                      bool more_data) const {
  // Tells the caller, assuming we get given "num_new_frames" of input (and
  // given knowledge of whether there is more data coming), how many frames
  // would we be able to output?

  int32 max_t = t_in_ + num_new_frames;
  if (max_t >= min_window_ || !more_data) {
    // If this takes us to "min_window_" frames, we'll output all we have.
    return num_new_frames + t_in_ - t_out_;
  } else {
    return 0; // We'll wait till we have at least "min_window_" frames.
  }
}


// What happens at the start of the utterance is not really ideal, it would be
// better to have some "fake stats" extracted from typical data from this domain,
// to start with.  We'll have to do this later.
bool OnlineCmnInput::ComputeInternal(Matrix<BaseFloat> *output) {
  KALDI_ASSERT(output->NumRows() > 0 && output->NumCols() == Dim());

  Matrix<BaseFloat> input;
  input.Swap(output);
  
  bool more_data = input_->Compute(&input);

  int32 num_input_frames = input.NumRows();
  
  int32 output_frames = NumOutputFrames(num_input_frames,
                                        more_data);
  output->Resize(output_frames,
                 output_frames == 0 ? 0 : Dim());
  
  int32 output_counter = 0;
  for (int32 i = 0; i < num_input_frames; i++) {
    AcceptFrame(input.Row(i));
    while (t_in_ >= cmn_window_ && t_out_ < t_in_) {
      // We must output a frame now or we'll overwrite
      // frames we need in the buffer.
      SubVector<BaseFloat> this_frame(*output, output_counter);
      OutputFrame(&this_frame);
      output_counter++;
    }
  }
  for (; output_counter < output_frames; output_counter++) {
    SubVector<BaseFloat> this_frame(*output, output_counter);
    OutputFrame(&this_frame);
  }
  return more_data;
}

void OnlineCmnInput::AcceptFrame(const VectorBase<BaseFloat> &input) {
  KALDI_ASSERT(t_in_ <= t_out_ + cmn_window_);
  history_.Row(t_in_ % (cmn_window_ + 1)).CopyFromVec(input);
  t_in_++;
}

// Output the frame indexed "t_out_".
void OnlineCmnInput::OutputFrame(VectorBase<BaseFloat> *output) {
  KALDI_ASSERT(t_out_ < t_in_); // or there is nothing to output.
  // First set "sum_".
  if (t_out_ == 0) { // This is the first request for an output frame,
    // so in general we need to set sum_ to the sum of the first "min_window_"
    // frames.  We will have less than min_window_ frames if the input finished
    // before then (if the input were not finished, we'd not have reached this
    // code).
    int32 num_frames = t_in_ < min_window_ ? t_in_ : min_window_;
    for (int32 i = 0; i < num_frames; i++)
      sum_.AddVec(1.0, history_.Row(i));
  }
  int32 num_history_frames;
  if (t_out_ >= cmn_window_) num_history_frames = cmn_window_;
  else if (t_out_ < min_window_)
    num_history_frames = (t_in_ < min_window_ ? t_in_ : min_window_);
  else
    num_history_frames = t_out_;
  
  SubVector<BaseFloat> input_frame(history_, t_out_ % (cmn_window_ + 1));
  output->CopyFromVec(input_frame);
  output->AddVec(-1.0 / num_history_frames, sum_); // Apply CMN to the output.
  
  // Update sum.
  if (t_out_ >= min_window_)
    sum_.AddVec(1.0, input_frame);
  if (t_out_ >= cmn_window_) { // Remove the frame from "cmn_window_" frames ago.
    sum_.AddVec(-1.0, history_.Row((t_out_ - cmn_window_) % (cmn_window_ + 1)));
    KALDI_ASSERT(t_in_ == t_out_ + 1); // or else the frame indexed t_out_ -
                                       // cmn_window_ would not be the right one.
  }
  t_out_++;
}

#if !defined(_MSC_VER)

OnlineUdpInput::OnlineUdpInput(int32 port, int32 feature_dim):
    feature_dim_(feature_dim) {
  server_addr_.sin_family = AF_INET; // IPv4
  server_addr_.sin_addr.s_addr = INADDR_ANY; // listen on all interfaces
  server_addr_.sin_port = htons(port);
  sock_desc_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (sock_desc_ == -1)
    KALDI_ERR << "socket() call failed!";
  int32 rcvbuf_size = 30000;
  if (setsockopt(sock_desc_, SOL_SOCKET, SO_RCVBUF,
                 &rcvbuf_size, sizeof(rcvbuf_size)) == -1)
      KALDI_ERR << "setsockopt() failed to set receive buffer size!";
  if (bind(sock_desc_,
           reinterpret_cast<sockaddr*>(&server_addr_),
           sizeof(server_addr_)) == -1)
    KALDI_ERR << "bind() call failed!";
}


bool OnlineUdpInput::Compute(Matrix<BaseFloat> *output) {
  char buf[65535];
  socklen_t caddr_len = sizeof(client_addr_);
  ssize_t nrecv = recvfrom(sock_desc_, buf, sizeof(buf), 0,
                           reinterpret_cast<sockaddr*>(&client_addr_),
                           &caddr_len);
  if (nrecv == -1) {
    KALDI_WARN << "recvfrom() call error!";
    output->Resize(0, 0);
    return false;
  }
  std::stringstream ss(std::stringstream::in | std::stringstream::out);
  ss.write(buf, nrecv);
  output->Read(ss, true);
  return true;
}

#endif


OnlineLdaInput::OnlineLdaInput(OnlineFeatInputItf *input,
                               const Matrix<BaseFloat> &transform,
                               int32 left_context,
                               int32 right_context):
    input_(input), input_dim_(input->Dim()),
    left_context_(left_context), right_context_(right_context) {

  int32 tot_context = left_context + 1 + right_context;
  if (transform.NumCols() == input_dim_ * tot_context) {
    linear_transform_ = transform;
    // and offset_ stays empty.
  } else if (transform.NumCols() == input_dim_ * tot_context + 1) {
    linear_transform_.Resize(transform.NumRows(), transform.NumCols() - 1);
    linear_transform_.CopyFromMat(transform.Range(0, transform.NumRows(),
                                           0, transform.NumCols() - 1));
    offset_.Resize(transform.NumRows());
    offset_.CopyColFromMat(transform, transform.NumCols() - 1);
  } else {
    KALDI_ERR << "Invalid parameters supplied to OnlineLdaInput";
  }
}

// static
void OnlineLdaInput::SpliceFrames(const MatrixBase<BaseFloat> &input1,
                                  const MatrixBase<BaseFloat> &input2,
                                  const MatrixBase<BaseFloat> &input3,
                                  int32 context_window,
                                  Matrix<BaseFloat> *output) {
  KALDI_ASSERT(context_window > 0);
  const int32 size1 = input1.NumRows(), size2 = input2.NumRows(),
      size3 = input3.NumRows();
  int32 num_frames_in = size1 + size2 + size3,
      num_frames_out = num_frames_in - (context_window - 1),
      dim = std::max(input1.NumCols(), std::max(input2.NumCols(), input3.NumCols()));
  // do std::max in case one or more of the input matrices is empty.
  
  if (num_frames_out <= 0) {
    output->Resize(0, 0);
    return;
  }
  output->Resize(num_frames_out, dim * context_window);
  for (int32 t_out = 0; t_out < num_frames_out; t_out++) {
    for (int32 pos = 0; pos < context_window; pos++) {
      int32 t_in = t_out + pos;
      SubVector<BaseFloat> vec_out(output->Row(t_out), pos * dim, dim);
      if (t_in < size1)
        vec_out.CopyFromVec(input1.Row(t_in));
      else if (t_in < size1 + size2)
        vec_out.CopyFromVec(input2.Row(t_in - size1));
      else
        vec_out.CopyFromVec(input3.Row(t_in - size1 - size2));
    }
  }
}

void OnlineLdaInput::TransformToOutput(const MatrixBase<BaseFloat> &spliced_feats,
                                       Matrix<BaseFloat> *output) {
  if (spliced_feats.NumRows() == 0) {
    output->Resize(0, 0);
  } else {
    output->Resize(spliced_feats.NumRows(), linear_transform_.NumRows());
    output->AddMatMat(1.0, spliced_feats, kNoTrans,
                      linear_transform_, kTrans, 0.0);
    if (offset_.Dim() != 0)
      output->AddVecToRows(1.0, offset_);
  }
}

bool OnlineLdaInput::Compute(Matrix<BaseFloat> *output) {
  KALDI_ASSERT(output->NumRows() > 0 &&
               output->NumCols() == linear_transform_.NumRows());
  // If output->NumRows() == 0, it corresponds to a request for zero frames,
  // which makes no sense.

  // We request the same number of frames of data that we were requested.
  Matrix<BaseFloat> input(output->NumRows(), input_dim_);
  bool ans = input_->Compute(&input);
  // If we got no input (timed out) and we're not at the end, we return
  // empty output.

  if (input.NumRows() == 0 && ans) {
    output->Resize(0, 0);
    return ans;
  } else if (input.NumRows() == 0 && !ans) {
    // The end of the input stream, but no input this time.
    if (remainder_.NumRows() == 0) {
      output->Resize(0, 0);
      return ans;
    }
  }

  // If this is the first segment of the utterance, we put in the
  // initial duplicates of the first frame, numbered "left_context".
  if (remainder_.NumRows() == 0 && input.NumRows() != 0 && left_context_ != 0) {
    remainder_.Resize(left_context_, input_dim_);
    for (int32 i = 0; i < left_context_; i++)
      remainder_.Row(i).CopyFromVec(input.Row(0));
  }

  // If this is the last segment, we put in the final duplicates of the
  // last frame, numbered "right_context".
  Matrix<BaseFloat> tail;
  if (!ans && right_context_ > 0) {
    tail.Resize(right_context_, input_dim_);
    for (int32 i = 0; i < right_context_; i++) {
      if (input.NumRows() > 0)
        tail.Row(i).CopyFromVec(input.Row(input.NumRows() - 1));
      else
        tail.Row(i).CopyFromVec(remainder_.Row(remainder_.NumRows() - 1));
    }
  }
  
  Matrix<BaseFloat> spliced_feats;
  int32 context_window = left_context_ + 1 + right_context_;
  // The next line is a call to a member function.
  SpliceFrames(remainder_, input, tail, context_window, &spliced_feats);
  TransformToOutput(spliced_feats, output);
  ComputeNextRemainder(input);
  return ans; 
}

void OnlineLdaInput::ComputeNextRemainder(const MatrixBase<BaseFloat> &input) {
  // The size of the remainder that we propagate to the next frame is
  // context_window - 1, if available.
  int32 context_window = left_context_ + 1 + right_context_;
  int32 next_remainder_len = std::min(context_window - 1,
                                      remainder_.NumRows() + input.NumRows());
  if (next_remainder_len == 0) {
    remainder_.Resize(0, 0);
    return;
  }
  Matrix<BaseFloat> next_remainder(next_remainder_len, input_dim_);
  int32 rsize = remainder_.NumRows(), isize = input.NumRows();
  for (int32 i = 0; i < next_remainder_len; i++) {
    SubVector<BaseFloat> dest(next_remainder, i);
    int32 t = (rsize + isize) - next_remainder_len + i;
    // Here, t is an offset into a numbering of the frames where we first have
    // the old "remainder" frames, then the regular frames.
    if (t < rsize) dest.CopyFromVec(remainder_.Row(t));
    else dest.CopyFromVec(input.Row(t - rsize));
  }
  remainder_ = next_remainder;
}


bool OnlineCacheInput::Compute(Matrix<BaseFloat> *output) {
  bool ans = input_->Compute(output);
  if (output->NumRows() != 0)
    data_.push_back(new Matrix<BaseFloat>(*output));
  return ans;
}

void OnlineCacheInput::GetCachedData(Matrix<BaseFloat> *output) {
  int32 num_frames = 0, dim = 0;
  for (size_t i = 0; i < data_.size(); i++) {
    num_frames += data_[i]->NumRows();
    dim = data_[i]->NumCols();
  }
  output->Resize(num_frames, dim);
  int32 frame_offset = 0;
  for (size_t i = 0; i < data_.size(); i++) {
    int32 this_frames = data_[i]->NumRows();
    output->Range(frame_offset, this_frames, 0, dim).CopyFromMat(*data_[i]);
    frame_offset += this_frames;
  }
  KALDI_ASSERT(frame_offset == num_frames);
}

void OnlineCacheInput::Deallocate() {
  for (size_t i = 0; i < data_.size(); i++) delete data_[i];
  data_.clear();
}


OnlineDeltaInput::OnlineDeltaInput(const DeltaFeaturesOptions &delta_opts,
                                   OnlineFeatInputItf *input):
    input_(input), opts_(delta_opts), input_dim_(input_->Dim()) { }


// static
void OnlineDeltaInput::AppendFrames(const MatrixBase<BaseFloat> &input1,
                                    const MatrixBase<BaseFloat> &input2,
                                    const MatrixBase<BaseFloat> &input3,
                                    Matrix<BaseFloat> *output) {
  const int32 size1 = input1.NumRows(), size2 = input2.NumRows(),
      size3 = input3.NumRows(), size_out = size1 + size2 + size3;
  if (size_out == 0) {
    output->Resize(0, 0);
    return;
  }
  // do std::max in case one or more of the input matrices is empty.  
  int32 dim = std::max(input1.NumCols(),
                       std::max(input2.NumCols(), input3.NumCols()));

  output->Resize(size_out, dim);
  if (size1 != 0)
    output->Range(0, size1, 0, dim).CopyFromMat(input1);
  if (size2 != 0)
    output->Range(size1, size2, 0, dim).CopyFromMat(input2);
  if (size3 != 0)
    output->Range(size1 + size2, size3, 0, dim).CopyFromMat(input3);
}

void OnlineDeltaInput::DeltaComputation(const MatrixBase<BaseFloat> &input,
                                        Matrix<BaseFloat> *output,
                                        Matrix<BaseFloat> *remainder) const {
  int32 input_rows = input.NumRows(),
      output_rows = std::max(0, input_rows - Context() * 2),
      remainder_rows = std::min(input_rows, Context() * 2),
      input_dim = input_dim_,
      output_dim = Dim();
  if (remainder_rows > 0) {
    remainder->Resize(remainder_rows, input_dim);
    remainder->CopyFromMat(input.Range(input_rows - remainder_rows,
                                       remainder_rows, 0, input_dim));
  } else {
    remainder->Resize(0, 0);
  }
  if (output_rows > 0) {
    output->Resize(output_rows, output_dim);
    DeltaFeatures delta(opts_);
    for (int32 output_frame = 0; output_frame < output_rows; output_frame++) {
      int32 input_frame = output_frame + Context();
      SubVector<BaseFloat> output_row(*output, output_frame);
      delta.Process(input, input_frame, &output_row);
    }
  } else {
    output->Resize(0, 0);
  }
}                                     

bool OnlineDeltaInput::Compute(Matrix<BaseFloat> *output) {
  KALDI_ASSERT(output->NumRows() > 0 &&
               output->NumCols() == Dim());
  // If output->NumRows() == 0, it corresponds to a request for zero frames,
  // which makes no sense.

  // We request the same number of frames of data that we were requested.
  Matrix<BaseFloat> input(output->NumRows(), input_dim_);
  bool ans = input_->Compute(&input);

  // If we got no input (timed out) and we're not at the end, we return
  // empty output.
  if (input.NumRows() == 0 && ans) {
    output->Resize(0, 0);
    return ans;
  } else if (input.NumRows() == 0 && !ans) {
    // The end of the input stream, but no input this time.
    if (remainder_.NumRows() == 0) {
      output->Resize(0, 0);
      return ans;
    }
  }

  // If this is the first segment of the utterance, we put in the
  // initial duplicates of the first frame, numbered "Context()"
  if (remainder_.NumRows() == 0 && input.NumRows() != 0 && Context() != 0) {
    remainder_.Resize(Context(), input_dim_);
    for (int32 i = 0; i < Context(); i++)
      remainder_.Row(i).CopyFromVec(input.Row(0));
  }

  // If this is the last segment, we put in the final duplicates of the
  // last frame, numbered "Context()".
  Matrix<BaseFloat> tail;
  if (!ans && Context() > 0) {
    tail.Resize(Context(), input_dim_);
    for (int32 i = 0; i < Context(); i++) {
      if (input.NumRows() > 0)
        tail.Row(i).CopyFromVec(input.Row(input.NumRows() - 1));
      else
        tail.Row(i).CopyFromVec(remainder_.Row(remainder_.NumRows() - 1));
    }
  }
  
  Matrix<BaseFloat> appended_feats;
  AppendFrames(remainder_, input, tail, &appended_feats);
  DeltaComputation(appended_feats, output, &remainder_);
  return ans; 
}



void OnlineFeatureMatrix::GetNextFeatures() {
  if (finished_) return; // Nothing to do.
  
  // We always keep the most recent frame of features, if present,
  // in case it is needed (this may happen when someone calls
  // IsLastFrame(), which requires us to get the next frame, while
  // they're stil processing this frame.
  bool have_last_frame = (feat_matrix_.NumRows() != 0);
  Vector<BaseFloat> last_frame;
  if (have_last_frame)
    last_frame = feat_matrix_.Row(feat_matrix_.NumRows() - 1);

  int32 iter;
  for (iter = 0; iter < opts_.num_tries; iter++) {
    Matrix<BaseFloat> next_features(opts_.batch_size, feat_dim_);
    finished_ = ! input_->Compute(&next_features);
    if (next_features.NumRows() == 0 && ! finished_) {
      // It timed out.  Try again.
      continue;
    }
    if (next_features.NumRows() > 0) {
      int32 new_size = (have_last_frame ? 1 : 0) +
          next_features.NumRows();
      feat_offset_ += feat_matrix_.NumRows() -
          (have_last_frame ? 1 : 0); // we're discarding this many
                                     // frames.
      feat_matrix_.Resize(new_size, feat_dim_, kUndefined);
      if (have_last_frame) {
        feat_matrix_.Row(0).CopyFromVec(last_frame);
        feat_matrix_.Range(1, next_features.NumRows(), 0, feat_dim_).
            CopyFromMat(next_features);
      } else {
        feat_matrix_.CopyFromMat(next_features);
      }
    }
    break;
  }
  if (iter == opts_.num_tries) { // we fell off the loop
    KALDI_WARN << "After " << opts_.num_tries << ", got no features, giving up.";
    finished_ = true; // We set finished_ to true even though the stream
    // doesn't say it's finished, because the delay is too much-- we gave up.
  }
}


bool OnlineFeatureMatrix::IsValidFrame (int32 frame) {
   KALDI_ASSERT(frame >= feat_offset_ &&
               "You are attempting to get expired frames.");
  if (frame < feat_offset_ + feat_matrix_.NumRows())
    return true;
  else {
    GetNextFeatures();
    if (frame < feat_offset_ + feat_matrix_.NumRows())
      return true;
    else {
      if (finished_) return false;
      else {
        KALDI_WARN << "Unexpected point reached in code: "
                   << "possibly you are skipping frames?";
        return false;
      }
    }
  }
}

SubVector<BaseFloat> OnlineFeatureMatrix::GetFrame(int32 frame) {
  if (frame < feat_offset_)
    KALDI_ERR << "Attempting to get a discarded frame.";
  if (frame >= feat_offset_ + feat_matrix_.NumRows())
    KALDI_ERR << "Attempt get frame without check its validity.";
  return feat_matrix_.Row(frame - feat_offset_);
}


} // namespace kaldi

