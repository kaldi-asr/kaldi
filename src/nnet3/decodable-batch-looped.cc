// nnet3/decodable-batch-looped.cc

// Copyright      2020-2021  Xiaomi Corporation (Author: Zhao Yan)

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

#if HAVE_CUDA == 1
#include "nnet3/decodable-batch-looped.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-compile-looped.h"
#include "base/timer.h"

namespace kaldi {
namespace nnet3 {


DecodableNnetBatchLoopedInfo::DecodableNnetBatchLoopedInfo(
    const NnetBatchLoopedComputationOptions &opts,
    Nnet *nnet):
    opts(opts), nnet(*nnet) {
  Init(opts, nnet);
}

DecodableNnetBatchLoopedInfo::DecodableNnetBatchLoopedInfo(
    const NnetBatchLoopedComputationOptions &opts,
    const Vector<BaseFloat> &priors,
    Nnet *nnet):
    opts(opts), nnet(*nnet), log_priors(priors) {
  if (log_priors.Dim() != 0)
    log_priors.ApplyLog();
  Init(opts, nnet);
}


DecodableNnetBatchLoopedInfo::DecodableNnetBatchLoopedInfo(
    const NnetBatchLoopedComputationOptions &opts,
    AmNnetSimple *am_nnet):
    opts(opts), nnet(am_nnet->GetNnet()), log_priors(am_nnet->Priors()) {
  if (log_priors.Dim() != 0)
    log_priors.ApplyLog();
  Init(opts, &(am_nnet->GetNnet()));
}


void DecodableNnetBatchLoopedInfo::Init(
    const NnetBatchLoopedComputationOptions &opts,
    Nnet *nnet) {
  opts.Check();
  KALDI_ASSERT(IsSimpleNnet(*nnet));
  has_ivectors = (nnet->InputDim("ivector") > 0);
  int32 left_context, right_context;
  int32 extra_right_context = 0;
  ComputeSimpleNnetContext(*nnet, &left_context, &right_context);
  frames_left_context = left_context + opts.extra_left_context_initial;
  frames_right_context = right_context + extra_right_context;
  frames_per_chunk = GetChunkSize(*nnet, opts.frame_subsampling_factor,
                                  opts.frames_per_chunk);
  output_dim = nnet->OutputDim("output");
  KALDI_ASSERT(output_dim > 0);
  // note, ivector_period is hardcoded to the same as frames_per_chunk_.
  int32 ivector_period = frames_per_chunk;
  if (has_ivectors)
    ModifyNnetIvectorPeriod(ivector_period, nnet);

  num_chunk1_ivector_frames = 0;
  num_ivector_frames = 0;
  computation.resize(opts.max_batch_size + 1);
  for (int32 num_sequences = 1; 
       num_sequences <= opts.max_batch_size; 
       num_sequences++) {
    CreateLoopedComputationRequest(*nnet, frames_per_chunk,
                                   opts.frame_subsampling_factor,
                                   ivector_period,
                                   frames_left_context,
                                   frames_right_context,
                                   num_sequences,
                                   &request1, &request2, &request3);
    CompileLooped(*nnet, opts.optimize_config, 
                  request1, request2, request3,
                  &computation[num_sequences]);
    computation[num_sequences].ComputeCudaIndexes();
    
    if (num_sequences == 1 && has_ivectors) {
      KALDI_ASSERT(request1.inputs.size() == 2);
      num_chunk1_ivector_frames = request1.inputs[1].indexes.size();
      num_ivector_frames = request2.inputs[1].indexes.size();
    }
  }

  if (has_ivectors) {
    KALDI_ASSERT(num_chunk1_ivector_frames > 0 && num_ivector_frames > 0);
  }
}

NnetBatchLoopedComputer::NnetBatchLoopedComputer(
    const DecodableNnetBatchLoopedInfo &info):
    info_(info), is_working_(false) {
  snapshots_.resize(info_.opts.max_batch_size+1);
  int32 batch_size;
  for (batch_size = 1; batch_size <= info_.opts.max_batch_size; batch_size++) {
    std::vector<bool> batch_first;
    AdvanceChunkUntilStable(batch_size, &batch_first);
    if (batch_first_.size() == 0) batch_first_ = batch_first;
    else KALDI_ASSERT(batch_first_ == batch_first);
  }

  KALDI_ASSERT(batch_first_.size() > 0);
  KALDI_ASSERT(CuDevice::Instantiate().Enabled());
  
  Start();
}

NnetBatchLoopedComputer::~NnetBatchLoopedComputer() {
  // Stop the thread which handles computation
  Stop();
}

void NnetBatchLoopedComputer::AdvanceChunkUntilStable(
    int32 batch_size, 
    std::vector<bool> *batch_first) {
  int32 input_dim = info_.nnet.InputDim("input");
  int32 ivector_dim = info_.has_ivectors ? info_.nnet.InputDim("ivector") : 0;
  NnetComputer computer(info_.opts.compute_config,
                        info_.computation[batch_size],
                        info_.nnet,
                        NULL);
  batch_first->clear();

  Timer timer;
  for(int32 num_chunks = 0; ; num_chunks++) {
    if (num_chunks > 8)
      KALDI_ERR << "Try too many chunks for batch-size=" << batch_size 
                << ", maybe there is something wrong!";

    NnetComputer temp(computer);

    int32 num_input_frames;
    if (0 == num_chunks) {
      num_input_frames = info_.frames_left_context 
        + info_.frames_per_chunk + info_.frames_right_context;
    }
    else {
      num_input_frames = info_.frames_per_chunk;
    }
    CuMatrix<BaseFloat> feats_base(num_input_frames, input_dim, kUndefined);
    feats_base.SetRandn();

    CuMatrix<BaseFloat> feats_chunk(batch_size * num_input_frames, 
                                    input_dim, 
                                    kUndefined);
    for (int32 i = 0; i < batch_size; i++) {
      CuSubMatrix<BaseFloat> sub = feats_chunk.RowRange(i * num_input_frames, 
                                                        num_input_frames);
      sub.CopyFromMat(feats_base);
    }

    computer.AcceptInput("input", &feats_chunk);

    if (info_.has_ivectors) {
      int32 num_ivectors = num_chunks == 0 ?
                           info_.num_chunk1_ivector_frames :
                           info_.num_ivector_frames;
      KALDI_ASSERT(num_ivectors > 0);

      CuMatrix<BaseFloat> ivectors_base(num_ivectors, ivector_dim, kUndefined);
      ivectors_base.SetRandn();

      CuMatrix<BaseFloat> ivectors_chunk(batch_size * num_ivectors, 
                                         ivector_dim, 
                                         kUndefined);
      for (int32 i = 0; i < batch_size; i++) {
        CuSubMatrix<BaseFloat> sub = ivectors_chunk.RowRange(i * num_ivectors, 
                                                             num_ivectors);
        sub.CopyFromMat(ivectors_base);
      }
      computer.AcceptInput("ivector", &ivectors_chunk);
    }

    CuMatrix<BaseFloat> output;
    computer.Run();
    computer.GetOutputDestructive("output", &output);

    if (computer.Equal(temp)) {
      if (batch_size > 1) {
        const std::vector< CuMatrix<BaseFloat> > &matrices = computer.GetMatrices();
        for (std::size_t i = 0; i < matrices.size(); i++) {
          const CuMatrix<BaseFloat> &matrix = matrices[i];
          KALDI_ASSERT(matrix.NumRows() % batch_size == 0);
          int32 stream_num_rows = matrix.NumRows() / batch_size;
          if (matrix.NumRows() > 0 && matrix.NumCols() > 0) {
            CuSubMatrix<BaseFloat> submatrix1 = matrix.RowRange(0, stream_num_rows);
            CuSubMatrix<BaseFloat> submatrix2 = matrix.RowRange(
                stream_num_rows * (batch_size - 1), 
                stream_num_rows);
            CuSubMatrix<BaseFloat> submatrix3(matrix.RowData(0), 
                                              stream_num_rows,
                                              matrix.NumCols(),
                                              matrix.Stride() * batch_size); 
            CuSubMatrix<BaseFloat> submatrix4(matrix.RowData(batch_size - 1),
                                              stream_num_rows,
                                              matrix.NumCols(),
                                              matrix.Stride() * batch_size);

            if (submatrix1.ApproxEqual(submatrix2))
              batch_first->push_back(true);
            else if (submatrix3.ApproxEqual(submatrix4))
              batch_first->push_back(false);
            else
              KALDI_ERR << "One of matrix which storing state for multiple "
                        << "streams is neither batch first nor time first.";
          }
        }
      }

      computer.GetSnapshot(&snapshots_[batch_size]);
      double elapsed = timer.Elapsed();
      KALDI_WARN << "After try " << (num_chunks+1) 
                 << " chunks, the NnetComputer for batch-size=" << batch_size
                 << " becomes stable, it taken " << elapsed << "s.";
      break;
    }
  }
}

void NnetBatchLoopedComputer::Enqueue(NnetComputeRequest *request) {
  KALDI_ASSERT(request != NULL);
  bool first_chunk = request->first_chunk;
  std::queue<QueueElement> &queue = first_chunk ? chunk1_queue_ : queue_;
 
  std::unique_lock<std::mutex> lck(mtx_);
  queue.push(std::make_pair(request, std::chrono::system_clock::now()));
  if (queue.size() >= info_.opts.max_batch_size)
    condition_variable_.notify_one();
}

bool NnetBatchLoopedComputer::Continue() {
  if (!is_working_) {
    std::unique_lock<std::mutex> lck(mtx_);
    return (queue_.size() > 0 || chunk1_queue_.size() > 0);
  }

  return true;
}

void NnetBatchLoopedComputer::Compute() {
  int32 max_batch_size = info_.opts.max_batch_size;
  std::vector<NnetComputeRequest*> batch_requests;
  std::vector<NnetComputeRequest*> chunk1_batch_requests;

  {
    std::unique_lock<std::mutex> lck(mtx_);
    if (queue_.size() < max_batch_size && chunk1_queue_.size() < max_batch_size) {
      TimePoint now = std::chrono::system_clock::now();
      TimePoint target = now;
      target += std::chrono::microseconds(info_.opts.compute_interval);
      if (target > now) condition_variable_.wait_until(lck, target);
    }

    while (!queue_.empty()) {
      if (batch_requests.size() == max_batch_size) break;
      batch_requests.push_back(queue_.front().first);
      queue_.pop();
    }

    while (!chunk1_queue_.empty()) {
      if (chunk1_batch_requests.size() == max_batch_size) break;
      chunk1_batch_requests.push_back(chunk1_queue_.front().first);
      chunk1_queue_.pop();
    }
  }

  if (batch_requests.size() > 0) AdvanceChunk(batch_requests);
  if (chunk1_batch_requests.size() > 0) AdvanceChunk(chunk1_batch_requests);
}

void *NnetBatchLoopedComputer::ThreadFunction(void *para) {
  NnetBatchLoopedComputer *computer = 
    reinterpret_cast<NnetBatchLoopedComputer *>(para);

  while (computer->Continue()) {
    computer->Compute();
  }

  return NULL;
}

void NnetBatchLoopedComputer::Start() {
  is_working_ = true;
  work_thread_ = std::thread(ThreadFunction, this);
}

void NnetBatchLoopedComputer::Stop() {
  is_working_ = false;
  if (work_thread_.joinable()) work_thread_.join();
}

void NnetBatchLoopedComputer::AdvanceChunk(
    const std::vector<NnetComputeRequest*> &requests) {
  KALDI_ASSERT(requests.size() > 0 && 
               requests.size() <= info_.opts.max_batch_size);
  int32 batch_size = requests.size();

  std::vector<NnetComputeState *> state;
  int32 num_inputs = requests[0]->inputs.NumRows();
  int32 num_ivectors = requests[0]->ivectors.NumRows();
  int32 dim_inputs = requests[0]->inputs.NumCols();
  int32 dim_ivectors = requests[0]->ivectors.NumCols();
  bool  first_chunk = requests[0]->first_chunk;
  
  NnetComputer computer(info_.opts.compute_config, 
                        info_.computation[batch_size], 
                        info_.nnet, 
                        NULL,
                        first_chunk ? NULL : &(snapshots_[batch_size]));
  
  for (std::size_t i = 0; i < requests.size(); i++) {
    if (!requests[i]->first_chunk) state.push_back(&(requests[i]->state));
    KALDI_ASSERT(requests[i]->first_chunk == first_chunk);
    KALDI_ASSERT(requests[i]->inputs.NumRows() == num_inputs);
    KALDI_ASSERT(requests[i]->ivectors.NumRows() == num_ivectors);
  }

  if (!first_chunk) computer.SetState(batch_first_, batch_size, state);

  CuMatrix<BaseFloat> feats_chunk(batch_size * num_inputs, 
                                  dim_inputs, 
                                  kUndefined);
  for (std::size_t i = 0; i < requests.size(); i++) {
    CuSubMatrix<BaseFloat> feats = feats_chunk.RowRange(i * num_inputs, 
                                                        num_inputs);
    feats.CopyFromMat(requests[i]->inputs);
  }
  
  computer.AcceptInput("input", &feats_chunk);

  if (info_.has_ivectors) {
    int32 num_ivectors_request = first_chunk ? 
                                 info_.num_chunk1_ivector_frames : 
                                 info_.num_ivector_frames;
    KALDI_ASSERT(num_ivectors == num_ivectors_request);
    CuMatrix<BaseFloat> ivectors_chunk(batch_size * num_ivectors, 
                                       dim_ivectors, 
                                       kUndefined);
    for (std::size_t i = 0; i < requests.size(); i++) {
      CuSubMatrix<BaseFloat> ivectors = ivectors_chunk.RowRange(i * num_ivectors, 
                                                                num_ivectors);
      ivectors.CopyFromMat(requests[i]->ivectors);
    }

    computer.AcceptInput("ivector", &ivectors_chunk);
  }

  computer.Run();
  
  {
    CuMatrix<BaseFloat> outputs_chunk;
    CuMatrix<BaseFloat> outputs_xent_chunk;
    computer.GetOutputDestructive("output", &outputs_chunk);

    if (info_.log_priors.Dim() != 0) {
      outputs_chunk.AddVecToRows(-1.0, info_.log_priors);
    }

    outputs_chunk.Scale(info_.opts.acoustic_scale);

    // Get state from computer, it must be execute after GetOutputDestructive,
    // and before notify the notifiable.
    std::vector<NnetComputeState *> out_state;
    for (std::size_t i = 0; i < requests.size(); i++) {
      out_state.push_back(&(requests[i]->state));
    }
    computer.GetState(batch_first_, batch_size, &out_state);

    int32 num_outputs = outputs_chunk.NumRows() / batch_size;
    for (std::size_t i = 0; i < requests.size(); i++) {
      CuSubMatrix<BaseFloat> outputs = outputs_chunk.RowRange(i * num_outputs, 
                                                              num_outputs);
      requests[i]->notifiable->Receive(outputs);
    }
  }
}


DecodableNnetBatchLoopedOnline::DecodableNnetBatchLoopedOnline(
    NnetBatchLoopedComputer *computer,
    OnlineFeatureInterface *input_features,
    OnlineFeatureInterface *ivector_features):
    num_chunks_computed_(0),
    current_log_post_subsampled_offset_(-1),
    input_features_(input_features),
    ivector_features_(ivector_features),
    computer_(computer),
    info_(computer->GetInfo()),
    semaphone_(0) {
  KALDI_ASSERT(computer_ != NULL);
  // Check that feature dimensions match.
  KALDI_ASSERT(input_features_ != NULL);
  int32 nnet_input_dim = info_.nnet.InputDim("input"),
      nnet_ivector_dim = info_.nnet.InputDim("ivector"),
        feat_input_dim = input_features_->Dim(),
      feat_ivector_dim = ivector_features_ != NULL ?
                         ivector_features_->Dim() : 
                         -1;

  if (nnet_input_dim != feat_input_dim) {
    KALDI_ERR << "Input feature dimension mismatch: got " << feat_input_dim
              << " but network expects " << nnet_input_dim;
  }
  if (nnet_ivector_dim != feat_ivector_dim) {
    KALDI_ERR << "Ivector feature dimension mismatch: got " << feat_ivector_dim
              << " but network expects " << nnet_ivector_dim;
  }
}

int32 DecodableNnetBatchLoopedOnline::NumFramesReady() const {
  // note: the ivector_features_ may have 2 or 3 fewer frames ready than
  // input_features_, but we don't wait for them; we just use the most recent
  // iVector we can.
  int32 features_ready = input_features_->NumFramesReady();
  if (features_ready == 0)
    return 0;
  bool input_finished = input_features_->IsLastFrame(features_ready - 1);

  int32 sf = info_.opts.frame_subsampling_factor;

  if (input_finished) {
    // if the input has finished,... we'll pad with duplicates of the last frame
    // as needed to get the required right context.
    return (features_ready + sf - 1) / sf;
  } else {
    // note: info_.right_context_ includes both the model context and any
    // extra_right_context_ (but this
    int32 non_subsampled_output_frames_ready =
        std::max<int32>(0, features_ready - info_.frames_right_context);
    int32 num_chunks_ready = non_subsampled_output_frames_ready /
                             info_.frames_per_chunk;
    // note: the division by the frame subsampling factor 'sf' below
    // doesn't need any attention to rounding because info_.frames_per_chunk
    // is always a multiple of 'sf' (see 'frames_per_chunk = GetChunksize..."
    // in decodable-simple-looped.cc).
    return num_chunks_ready * info_.frames_per_chunk / sf;
  }
}

// note: the frame-index argument is on the output of the network, i.e. after any
// subsampling, so we call it 'subsampled_frame'.
bool DecodableNnetBatchLoopedOnline::IsLastFrame(
    int32 subsampled_frame) const {
  // To understand this code, compare it with the code of NumFramesReady(),
  // it follows the same structure.
  int32 features_ready = input_features_->NumFramesReady();
  if (features_ready == 0) {
    if (subsampled_frame == -1 && input_features_->IsLastFrame(-1)) {
      // the attempt to handle this rather pathological case (input finished
      // but no frames ready) is a little quixotic as we have not properly
      // tested this and other parts of the code may die.
      return true;
    } else {
      return false;
    }
  }
  bool input_finished = input_features_->IsLastFrame(features_ready - 1);
  if (!input_finished)
    return false;
  int32 sf = info_.opts.frame_subsampling_factor,
     num_subsampled_frames_ready = (features_ready + sf - 1) / sf;
  return (subsampled_frame == num_subsampled_frames_ready - 1);
}

void DecodableNnetBatchLoopedOnline::AdvanceChunk() {
  // Prepare the input data for the next chunk of features.
  // note: 'end' means one past the last.
  int32 begin_input_frame, end_input_frame;
  if (num_chunks_computed_ == 0) {
    begin_input_frame = -info_.frames_left_context;
    // note: end is last plus one.
    end_input_frame = info_.frames_per_chunk + info_.frames_right_context;
  } else {
    // note: begin_input_frame will be the same as the previous end_input_frame.
    // you can verify this directly if num_chunks_computed_ == 0, and then by
    // induction.
    begin_input_frame = num_chunks_computed_ * info_.frames_per_chunk +
                        info_.frames_right_context;
    end_input_frame = begin_input_frame + info_.frames_per_chunk;
  }

  int32 num_feature_frames_ready = input_features_->NumFramesReady();
  bool is_finished = input_features_->IsLastFrame(num_feature_frames_ready - 1);

  if (end_input_frame > num_feature_frames_ready && !is_finished) {
    // we shouldn't be attempting to read past the end of the available features
    // until we have reached the end of the input (i.e. the end-user called
    // InputFinished(), announcing that there is no more waveform; at this point
    // we pad as needed with copies of the last frame, to flush out the last of
    // the output.
    // If the following error happens, it likely indicates a bug in this
    // decodable code somewhere (although it could possibly indicate the
    // user asking for a frame that was not ready, which would be a misuse
    // of this class.. it can be figured out from gdb as in either case it
    // would be a bug in the code.
    KALDI_ERR << "Attempt to access frame past the end of the available input";
  }


  { // this block sets 'feats_chunk'.
    Matrix<BaseFloat> this_feats(end_input_frame - begin_input_frame,
                                 input_features_->Dim());
    for (int32 i = begin_input_frame; i < end_input_frame; i++) {
      SubVector<BaseFloat> this_row(this_feats, i - begin_input_frame);
      int32 input_frame = i;
      if (input_frame < 0) input_frame = 0;
      if (input_frame >= num_feature_frames_ready)
        input_frame = num_feature_frames_ready - 1;
      input_features_->GetFrame(input_frame, &this_row);
    }
    request_.inputs.Swap(&this_feats);
  }

  if (info_.has_ivectors) {
    KALDI_ASSERT(ivector_features_ != NULL);
    // all but the 1st chunk should have 1 iVector, but there is no need to
    // assume this.
    int32 num_ivectors = num_chunks_computed_ == 0 ?
			                   info_.num_chunk1_ivector_frames :
			                   info_.num_ivector_frames;
    KALDI_ASSERT(num_ivectors > 0);

    Vector<BaseFloat> ivector(ivector_features_->Dim());
    // we just get the iVector from the last input frame we needed,
    // reduced as necessary
    // we don't bother trying to be 'accurate' in getting the iVectors
    // for their 'correct' frames, because in general using the
    // iVector from as large 't' as possible will be better.

    int32 most_recent_input_frame = num_feature_frames_ready - 1;
    int32 num_ivector_frames_ready = ivector_features_->NumFramesReady();

    if (num_ivector_frames_ready > 0) {
      int32 ivector_frame_to_use = std::min<int32>(
          most_recent_input_frame, num_ivector_frames_ready - 1);
      ivector_features_->GetFrame(ivector_frame_to_use,
                                  &ivector);
    }
    // else just leave the iVector zero (would only happen with very small
    // chunk-size, like a chunk size of 2 which would be very inefficient; and
    // only at file begin.

    // note: we expect num_ivectors to be 1 in practice.
    Matrix<BaseFloat> ivectors(num_ivectors, ivector.Dim());
    ivectors.CopyRowsFromVec(ivector);
    request_.ivectors.Swap(&ivectors);
  }

  request_.notifiable = this;
  if (num_chunks_computed_ == 0) {
    request_.first_chunk = true;
    request_.state.matrices.clear();
  }
  else {
    request_.first_chunk = false;
  }

  computer_->Enqueue(&request_);

  semaphone_.Wait();
  KALDI_ASSERT(current_log_post_.NumRows() == info_.frames_per_chunk /
               info_.opts.frame_subsampling_factor &&
               current_log_post_.NumCols() == info_.output_dim);

  num_chunks_computed_++;

  current_log_post_subsampled_offset_ =
      (num_chunks_computed_ - 1) *
      (info_.frames_per_chunk / info_.opts.frame_subsampling_factor);
}

BaseFloat DecodableNnetBatchLoopedOnline::LogLikelihood(int32 subsampled_frame,
                                                        int32 index) {
  EnsureFrameIsComputed(subsampled_frame);
  return current_log_post_(
      subsampled_frame - current_log_post_subsampled_offset_,
      index);
}

void DecodableNnetBatchLoopedOnline::GetOutputForFrame(
    int32 subsampled_frame, 
    VectorBase<BaseFloat> *output) {
  EnsureFrameIsComputed(subsampled_frame);
  output->CopyFromVec(current_log_post_.Row(
        subsampled_frame - current_log_post_subsampled_offset_));
}

DecodableAmNnetBatchLoopedOnline::DecodableAmNnetBatchLoopedOnline(
    NnetBatchLoopedComputer *computer,
    const TransitionModel &trans_model,
    OnlineFeatureInterface *input_features,
    OnlineFeatureInterface *ivector_features):
    decodable_nnet_(computer, input_features, ivector_features),
    trans_model_(trans_model) { }

BaseFloat DecodableAmNnetBatchLoopedOnline::LogLikelihood(
    int32 frame,
    int32 transition_id) {
  int32 pdf_id = trans_model_.TransitionIdToPdfFast(transition_id);
  return decodable_nnet_.LogLikelihood(frame, pdf_id);
}

DecodableNnetBatchSimpleLooped::DecodableNnetBatchSimpleLooped(
    NnetBatchLoopedComputer *computer,
    const MatrixBase<BaseFloat> &feats,
    const VectorBase<BaseFloat> *ivector,
    const MatrixBase<BaseFloat> *online_ivectors,
    int32 online_ivector_period):
    num_chunks_computed_(0),
    current_log_post_subsampled_offset_(-1),
    feats_(feats),
    ivector_(ivector),
    online_ivector_feats_(online_ivectors),
    online_ivector_period_(online_ivector_period),
    computer_(computer),
    info_(computer->GetInfo()),
    semaphone_(0) {
  num_subsampled_frames_ = 
    (feats_.NumRows() + info_.opts.frame_subsampling_factor - 1) /
    info_.opts.frame_subsampling_factor;
  KALDI_ASSERT(computer != NULL);
  KALDI_ASSERT(!(ivector != NULL && online_ivectors != NULL));
  KALDI_ASSERT(!(online_ivectors != NULL && online_ivector_period <= 0 &&
                 "You need to set the --online-ivector-period option!"));
}

void DecodableNnetBatchSimpleLooped::AdvanceChunk() {
  // Prepare the input data for the next chunk of features.
  // note: 'end' means one past the last.
  int32 begin_input_frame, end_input_frame;
  if (num_chunks_computed_ == 0) {
    begin_input_frame = -info_.frames_left_context;
    // note: end is last plus one.
    end_input_frame = info_.frames_per_chunk + info_.frames_right_context;
  } else {
    // note: begin_input_frame will be the same as the previous end_input_frame.
    // you can verify this directly if num_chunks_computed_ == 0, and then by
    // induction.
    begin_input_frame = num_chunks_computed_ * info_.frames_per_chunk +
                        info_.frames_right_context;
    end_input_frame = begin_input_frame + info_.frames_per_chunk;
  }
 
  Matrix<BaseFloat> feats_chunk(end_input_frame - begin_input_frame,
                                feats_.NumCols(), kUndefined);
  int32 num_features = feats_.NumRows();
  if (begin_input_frame >= 0 && end_input_frame <= num_features) {
    SubMatrix<BaseFloat> this_feats(feats_,
                                    begin_input_frame, 
                                    end_input_frame - begin_input_frame,
                                    0, feats_.NumCols());
    feats_chunk.CopyFromMat(this_feats);
  } else {
    for (int32 r = begin_input_frame; r < end_input_frame; r++) {
      int32 input_frame = r;
      if (input_frame < 0) input_frame = 0;
      if (input_frame >= num_features) input_frame = num_features - 1;
      feats_chunk.Row(r - begin_input_frame).CopyFromVec(feats_.Row(input_frame));
    }
  }
  request_.inputs.Swap(&feats_chunk);

  if (info_.has_ivectors) {
    // all but the 1st chunk should have 1 iVector, but there is no need to
    // assume this.
    int32 num_ivectors = num_chunks_computed_ == 0 ?
			                   info_.num_chunk1_ivector_frames :
			                   info_.num_ivector_frames;
    KALDI_ASSERT(num_ivectors > 0);

    Vector<BaseFloat> ivector;
    // we just get the iVector from the last input frame we needed...
    // we don't bother trying to be 'accurate' in getting the iVectors
    // for their 'correct' frames, because in general using the
    // iVector from as large 't' as possible will be better.
    GetCurrentIvector(end_input_frame, &ivector);
    Matrix<BaseFloat> ivectors(num_ivectors, ivector.Dim());
    ivectors.CopyRowsFromVec(ivector);
    request_.ivectors.Swap(&ivectors);
  }

  request_.notifiable = this;
  if (num_chunks_computed_ == 0) {
    request_.first_chunk = true;
    request_.state.matrices.clear();
  }
  else {
    request_.first_chunk = false;
  }

  computer_->Enqueue(&request_);

  semaphone_.Wait();
  KALDI_ASSERT(current_log_post_.NumRows() == info_.frames_per_chunk /
               info_.opts.frame_subsampling_factor &&
               current_log_post_.NumCols() == info_.output_dim);

  num_chunks_computed_++;

  current_log_post_subsampled_offset_ =
      (num_chunks_computed_ - 1) *
      (info_.frames_per_chunk / info_.opts.frame_subsampling_factor);
}

void DecodableNnetBatchSimpleLooped::GetCurrentIvector(
    int32 input_frame, 
    Vector<BaseFloat> *ivector) {
  if (!info_.has_ivectors)
    return;
  if (ivector_ != NULL) {
    *ivector = *ivector_;
    return; 
  } else if (online_ivector_feats_ == NULL) {
    KALDI_ERR << "Neural net expects iVectors but none provided.";
  }
  KALDI_ASSERT(online_ivector_period_ > 0);
  int32 ivector_frame = input_frame / online_ivector_period_;
  KALDI_ASSERT(ivector_frame >= 0);
  if (ivector_frame >= online_ivector_feats_->NumRows())
    ivector_frame = online_ivector_feats_->NumRows() - 1;
  KALDI_ASSERT(ivector_frame >= 0 && "ivector matrix cannot be empty.");
  *ivector = online_ivector_feats_->Row(ivector_frame);
}

DecodableAmNnetBatchSimpleLooped::DecodableAmNnetBatchSimpleLooped(
    NnetBatchLoopedComputer *computer,
    const TransitionModel &trans_model,
    const MatrixBase<BaseFloat> &feats,
    const VectorBase<BaseFloat> *ivector,
    const MatrixBase<BaseFloat> *online_ivectors,
    int32 online_ivector_period):
    decodable_nnet_(computer, feats, ivector, online_ivectors, online_ivector_period),
    trans_model_(trans_model) { }

BaseFloat DecodableAmNnetBatchSimpleLooped::LogLikelihood(
    int32 frame,
    int32 transition_id) {
  int32 pdf_id = trans_model_.TransitionIdToPdfFast(transition_id);
  return decodable_nnet_.GetOutput(frame, pdf_id);
}

DecodableAmNnetBatchSimpleLoopedParallel::DecodableAmNnetBatchSimpleLoopedParallel(
    NnetBatchLoopedComputer *computer,
    const TransitionModel &trans_model,
    const MatrixBase<BaseFloat> &feats,
    const VectorBase<BaseFloat> *ivector,
    const MatrixBase<BaseFloat> *online_ivectors,
    int32 online_ivector_period):
    decodable_nnet_(NULL),
    trans_model_(trans_model), 
    feats_copy_(feats),
    ivector_copy_(NULL),
    online_ivectors_copy_(NULL) { 
  try {
    if (ivector != NULL) 
      ivector_copy_ = new Vector<BaseFloat>(*ivector);
    if (online_ivectors != NULL)
      online_ivectors_copy_ = new Matrix<BaseFloat>(*online_ivectors);

    decodable_nnet_ = new DecodableNnetBatchSimpleLooped(
        computer, feats_copy_, ivector_copy_, 
        online_ivectors_copy_, online_ivector_period);
  } catch (...) {
    DeletePointers();
    KALDI_ERR << "Error occurred in constructor (see above)";
  }
}

void DecodableAmNnetBatchSimpleLoopedParallel::DeletePointers() {
  // delete[] does nothing for null pointers, so we have no checks.
  delete decodable_nnet_;
  decodable_nnet_ = NULL;
  delete ivector_copy_;
  ivector_copy_ = NULL;
  delete online_ivectors_copy_;
  online_ivectors_copy_ = NULL;
}


BaseFloat DecodableAmNnetBatchSimpleLoopedParallel::LogLikelihood(
    int32 frame,
    int32 transition_id) {
  int32 pdf_id = trans_model_.TransitionIdToPdfFast(transition_id);
  return decodable_nnet_->GetOutput(frame, pdf_id);
}

} // namespace nnet3
} // namespace kaldi
#endif
