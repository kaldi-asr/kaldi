// nnet3/nnet-batch-compute.cc

// Copyright 2012-2018  Johns Hopkins University (author: Daniel Povey)
//           2018       Hang Lyu

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

#include <algorithm>
#include "nnet3/nnet-batch-compute.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {


NnetBatchComputer::NnetBatchComputer(
    const NnetBatchComputerOptions &opts,
    const Nnet &nnet,
    const VectorBase<BaseFloat> &priors):
    opts_(opts),
    nnet_(nnet),
    log_priors_(priors),
    compiler_(nnet_, opts.optimize_config) {
  log_priors_.ApplyLog();
  CheckAndFixConfigs();
}

std::shared_ptr<const NnetComputation> NnetBatchComputer::FindHighestPriorityGroup(
    ComputationGroupKey *key,
    std::vector<NnetComputationTask*> *tasks) {
  tasks->clear();
  std::unique_lock<std::mutex>(mutex_);
  MapType::const_iterator iter = tasks_.begin(), end = tasks_.end();

}


std::shared_ptr<const NnetComputation> NnetBatchComputer::GetComputation(
    const ComputationGroupKey &key,
    int32 minibatch_size) {
  MapType::iterator iter = tasks_.Find(key);
  if (iter == tasks_.end() || iter->second.tasks.empty()) {
    KALDI_ERR << "Expected to have at least one example to compile for.";
  }
  NnetComputationTask *example_task = iter->second.tasks[0];

  NnetComputationRequest request;
  GetComputationRequest(*example_task, minibatch_size, &request);
  return compiler_.Compile(request);
}

// static
void NnetBatchComputer::GetComputationRequest(
    const NnetComputationTask &task,
    int32 minibatch_size,
    ComputationRequest *request) {
  request->need_model_derivative = false;
  request->store_component_stats = false;
  request->inputs.reserve(2);

  int32 num_input_frames = task.input_matrix.NumRows(),
      first_input_t = task.first_input_t,
      num_output_frames = task.num_output_frames,
      output_t_stride = task.output_t_stride;
  bool has_ivector = (task.ivector.Dim() != 0);

  std::vector<Index> input_indexes, ivector_indexes, output_indexes;
  input_indexes.reserve(minibatch_size * num_input_frames);
  output_indexes.reserve(minibatch_size * num_output_frames);
  if (has_ivector)
    ivector_indexes.reserve(minibatch_size);

  for (int32 n = 0; n < minibatch_size; n++) {
    for (int32 t = first_input_t; t < first_input_t + num_input_rows; t++)
      input_indexes.push_back(Index(n, t, 0));
    if (has_ivector)
      ivector_indexes.push_back(Index(n, 0, 0));
    for (int32 t = 0; t < num_output_rows; t++)
      output_indexes.push_back(Index(n, t, 0));
  }

  request->inputs.push_back(IoSpecification("input", input_indexes));
  if (has_ivector)
    request->inputs.push_back(IoSpecification("ivector", ivector_indexes));
  request->outputs.push_back(IoSpecification("output", output_indexes));
}



void NnetBatchComputer::CheckAndFixConfigs() {
  static bool warned_frames_per_chunk = false;
  int32 nnet_modulus = nnet_.Modulus();
  if (opts_.frame_subsampling_factor < 1 ||
      opts_.frames_per_chunk < 1) {
    KALDI_ERR << "--frame-subsampling-factor and "
              << "--frames-per-chunk must be > 0";
  }
  KALDI_ASSERT(nnet_modulus > 0);
  int32 n = Lcm(opts_.frame_subsampling_factor, nnet_modulus);

  if (opts_.frames_per_chunk % n != 0) {
    // round up to the nearest multiple of n.
    int32 frames_per_chunk = n * ((opts_.frames_per_chunk + n - 1) / n);
    if (!warned_frames_per_chunk) {
      warned_frames_per_chunk = true;
      if (nnet_modulus == 1) {
        // simpler error message.
        KALDI_LOG << "Increasing --frames-per-chunk from "
                  << opts_.frames_per_chunk << " to "
                  << frames_per_chunk << " to make it a multiple of "
                  << "--frame-subsampling-factor="
                  << opts_.frame_subsampling_factor;
      } else {
        KALDI_LOG << "Increasing --frames-per-chunk from "
                  << opts_.frames_per_chunk << " to "
                  << frames_per_chunk << " due to "
                  << "--frame-subsampling-factor="
                  << opts_.frame_subsampling_factor << " and "
                  << "nnet shift-invariance modulus = " << nnet_modulus;
      }
    }
    opts_.frames_per_chunk = frames_per_chunk;
  }
}

int32 NnetBatchComputer::NumTasksQueued() const {
  std::unique_lock<std::mutex> lock(mutex_);
  MapType::const_iterator iter = tasks_.begin(), end = tasks_.end();
  int32 ans = 0;
  for (; iter != end; ++iter) {
    const ComputationGroupInfo &info = iter->second;
    ans += static_cast<int32>(info.tasks.size());
  }
  return ans;
}

bool NnetBatchComputer::FullMinibatchReady() const {
  std::unique_lock<std::mutex> lock(mutex_);
  MapType::const_iterator iter = tasks_.begin(), end = tasks_.end();
  int32 ans = 0;
  for (; iter != end; ++iter) {
    const ComputationGroupInfo &info = iter->second;
    if (info.tasks.empty())
      continue;
    // 'is_edge' is true only if this task is for the beginning or end of an
    // utterance, *AND* it's structurally different from the non-edge tasks
    // (e.g. due to extra_left_context_initial or extra_right_initial).
    bool is_edge = info.tasks[0]->is_edge;
    int32 this_minibatch_size = (is_edge ? opts_.edge_minibatch_size :
                                 opts_.minibatch_size);
    if (static_cast<int32>(info.tasks.size()) >= this_minibatch_size)
      return true;
  }
  return false;
}

bool NnetBatchComputer::Compute() {



}


void BatchNnetComputer::AcceptInput(
    const std::string &utt_id,
    const Matrix<BaseFloat> *feats,
    const Vector<BaseFloat> *ivector,
    const Matrix<BaseFloat> *online_ivectors) {
  // Check the input fits with the nnet.
  CheckInput(feats, ivector, online_ivectors);

  utt_list_.push_back(utt_id);
  feats_[utt_id] = feats;
  if ( ivector != NULL ) {
    ivectors_[utt_id] = ivector;
  }
  if ( online_ivectors != NULL ) {
    online_ivector_feats_[utt_id] = online_ivectors;
  }
  is_computed_[utt_id] = false;

  // Compute number of output frames
  int32 cur_num_subsampled_frames =
    (feats->NumRows() + opts_.frame_subsampling_factor - 1) /
    opts_.frame_subsampling_factor;

  num_subsampled_frames_[utt_id] = cur_num_subsampled_frames;

  // Allocate storage space for output
  Matrix<BaseFloat> *log_post = new Matrix<BaseFloat>(
      cur_num_subsampled_frames, output_dim_);
  log_post_[utt_id] = log_post;

  PrepareBatchInfo();
}


void BatchNnetComputer::CheckInput(const Matrix<BaseFloat> *feats,
                                   const Vector<BaseFloat> *ivector,
                                   const Matrix<BaseFloat> *online_ivectors) {
  KALDI_ASSERT(!(ivector != NULL && online_ivectors != NULL));
  KALDI_ASSERT(!(online_ivectors != NULL && online_ivector_period_ <= 0 &&
                 "You need to set the --online-ivector-period option!"));
  int32 feature_dim = feats->NumCols();
  int32 ivector_dim = 0;
  if (ivector != NULL) {
    ivector_dim = ivector->Dim();
  }
  if (online_ivectors != NULL) {
    ivector_dim = online_ivectors->NumCols();
  }
  int32 nnet_input_dim = nnet_.InputDim("input"),
        nnet_ivector_dim = std::max<int32>(0, nnet_.InputDim("ivector"));
  if (feature_dim != nnet_input_dim)
    KALDI_ERR << "Neural net expects 'input' features with dimension "
              << nnet_input_dim << " but you provided "
              << feature_dim;
  if (ivector_dim != std::max<int32>(0, nnet_.InputDim("ivector")))
    KALDI_ERR << "Neural net expects 'ivector' features with dimension "
              << nnet_ivector_dim << " but you provided " << ivector_dim;
}


void BatchNnetComputer::PrepareBatchInfo() {
  if (Ready()) {
    return;
  }
  std::string utt_id = last_batch_info_.utt_id;
  int32 first_subsampled_frame, last_subsampled_frame;

  // Prepare the utt_id and start_subsampled_frame
  if (utt_id.empty() ||
      num_subsampled_frames_.find(utt_id) == num_subsampled_frames_.end()) {
    KALDI_ASSERT(utt_list_.size() == 1);
    utt_id = utt_list_.front();
    first_subsampled_frame = 0;
  } else {
    int32 past_last_subsampled_frame =
      last_batch_info_.last_output_subsampled_frame_index;
    int32 num_subsampled_frames = (num_subsampled_frames_.find(utt_id))->second;
    if ( past_last_subsampled_frame == num_subsampled_frames -1 ) {
      // Current utterance has been processed.
      if (utt_id == utt_list_.back()) {
        // Current utterance is the last one.
        return;
      } else {
        std::list<std::string>::iterator iter =
          std::find(utt_list_.begin(), utt_list_.end(), utt_id);
        utt_id = *(++iter);
        first_subsampled_frame = 0;
      }
    } else {
      first_subsampled_frame = past_last_subsampled_frame + 1;
    }
  }

  int32 subsampling_factor = opts_.frame_subsampling_factor,
        subsampled_frames_per_chunk = opts_.frames_per_chunk /
                                      subsampling_factor,
        num_subsampled_frames = num_subsampled_frames_.find(utt_id)->second;
  KALDI_ASSERT(num_subsampled_frames > 0);

  for (; first_subsampled_frame < num_subsampled_frames && !Ready();
        first_subsampled_frame = last_subsampled_frame + 1) {
    int32 extra_left_context = opts_.extra_left_context,
          extra_right_context = opts_.extra_right_context;
    // Prepare first_subsampled_frame, last_subsampled_frame. They are the
    // indexes of the output matrix.
    int32 cur_num_subsampled_frames =
      std::min<int32>(subsampled_frames_per_chunk,
                      num_subsampled_frames - first_subsampled_frame);
    last_subsampled_frame = first_subsampled_frame +
      cur_num_subsampled_frames - 1;

    int32 output_offset = subsampled_frames_per_chunk -
      cur_num_subsampled_frames;
    // Prepare first_input_frame, last_input_frame
    int32 first_output_frame = first_subsampled_frame * subsampling_factor,
          last_output_frame = last_subsampled_frame * subsampling_factor;

    if ( first_output_frame == 0 && opts_.extra_left_context_initial >= 0 ) {
      extra_left_context = opts_.extra_left_context_initial;
    }
    if (last_subsampled_frame == num_subsampled_frames - 1 &&
        opts_.extra_right_context_final >= 0) {
      extra_right_context = opts_.extra_right_context_final;
    }
    // If ensure_exact_final_context_ is false, the "shorter than chunk size"
    // case will be arranged in "(extra_left_context_initial,
    // extra_right_context)" batch type.
    if (!ensure_exact_final_context_ &&
        first_output_frame == 0 &&
        last_subsampled_frame == num_subsampled_frames - 1 ) {
      extra_right_context = opts_.extra_right_context;
    }

    int32 left_context = nnet_left_context_ + extra_left_context;
    int32 right_context = nnet_right_context_ + extra_right_context;

    // first_input_frame can overlap with previous chunk
    int32 last_input_frame = last_output_frame + right_context;
    int32 first_input_frame = last_input_frame +
      opts_.frame_subsampling_factor - right_context -
      opts_.frames_per_chunk - left_context;

    // "shorter than chunk size" utterance case
    if (ensure_exact_final_context_ && first_output_frame == 0 &&
        last_subsampled_frame == num_subsampled_frames - 1 ) {
      first_input_frame = first_output_frame - left_context;
      left_context = -1;
      right_context = -1;
    }

    std::pair<int32, int32> context(left_context, right_context);

    // Update class private member
    last_batch_info_.utt_id = utt_id;
    last_batch_info_.first_input_frame_index = first_input_frame;
    last_batch_info_.last_input_frame_index = last_input_frame;
    last_batch_info_.first_output_subsampled_frame_index =
      first_subsampled_frame;
    last_batch_info_.last_output_subsampled_frame_index =
      last_subsampled_frame;
    last_batch_info_.output_offset = output_offset;
    (batch_info_[context])->push_back(last_batch_info_);
  }
  // recursive invocation.
  PrepareBatchInfo();
}


void BatchNnetComputer::DoNnetComputationOnes() {
  std::pair<int32, int32> current_context(-1, -1);
  if (batch_info_[current_context]->size() == 0) {
    return;
  }
  BatchInfoQueue* current_queue = batch_info_[current_context];
  std::string utt_id;
  int32 first_input_frame, last_input_frame;
  int32 first_subsampled_frame, last_subsampled_frame;

  // Prepare ComputationRequest
  ComputationRequest request;
  request.need_model_derivative = false;
  request.store_component_stats = false;
  request.inputs.reserve(2);
  request.outputs.reserve(1);
  std::vector<Index> input_indexes, ivector_indexes, output_indexes;
  int32 ivector_dim = nnet_.InputDim("ivector");

  int32 tot_input_rows = 0;
  int32 extra_left_context = opts_.extra_left_context;
  if (opts_.extra_left_context_initial >= 0) {
    extra_left_context = opts_.extra_left_context_initial;
  }
  int32 left_context = nnet_left_context_ + extra_left_context;

  BatchInfoQueue::iterator iter = current_queue->begin();
  for (int32 n = 0; iter != current_queue->end(); iter++, n++) {
    utt_id = iter->utt_id;
    first_input_frame = iter->first_input_frame_index;
    last_input_frame = iter->last_input_frame_index;
    first_subsampled_frame = iter->first_output_subsampled_frame_index;
    last_subsampled_frame = iter->last_output_subsampled_frame_index;

    int32 num_input_rows = last_input_frame - first_input_frame + 1;
    int32 num_output_rows = last_subsampled_frame - first_subsampled_frame + 1;
    tot_input_rows += num_input_rows;

    for (int32 t = 0; t < num_input_rows; t++) {
      input_indexes.push_back(Index(n, t - left_context, 0));
    }
    if (ivector_dim > 0) {
      ivector_indexes.push_back(Index(n, 0, 0));
    }
    for (int32 i = 0; i < num_output_rows; i++) {
      output_indexes.push_back(Index(n, i * opts_.frame_subsampling_factor, 0));
    }
  }
  request.inputs.push_back(IoSpecification("input", input_indexes));
  if (ivector_dim > 0) {
    request.inputs.push_back(IoSpecification("ivector", ivector_indexes));
  }
  request.outputs.push_back(IoSpecification("output", output_indexes));
  // Prepare Data
  Matrix<BaseFloat> tot_input(tot_input_rows, nnet_.InputDim("input"),
                              kSetZero);
  Matrix<BaseFloat> tot_ivector;
  if (ivector_dim > 0) {
    tot_ivector.Resize(current_queue->size(), ivector_dim, kSetZero);
  }
  BatchInfoQueue::iterator iter_prep = current_queue->begin();
  int32 input_count = 0;
  for (int32 n = 0; iter_prep != current_queue->end(); iter_prep++, n++) {
    utt_id = iter_prep->utt_id;
    first_input_frame = iter_prep->first_input_frame_index;
    last_input_frame = iter_prep->last_input_frame_index;
    first_subsampled_frame = iter_prep->first_output_subsampled_frame_index;
    last_subsampled_frame = iter_prep->last_output_subsampled_frame_index;

    std::unordered_map<std::string, const Matrix<BaseFloat>*,
                       StringHasher>::iterator feats_iter;
    feats_iter = feats_.find(utt_id);
    int32 num_input_frames = last_input_frame - first_input_frame + 1;
    if (first_input_frame >= 0 &&
      last_input_frame < (feats_iter->second)->NumRows()) {
      tot_input.RowRange(input_count, num_input_frames).CopyFromMat(
        (feats_iter->second)->RowRange(first_input_frame, num_input_frames));
    } else {
      int32 tot_input_feats = (feats_iter->second)->NumRows();
      for (int32 i = 0; i < num_input_frames; i++) {
        SubVector<BaseFloat> dest(tot_input, input_count + i);
        int32 t = i + first_input_frame;
        if (t < 0) t = 0;
        if (t >= tot_input_feats) t = tot_input_feats - 1;
        const SubVector<BaseFloat> src(*(feats_iter->second), t);
        dest.CopyFromVec(src);
      }
    }
    // Update ivector matrix
    // If the nnet_ doesn't have ivector, nothing will be returned by
    // GetCurrentIvector. So the ivector.Dim() == 0, and the tot_ivector will
    // not be filled.
    int32 first_output_frame =
      first_subsampled_frame * opts_.frame_subsampling_factor,
          last_output_frame =
      last_subsampled_frame * opts_.frame_subsampling_factor;
    Vector<BaseFloat> ivector;
    GetCurrentIvector(utt_id, first_output_frame,
                      last_output_frame - first_output_frame, &ivector);
    if (ivector.Dim() != 0) {
      tot_ivector.Row(n).CopyFromVec(ivector);
    }
    input_count += num_input_frames;
  }
  // Compute
  std::shared_ptr<const NnetComputation> computation =
    compiler_.Compile(request);
  Nnet *nnet_to_update = NULL;  // we're not doing any update
  NnetComputer computer(opts_.compute_config, *computation,
                        nnet_, nnet_to_update);
  CuMatrix<BaseFloat> input_feats_cu(tot_input);
  computer.AcceptInput("input", &input_feats_cu);
  CuMatrix<BaseFloat> ivector_feats_cu;
  // tot_ivector.NumCols() == 0 means that nnet_ doesn't have ivector
  if (tot_ivector.NumCols() != 0) {
    ivector_feats_cu.Resize(tot_ivector.NumRows(), tot_ivector.NumCols());
    ivector_feats_cu.CopyFromMat(tot_ivector);
    computer.AcceptInput("ivector", &ivector_feats_cu);
  }
  computer.Run();
  CuMatrix<BaseFloat> cu_output;
  computer.GetOutputDestructive("output", &cu_output);
  // Get Output
  if (log_priors_.Dim() != 0) {
    cu_output.AddVecToRows(-1.0, log_priors_);
  }
  cu_output.Scale(opts_.acoustic_scale);
  int32 output_count = 0;
  BatchInfoQueue::iterator iter_out = current_queue->begin();
  for (; iter_out != current_queue->end(); iter_out++) {
    utt_id = iter_out->utt_id;
    first_input_frame = iter_out->first_input_frame_index;
    last_input_frame = iter_out->last_input_frame_index;
    first_subsampled_frame = iter_out->first_output_subsampled_frame_index;
    last_subsampled_frame = iter_out->last_output_subsampled_frame_index;

    std::unordered_map<std::string, Matrix<BaseFloat>*,
                       StringHasher>::iterator output_iter;
    output_iter = log_post_.find(utt_id);
    int32 num_rows = last_subsampled_frame - first_subsampled_frame + 1;
    (output_iter->second)->RowRange(first_subsampled_frame,
        num_rows).CopyFromMat(cu_output.RowRange(output_count, num_rows));
    output_count += num_rows;
    if ((last_subsampled_frame + 1) ==
         num_subsampled_frames_.find(utt_id)->second) {
      is_computed_.find(utt_id)->second = true;
    }
  }
  // Clear
  batch_info_[current_context]->clear();

  // comment out may help when you find the ComputationRequest in
  // contest_to_request_ is compiled more than one time. As when
  // CachingOptimizingCompiler compile a request, it will be moved to the end
  // of access_queue_.
  // ComputationRequestMap::iterator iter = context_to_request_.begin(),
  //                                  end = context_to_request_.end();
  // for (; iter != end; iter++) {
  //   compiler_.Compile(*(iter->second));
  // }
}


void BatchNnetComputer::DoNnetComputation() {
  int32 ivector_dim = nnet_.InputDim("ivector");
  // Use the index of context_to_request_ to do loop
  ComputationRequestMap::iterator iter;
  for (iter = context_to_request_.begin(); iter != context_to_request_.end();
       iter++) {
    std::pair<int32, int32> current_context = iter->first;
    if (batch_info_[current_context]->size() == 0) {
      break;
    }

    BatchInfoQueue* current_queue = batch_info_[current_context];
    std::string utt_id;
    int32 first_input_frame, last_input_frame;
    int32 first_subsampled_frame, last_subsampled_frame;
    int32 output_offset;

    int32 num_input_rows = current_context.first + opts_.frames_per_chunk +
                           current_context.second -
                           opts_.frame_subsampling_factor + 1;
    Matrix<BaseFloat> tot_input(num_input_rows * minibatch_size_,
                                nnet_.InputDim("input"), kSetZero);
    Matrix<BaseFloat> tot_ivector;
    if (ivector_dim > 0) {
      tot_ivector.Resize(minibatch_size_, ivector_dim, kSetZero);
    }
    // Preapre data
    BatchInfoQueue::iterator iter = current_queue->begin();
    for (int32 n = 0; iter != current_queue->end(); iter++, n++) {
      utt_id = iter->utt_id;
      first_input_frame = iter->first_input_frame_index;
      last_input_frame = iter->last_input_frame_index;
      first_subsampled_frame = iter->first_output_subsampled_frame_index;
      last_subsampled_frame = iter->last_output_subsampled_frame_index;
      output_offset = iter->output_offset;

      std::unordered_map<std::string, const Matrix<BaseFloat>*,
                         StringHasher>::iterator feats_iter;
      feats_iter = feats_.find(utt_id);
      int32 num_input_frames = last_input_frame - first_input_frame + 1;
      KALDI_ASSERT(num_input_frames == num_input_rows);
      if (first_input_frame >= 0 &&
          last_input_frame < (feats_iter->second)->NumRows()) {
        tot_input.RowRange(n * num_input_rows, num_input_frames).CopyFromMat(
          (feats_iter->second)->RowRange(first_input_frame, num_input_frames));
      } else {
        int32 tot_input_feats = (feats_iter->second)->NumRows();
        for (int32 i = 0; i < num_input_frames; i++) {
          SubVector<BaseFloat> dest(tot_input, n * num_input_frames + i);
          int32 t = i + first_input_frame;
          if (t < 0) t = 0;
          if (t >= tot_input_feats) t = tot_input_feats - 1;
          const SubVector<BaseFloat> src(*(feats_iter->second), t);
          dest.CopyFromVec(src);
        }
      }
      // Update ivector matrix
      // If the nnet_ doesn't have ivector, nothing will be returned by
      // GetCurrentIvector. So the ivector.Dim() == 0, and the tot_ivector will
      // not be filled.
      int32 first_output_frame =
        first_subsampled_frame * opts_.frame_subsampling_factor,
            last_output_frame =
        last_subsampled_frame * opts_.frame_subsampling_factor;
      Vector<BaseFloat> ivector;
      GetCurrentIvector(utt_id, first_output_frame,
                        last_output_frame - first_output_frame, &ivector);
      if (ivector.Dim() != 0) {
        tot_ivector.Row(n).CopyFromVec(ivector);
      }
    }
    // Compute
    std::shared_ptr<const NnetComputation> computation =
      compiler_.Compile(*(context_to_request_[current_context]));
    Nnet *nnet_to_update = NULL;  // we're not doing any update
    NnetComputer computer(opts_.compute_config, *computation,
                          nnet_, nnet_to_update);
    CuMatrix<BaseFloat> input_feats_cu(tot_input);
    computer.AcceptInput("input", &input_feats_cu);
    CuMatrix<BaseFloat> ivector_feats_cu;
    // tot_ivector.NumCols() == 0 means that nnet_ doesn't have ivector
    if (tot_ivector.NumCols() != 0) {
      ivector_feats_cu.Resize(tot_ivector.NumRows(), tot_ivector.NumCols());
      ivector_feats_cu.CopyFromMat(tot_ivector);
      computer.AcceptInput("ivector", &ivector_feats_cu);
    }
    computer.Run();
    CuMatrix<BaseFloat> cu_output;
    computer.GetOutputDestructive("output", &cu_output);
    // Get Output
    if (log_priors_.Dim() != 0) {
      cu_output.AddVecToRows(-1.0, log_priors_);
    }
    cu_output.Scale(opts_.acoustic_scale);

    int32 num_batch_output_rows = opts_.frames_per_chunk /
                                  opts_.frame_subsampling_factor;
    BatchInfoQueue::iterator iter_out = current_queue->begin();
    for (int32 n = 0; iter_out != current_queue->end(); iter_out++, n++) {
      utt_id = iter_out->utt_id;
      first_input_frame = iter_out->first_input_frame_index;
      last_input_frame = iter_out->last_input_frame_index;
      first_subsampled_frame = iter_out->first_output_subsampled_frame_index;
      last_subsampled_frame = iter_out->last_output_subsampled_frame_index;
      output_offset = iter_out->output_offset;

      std::unordered_map<std::string, Matrix<BaseFloat>*,
                         StringHasher>::iterator output_iter;
      output_iter = log_post_.find(utt_id);
      int32 num_rows = last_subsampled_frame - first_subsampled_frame + 1;

      (output_iter->second)->RowRange(first_subsampled_frame,
          num_rows).CopyFromMat(cu_output.RowRange(
              n * num_batch_output_rows + output_offset, num_rows));
      if ((last_subsampled_frame + 1) ==
           num_subsampled_frames_.find(utt_id)->second) {
        is_computed_.find(utt_id)->second = true;
      }
    }
    // Clear
    batch_info_[current_context]->clear();
  }
}


void BatchNnetComputer::GetCurrentIvector(std::string utt_id,
                                          int32 output_t_start,
                                          int32 num_output_frames,
                                          Vector<BaseFloat> *ivector) {
  if (ivectors_.find(utt_id) != ivectors_.end()) {
    *ivector = *(ivectors_.find(utt_id)->second);
    return;
  } else if (online_ivector_feats_.find(utt_id) ==
             online_ivector_feats_.end()) {
    return;
  }
  std::unordered_map<std::string, const Matrix<BaseFloat>*,
                     StringHasher>::iterator iter;
  iter = online_ivector_feats_.find(utt_id);
  KALDI_ASSERT(online_ivector_period_ > 0);
  // frame_to_search is the frame that we want to get the most recent iVector
  // for.  We choose a point near the middle of the current window, the concept
  // being that this is the fairest comparison to nnet2.   Obviously we could do
  // better by always taking the last frame's iVector, but decoding with
  // 'online' ivectors is only really a mechanism to simulate online operation.
  int32 frame_to_search = output_t_start + num_output_frames / 2;
  int32 ivector_frame = frame_to_search / online_ivector_period_;
  KALDI_ASSERT(ivector_frame >= 0);
  if (ivector_frame >= (iter->second)->NumRows()) {
    int32 margin = ivector_frame - ((iter->second)->NumRows() - 1);
    if (margin * online_ivector_period_ > 50) {
      // Half a second seems like too long to be explainable as edge effects.
      KALDI_ERR << "Could not get iVector for frame " << frame_to_search
                << ", only available till frame "
                << (iter->second)->NumRows()
                << " * ivector-period=" << online_ivector_period_
                << " (mismatched --ivector-period?)";
    }
    ivector_frame = (iter->second)->NumRows() - 1;
  }
  *ivector = (iter->second)->Row(ivector_frame);
}


bool BatchNnetComputer::GetFinishedUtterance(std::string *uttid,
                                             Matrix<BaseFloat> *output_matrix) {
  if (!utt_list_.empty()) {
    std::string utt_id = utt_list_.front();
    if (is_computed_.find(utt_id)->second) {
      *uttid = utt_id;
      std::unordered_map<std::string, Matrix<BaseFloat>*,
                         StringHasher>::iterator iter;
      iter = log_post_.find(utt_id);
      int32 num_rows = (iter->second)->NumRows(),
            num_cols = (iter->second)->NumCols();
      output_matrix->Resize(num_rows, num_cols);
      output_matrix->CopyFromMat(*(iter->second));
      Clear(utt_id);
      return true;
    } else {
      return false;
    }
  }
  return false;
}


void BatchNnetComputer::Clear(std::string utt_id) {
  delete feats_.find(utt_id)->second;
  feats_.erase(utt_id);
  if (ivectors_.find(utt_id) != ivectors_.end()) {
    delete ivectors_.find(utt_id)->second;
    ivectors_.erase(utt_id);
  }
  if (online_ivector_feats_.find(utt_id) != online_ivector_feats_.end()) {
    delete online_ivector_feats_.find(utt_id)->second;
    online_ivector_feats_.erase(utt_id);
  }
  delete log_post_.find(utt_id)->second;
  log_post_.erase(utt_id);
  num_subsampled_frames_.erase(utt_id);
  utt_list_.remove(utt_id);
  is_computed_.erase(utt_id);
}


void BatchNnetComputer::Compute(bool flush) {
  if (flush) {
    while (!Empty()) {
      if (ensure_exact_final_context_) {
        DoNnetComputationOnes();
      }
      DoNnetComputation();
      PrepareBatchInfo();
    }
  } else {
    while (Ready()) {
      if (ensure_exact_final_context_) {
        DoNnetComputationOnes();
      }
      DoNnetComputation();
      PrepareBatchInfo();
    }
  }
}


BatchNnetComputer::~BatchNnetComputer() {
  BatchInfoMap::iterator iter =
    batch_info_.begin(), end = batch_info_.end();
  for (; iter != end; iter++) {
    delete iter->second;
  }
  ComputationRequestMap::iterator iter2 =
    context_to_request_.begin(), end2 = context_to_request_.end();
  for (; iter2 != end2; iter2++) {
    delete iter2->second;
  }
}


}  // namespace nnet3
}  // namespace kaldi
