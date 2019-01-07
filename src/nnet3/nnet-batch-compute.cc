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
#include <iomanip>
#include "nnet3/nnet-batch-compute.h"
#include "nnet3/nnet-utils.h"
#include "decoder/decodable-matrix.h"

namespace kaldi {
namespace nnet3 {


NnetBatchComputer::NnetBatchComputer(
    const NnetBatchComputerOptions &opts,
    const Nnet &nnet,
    const VectorBase<BaseFloat> &priors):
    opts_(opts),
    nnet_(nnet),
    compiler_(nnet_, opts.optimize_config),
    log_priors_(priors),
    num_full_minibatches_(0) {
  log_priors_.ApplyLog();
  CheckAndFixConfigs();
  ComputeSimpleNnetContext(nnet, &nnet_left_context_,
                           &nnet_right_context_);
  input_dim_ = nnet.InputDim("input");
  ivector_dim_ = std::max<int32>(0, nnet.InputDim("ivector"));
  output_dim_ = nnet.OutputDim("output");
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ > 0);
}

void NnetBatchComputer::PrintMinibatchStats() {
  int32 max_stats_to_print = 10;
  int64 tot_tasks = 0, tot_minibatches = 0;
  double tot_time = 0.0;
  std::ostringstream os;
  struct MinibatchStats {
    int32 num_frames_out;
    int32 num_frames_in;
    int32 minibatch_size;
    int32 num_done;
    int32 percent_full;
    BaseFloat seconds_taken;

    bool operator < (const MinibatchStats &other) const {
      return seconds_taken > other.seconds_taken;  // sort from most to least time.
    }
  };
  std::vector<MinibatchStats> all_stats;
  os << "Minibatch stats: seconds-taken,frames-in:frames-out*minibatch-size=num-done(percent-full%)  ";

  for (MapType::const_iterator iter = tasks_.begin();
       iter != tasks_.end(); ++iter) {
    for (std::map<int32, MinibatchSizeInfo>::const_iterator
             miter = iter->second.minibatch_info.begin();
         miter != iter->second.minibatch_info.end(); ++miter) {
      const ComputationGroupKey &key = iter->first;
      const MinibatchSizeInfo &minfo = miter->second;
      MinibatchStats stats;
      stats.num_frames_in = key.num_input_frames;
      stats.num_frames_out = key.num_output_frames;
      stats.minibatch_size = miter->first;
      stats.num_done = minfo.num_done;
      stats.seconds_taken = minfo.seconds_taken;

      tot_tasks += minfo.tot_num_tasks;
      tot_minibatches += minfo.num_done;
      tot_time += minfo.seconds_taken;
      stats.percent_full = int32(minfo.tot_num_tasks * 100.0 /
                                 (stats.minibatch_size * stats.num_done));
      all_stats.push_back(stats);
    }
  }

  std::sort(all_stats.begin(), all_stats.end());
  os << std::fixed << std::setprecision(2);
  int32 num_stats = all_stats.size();
  for (int32 i = 0; i < std::min<int32>(num_stats, max_stats_to_print); i++) {
    MinibatchStats &stats = all_stats[i];
    os << stats.seconds_taken << ',' << stats.num_frames_in << ':'
       << stats.num_frames_out << '*' << stats.minibatch_size
       << '=' << stats.num_done << '(' << stats.percent_full << "%) ";
  }
  if (num_stats > max_stats_to_print)
    os << "...";
  KALDI_LOG << os.str();
  KALDI_LOG << "Did " << tot_tasks << " tasks in " << tot_minibatches
            << " minibatches, taking " << tot_time << " seconds.";
}

NnetBatchComputer::~NnetBatchComputer() {
  PrintMinibatchStats();
  // the destructor shouldn't be called while the mutex is locked; if it is, it
  // likely means the program has already crashed, or it's a programming error.
  if (!mutex_.try_lock())
    KALDI_ERR << "Destructor called while object locked.";
  int32 num_pending_tasks = 0;
  for (auto iter = tasks_.begin(); iter != tasks_.end(); ++iter)
    num_pending_tasks += iter->second.tasks.size();
  if (num_pending_tasks > 0)
    KALDI_ERR << "Tasks are pending but object is being destroyed";
  for (auto iter = no_more_than_n_minibatches_full_.begin();
       iter != no_more_than_n_minibatches_full_.end(); ++iter) {
    std::condition_variable *cond = iter->second;
    // the next call will notify any threads that were waiting on this condition
    // variable-- there shouldn't be any, though, as it would be a programming
    // error, but better to wake them up so we can see any messages they print.
    cond->notify_all();
    delete cond;
  }
  KALDI_ASSERT(num_full_minibatches_ == 0);  // failure would be a coding error.
}

NnetBatchComputer::MinibatchSizeInfo*
NnetBatchComputer::GetHighestPriorityComputation(
    bool allow_partial_minibatch,
    int32 *minibatch_size_out,
    std::vector<NnetInferenceTask*> *tasks) {
  tasks->clear();
  std::unique_lock<std::mutex> lock(mutex_);
  MapType::iterator iter = tasks_.begin(), end = tasks_.end(),
      best_iter = tasks_.end();
  double highest_priority = -std::numeric_limits<double>::infinity();

  for (; iter != end; ++iter) {
    ComputationGroupInfo &info = iter->second;
    double this_priority = GetPriority(allow_partial_minibatch, info);
    if (this_priority > highest_priority) {
      highest_priority = this_priority;
      best_iter = iter;
    }
  }
  if (best_iter == tasks_.end()) {
    // either allow_partial_minibatch == false and there were no full
    // minibatches, or there were no pending tasks at all.
    return NULL;
  }
  ComputationGroupInfo &info = best_iter->second;
  int32 actual_minibatch_size = GetActualMinibatchSize(info);
  *minibatch_size_out = actual_minibatch_size;
  MinibatchSizeInfo *minfo = &(info.minibatch_info[actual_minibatch_size]);
  if (minfo->computation == NULL)
    minfo->computation = GetComputation(info, actual_minibatch_size);
  GetHighestPriorityTasks(actual_minibatch_size, &info, tasks);
  return minfo;
}


void NnetBatchComputer::GetHighestPriorityTasks(
    int32 num_tasks_needed,
    ComputationGroupInfo *info,
    std::vector<NnetInferenceTask*> *tasks) {
  int32 num_tasks_present = info->tasks.size(),
      minibatch_size = GetMinibatchSize(*info);
  KALDI_ASSERT(tasks->empty());
  if (num_tasks_needed >= num_tasks_present) {
    tasks->swap(info->tasks);
  } else {
    int32 num_tasks_not_needed = num_tasks_present - num_tasks_needed;
    // We don't sort the tasks with a comparator that dereferences the pointers,
    // because the priorities can change asynchronously, and we're concerned that
    // something weird might happen in the sorting if the things it's comparing
    // are changing.
    std::vector<std::pair<double, NnetInferenceTask*> > pairs(num_tasks_present);
    for (int32 i = 0; i < num_tasks_present; i++) {
      pairs[i].first = info->tasks[i]->priority;
      pairs[i].second = info->tasks[i];
    }
    std::nth_element(pairs.begin(), pairs.begin() + num_tasks_not_needed,
                     pairs.end());

    // The lowest-priority 'num_tasks_not_needed' stay in the 'info' struct.
    info->tasks.clear();
    for (int32 i = 0; i < num_tasks_not_needed; i++)
      info->tasks.push_back(pairs[i].second);
    // The highest-priority 'num_tasks_needed' tasks go to the output 'tasks'
    // array.
    for (int32 i = num_tasks_not_needed; i < num_tasks_present; i++)
      tasks->push_back(pairs[i].second);
    // The following assertion checks that the is_edge and is_irregular values
    // are the same for the entire minibatch, which they should always be.
    KALDI_ASSERT(GetMinibatchSize(*info) == minibatch_size);
  }

  {
    // This block updates num_full_minibatches_ and notifies threads waiting on
    // any related condition variable.
    int32 new_num_tasks_present = info->tasks.size(),
        full_minibatch_reduction =
        (num_tasks_present / minibatch_size) -
        (new_num_tasks_present / minibatch_size);
    for (int32 i = 0; i < full_minibatch_reduction; i++) {
      num_full_minibatches_--;
      KALDI_ASSERT(num_full_minibatches_ >= 0);
      std::unordered_map<int32, std::condition_variable*>::const_iterator
          iter = no_more_than_n_minibatches_full_.find(num_full_minibatches_);
      if (iter != no_more_than_n_minibatches_full_.end()) {
        std::condition_variable *cond = iter->second;
        cond->notify_all();
      }
    }
  }
}


int32 NnetBatchComputer::GetMinibatchSize(
    const ComputationGroupInfo &info) const {
  if (info.tasks.empty()) {
    return opts_.minibatch_size; // actually it shouldn't matter what we return
                                 // in this case.
  }
  const NnetInferenceTask &task = *(info.tasks[0]);
  if (task.is_irregular)
    return 1;
  else if (task.is_edge)
    return opts_.edge_minibatch_size;
  else
    return opts_.minibatch_size;
}

int32 NnetBatchComputer::GetActualMinibatchSize(
    const ComputationGroupInfo &info) const {
  KALDI_ASSERT(!info.tasks.empty());
  int32 num_tasks = info.tasks.size(),
      this_minibatch_size = GetMinibatchSize(info);
  KALDI_ASSERT(num_tasks > 0);
  while (num_tasks <
         int32(opts_.partial_minibatch_factor * this_minibatch_size))
    this_minibatch_size *= opts_.partial_minibatch_factor;
  return int32(this_minibatch_size);
}


std::shared_ptr<const NnetComputation> NnetBatchComputer::GetComputation(
    const ComputationGroupInfo &info,
    int32 minibatch_size) {
  KALDI_ASSERT(!info.tasks.empty());
  // note: all the tasks will have the same structure, in the respects that
  // would affect the computation.
  NnetInferenceTask *example_task = info.tasks[0];
  ComputationRequest request;
  GetComputationRequest(*example_task, minibatch_size, &request);
  return compiler_.Compile(request);
}


double NnetBatchComputer::GetPriority(bool allow_partial_minibatch,
                                      const ComputationGroupInfo &info) const {
  if (info.tasks.empty())
    return -std::numeric_limits<double>::infinity();
  int32 this_minibatch_size = GetMinibatchSize(info);
  int32 num_tasks = info.tasks.size();

  if (!allow_partial_minibatch && num_tasks < this_minibatch_size)
    return -std::numeric_limits<double>::infinity();

  // penalty_for_not_full will be negative if the minibatch is not full, up to a
  // maximum of 10.  the 10 is a heuristic; it could be changed.
  // Note: the penalty is effectively infinity if allow_partial_minibatch == false;
  // see the 'return' above.
  double proportion_full = std::min<int32>(num_tasks, this_minibatch_size) /
      double(this_minibatch_size),
      penalty_for_not_full = 10.0 * (proportion_full - 1.0),
      task_priority_sum = 0.0;


  if (num_tasks > this_minibatch_size) {
    // Get the average of the priorities of the highest-priority tasks (no more
    // than 'minibatch_size' of them.
    std::vector<double> priorities;
    priorities.resize(num_tasks);
    for (int32 i = 0; i < num_tasks; i++)
      priorities[i] = info.tasks[i]->priority;
    // sort from greatest to least.
    std::nth_element(priorities.begin(),
                     priorities.begin() + this_minibatch_size,
                     priorities.end(),
                     std::greater<double>());
    for (int32 i = 0; i < this_minibatch_size; i++)
      task_priority_sum += priorities[i];
    return penalty_for_not_full + task_priority_sum / this_minibatch_size;
  } else {
    for (int32 i = 0; i < num_tasks; i++)
      task_priority_sum += info.tasks[i]->priority;
    return penalty_for_not_full + task_priority_sum / num_tasks;
  }
}


// static
void NnetBatchComputer::GetComputationRequest(
    const NnetInferenceTask &task,
    int32 minibatch_size,
    ComputationRequest *request) {
  request->need_model_derivative = false;
  request->store_component_stats = false;
  request->inputs.reserve(2);

  int32 num_input_frames = task.input.NumRows(),
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
    for (int32 t = first_input_t; t < first_input_t + num_input_frames; t++)
      input_indexes.push_back(Index(n, t, 0));
    if (has_ivector)
      ivector_indexes.push_back(Index(n, 0, 0));
    for (int32 t = 0; t < num_output_frames; t++)
      output_indexes.push_back(Index(n, t * output_t_stride, 0));
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
  KALDI_ASSERT(opts_.minibatch_size >= 1 &&
               opts_.edge_minibatch_size >= 1 &&
               opts_.partial_minibatch_factor < 1.0 &&
               opts_.partial_minibatch_factor >= 0.0);
}


void NnetBatchComputer::FormatInputs(
    int32 minibatch_size,
    const std::vector<NnetInferenceTask*> &tasks,
    CuMatrix<BaseFloat> *input,
    CuMatrix<BaseFloat> *ivector) {
  int32 num_input_frames = tasks[0]->input.NumRows(),
      input_dim = tasks[0]->input.NumCols(),
      ivector_dim = tasks[0]->ivector.Dim(),
      num_tasks = tasks.size();
  KALDI_ASSERT(num_tasks > 0 && num_tasks <= minibatch_size);

  // We first aggregate the input frames and i-vectors in matrices on the CPU,
  // and then transfer them to the GPU.  Later on we'll change this code to
  // used pinned memory.
  Matrix<BaseFloat> input_cpu(num_tasks * num_input_frames, input_dim,
                              kUndefined);


  for (int32 n = 0; n < num_tasks; n++) {
    SubMatrix<BaseFloat> input_part(input_cpu,
                                    n * num_input_frames, num_input_frames,
                                    0, input_dim);
    input_part.CopyFromMat(tasks[n]->input);
  }
  input->Resize(minibatch_size * num_input_frames, input_dim,
                kUndefined);
  input->RowRange(0, num_tasks * num_input_frames).CopyFromMat(input_cpu);
  if (num_tasks < minibatch_size) {
    // The following will make things easier to debug if something fails, but
    // shouldn't be strictly necessary.
    // the -1 means 'take all remaining rows'.
    input->RowRange(num_tasks * num_input_frames,
                    (minibatch_size - num_tasks) * num_input_frames).SetZero();
  }

  if (ivector_dim != 0) {
    Matrix<BaseFloat> ivectors_cpu(num_tasks, ivector_dim, kUndefined);
    for (int32 n = 0; n < num_tasks; n++)
      ivectors_cpu.Row(n).CopyFromVec(tasks[n]->ivector);

    ivector->Resize(minibatch_size, ivector_dim, kUndefined);
    ivector->RowRange(0, num_tasks).CopyFromMat(ivectors_cpu);

    if (num_tasks < minibatch_size) {
      // The following will make things easier to debug if something fails, but
      // shouldn't be strictly necessary.
      // the -1 means 'take all remaining rows'.
      ivector->RowRange(num_tasks, minibatch_size - num_tasks).SetZero();
    }
  }
}

void NnetBatchComputer::FormatOutputs(
    const CuMatrix<BaseFloat> &output,
    const std::vector<NnetInferenceTask*> &tasks) {
  KALDI_ASSERT(!tasks.empty());
  int32 num_output_frames = tasks[0]->num_output_frames,
      output_dim = output.NumCols(),
      num_tasks = tasks.size();
  bool did_output_to_gpu = false;

  // Note: it may not be optimal to do so many individual calls to copy the
  // output to CPU; we'd have to test that, as I'm not sure how much the latency
  // of a GPU call is.  On the other hand, the downsides of one big call are
  // that we'd have to make another copy in CPU memory; and also we might not be
  // able to take advantage if not all frames of the output are used.

  // Also, we should probably used pinned memory.

  // We don't bother zeroing frames of the output that are unused, but you could
  // un-comment the commented lines of code below to do so.
  for (int32 n = 0; n < num_tasks; n++) {
    NnetInferenceTask *task = tasks[n];

    int32 left_unused = task->num_initial_unused_output_frames,
        used = task->num_used_output_frames;
     // int32 right_unused = num_output_frames - used - left_unused;

    if (task->output_to_cpu) {
      task->output_cpu.Resize(num_output_frames, output_dim,
                              kUndefined);
      // if (left_unused > 0)
      //   task->output_cpu.RowRange(0, left_unused).SetZero();
      task->output_cpu.RowRange(left_unused, used).CopyFromMat(
          output.RowRange(n * num_output_frames + left_unused, used));
      // if (right_unused > 0)
      //   task->output_cpu.RowRange(0, left_unused + used, right_unused).SetZero();
    } else {
      did_output_to_gpu = true;
      task->output.Resize(num_output_frames, output_dim,
                          kUndefined);
      // if (left_unused > 0)
      //   task->output.RowRange(0, left_unused).SetZero();
      task->output.RowRange(left_unused, used).CopyFromMat(
          output.RowRange(n * num_output_frames + left_unused, used));
      // if (right_unused > 0)
      //   task->output.RowRange(0, left_unused + used, right_unused).SetZero();
    }
  }
  // The output of this function will likely be consumed by another thread.
  // The following call will make sure the relevant kernels complete before
  // any kernels from the other thread use the output.
  if (did_output_to_gpu)
    SynchronizeGpu();
}

void NnetBatchComputer::AcceptTask(NnetInferenceTask *task,
                                   int32 max_minibatches_full) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (max_minibatches_full > 0 && num_full_minibatches_ > max_minibatches_full) {
    std::unordered_map<int32, std::condition_variable*>::iterator
        iter = no_more_than_n_minibatches_full_.find(max_minibatches_full);
    std::condition_variable *cond;
    if (iter != no_more_than_n_minibatches_full_.end()) {
      cond = iter->second;
    } else {
      cond = new std::condition_variable();
      no_more_than_n_minibatches_full_[max_minibatches_full] = cond;
    }
    while (num_full_minibatches_ > max_minibatches_full)
      cond->wait(lock);
  }
  ComputationGroupKey key(*task);
  ComputationGroupInfo &info = tasks_[key];
  info.tasks.push_back(task);
  int32 minibatch_size = GetMinibatchSize(info);
  if (static_cast<int32>(info.tasks.size()) % minibatch_size == 0)
    num_full_minibatches_++;
}

bool NnetBatchComputer::Compute(bool allow_partial_minibatch) {
  int32 minibatch_size;
  std::vector<NnetInferenceTask*> tasks;
  MinibatchSizeInfo *minfo =
      GetHighestPriorityComputation(allow_partial_minibatch,
                                    &minibatch_size,
                                    &tasks);
  if (minfo == NULL)
    return false;

  Timer tim;
  Nnet *nnet_to_update = NULL;  // we're not doing any update
  NnetComputer computer(opts_.compute_config, *(minfo->computation),
                        nnet_, nnet_to_update);


  CuMatrix<BaseFloat> input;
  CuMatrix<BaseFloat> ivector;
  FormatInputs(minibatch_size, tasks, &input, &ivector);
  computer.AcceptInput("input", &input);
  if (ivector.NumRows() != 0)
    computer.AcceptInput("ivector", &ivector);
  computer.Run();
  CuMatrix<BaseFloat> output;
  computer.GetOutputDestructive("output", &output);
  if (log_priors_.Dim() != 0) {
    output.AddVecToRows(-1.0, log_priors_);
  }
  output.Scale(opts_.acoustic_scale);
  FormatOutputs(output, tasks);

  // Update the stats, for diagnostics.
  minfo->num_done++;
  minfo->tot_num_tasks += static_cast<int64>(tasks.size());
  minfo->seconds_taken += tim.Elapsed();


  SynchronizeGpu();

  for (size_t i = 0; i < tasks.size(); i++)
    tasks[i]->semaphore.Signal();

  return true;
}


/**
   This namespace contains things needed for the implementation of
   the function NnetBatchComputer::SplitUtteranceIntoTasks().
 */
namespace utterance_splitting {
/**
   This function figures out how many chunks are needed for this utterance,
   sets 'tasks' to a vector with that many elements, and sets up the
   following elements in 'tasks':
   output_t_stride
   num_output_frames
   num_initial_unused_output_frames
   num_used_output_frames
   @param [in] opts   Options class
   @param [in] num_subsampled_frames  The number of output frames in this
   utterance.  Must be > 0.
   @param [in] num_subsampled_frames_per_chunk  The number of output frames
   per chunk
   @param [out] The 'tasks' array is output to here; it will have one
   task per chunk, with only the members 'output_t_stride',
    'num_output_frames', 'num_initial_unused_output_frames',
    'num_used_output_frames' and 'is_irregular' set up.
*/
void GetOutputFrameInfoForTasks(
    const NnetBatchComputerOptions &opts,
    int32 num_subsampled_frames,
    int32 num_subsampled_frames_per_chunk,
    std::vector<NnetInferenceTask> *tasks) {
  KALDI_ASSERT(num_subsampled_frames > 0);
  int32 fpc = num_subsampled_frames_per_chunk;
  int32 num_tasks = (num_subsampled_frames + fpc - 1) / fpc;
  tasks->resize(num_tasks);
  for (int32 i = 0; i < num_tasks; i++) {
    (*tasks)[i].output_t_stride = opts.frame_subsampling_factor;
  }
  if (num_subsampled_frames <= fpc) {  // there is one chunk.
    KALDI_ASSERT(num_tasks == 1);  // TODO: remove this.
    NnetInferenceTask &task = (*tasks)[0];
    task.first_used_output_frame_index = 0;
    if (opts.ensure_exact_final_context) {
      task.num_output_frames = num_subsampled_frames;
      task.num_initial_unused_output_frames = 0;
      task.num_used_output_frames = num_subsampled_frames;
      task.is_irregular = true;
    } else {
      task.num_output_frames = fpc;
      task.num_initial_unused_output_frames = 0;
      task.num_used_output_frames = num_subsampled_frames;
      task.is_irregular = false;
    }
  } else {
    for (int32 i = 0; i + 1 < num_tasks; i++) {
      NnetInferenceTask &task = (*tasks)[i];
      task.num_output_frames = fpc;
      task.num_initial_unused_output_frames = 0;
      task.num_used_output_frames = fpc;
      task.first_used_output_frame_index = i * fpc;
      task.is_irregular = false;
    }
    // The last chunk will end on the last frame of the file, but we won't use
    // the part of its output that overlaps with the preceding chunk.
    NnetInferenceTask &task = (*tasks)[num_tasks - 1];
    task.num_output_frames = fpc;
    task.num_initial_unused_output_frames = ((num_tasks - 1) * fpc) -
        (num_subsampled_frames - fpc);
    task.num_used_output_frames =
        num_subsampled_frames - ((num_tasks - 1) * fpc);
    task.first_used_output_frame_index = (num_tasks - 1) * fpc;
    task.is_irregular = false;
  }

  if (true) {
    // Do some checking.  TODO: remove this.
    KALDI_ASSERT((*tasks)[0].first_used_output_frame_index == 0);
    for (int32 i = 1; i < num_tasks; i++) {
      KALDI_ASSERT((*tasks)[i].first_used_output_frame_index ==
                   (*tasks)[i-1].first_used_output_frame_index +
                   (*tasks)[i-1].num_used_output_frames);
    }
    KALDI_ASSERT((*tasks)[num_tasks-1].first_used_output_frame_index +
                 (*tasks)[num_tasks-1].num_used_output_frames ==
                 num_subsampled_frames);
    for (int32 i = 0; i < num_tasks; i++) {
      const NnetInferenceTask &task = (*tasks)[i];
      KALDI_ASSERT(task.num_used_output_frames +
                   task.num_initial_unused_output_frames <=
                   task.num_output_frames);
    }
  }
}

void AddOnlineIvectorsToTasks(
    const NnetBatchComputerOptions &opts,
    const Matrix<BaseFloat> &online_ivectors,
    int32 online_ivector_period,
    std::vector<NnetInferenceTask> *tasks) {
  int32 f = opts.frame_subsampling_factor,
      num_tasks = tasks->size();
  for (int32 i = 0; i < num_tasks; i++) {
    NnetInferenceTask &task = (*tasks)[i];
    // begin_output_t and end_output_t are the subsampled frame indexes at
    // the output; you'd have to multiply them by f to get real frame indexes.
    int32 begin_output_t = task.first_used_output_frame_index -
        task.num_initial_unused_output_frames,
        mid_output_t = begin_output_t + (task.num_output_frames / 2),
        mid_input_t = mid_output_t * f,
        ivector_frame = mid_input_t / online_ivector_period,
        num_ivector_frames = online_ivectors.NumRows(),
        margin_in_frames = 20,
        margin_in_ivector_frames =
        (margin_in_frames + online_ivector_period - 1) / online_ivector_period;
    // the 'margin' is our tolerance for when the number of rows of
    // 'online_ivectors' is less than what we expected; we allow 20 frames of
    // tolerance in the numbering of the original (input) features.
    if (ivector_frame >= num_ivector_frames) {
      if (num_ivector_frames > 0 && ivector_frame > num_ivector_frames -
          margin_in_ivector_frames) {
        ivector_frame = num_ivector_frames - 1;  // Just take the last available one.
      } else {
        KALDI_ERR << "Could not get iVector for frame " << ivector_frame
                  << ", online-ivectors matrix has "
                  << online_ivectors.NumRows()
                  << " rows.  Mismatched --online-ivector-period?";
      }
    }
    task.ivector = online_ivectors.Row(ivector_frame);
  }
}



/**
   This function sets up the 'input' and 'first_input_t' and 'is_edge' members
   of the 'tasks' array; it is responsible for working out, for each task,
   which input frames it needs (including left-context and right-context).

   The 'nnet_left_context' and 'nnet_right_context' are the inherent left
   and right context of the network (num-frames required on left and right
   to compute an output frame), and may be computed by doing:
    ComputeSimpleNnetContext(nnet, &nnet_left_context_, &nnet_right_context_)
*/
static void SplitInputToTasks(const NnetBatchComputerOptions &opts,
                              int32 nnet_left_context,
                              int32 nnet_right_context,
                              const Matrix<BaseFloat> &input,
                              std::vector<NnetInferenceTask> *tasks) {
  int32 num_input_frames = input.NumRows(),
      f = opts.frame_subsampling_factor,
      num_subsampled_frames = (num_input_frames + f - 1) / f,
      extra_left_context_initial = (opts.extra_left_context_initial < 0 ?
                                    opts.extra_left_context :
                                    opts.extra_left_context_initial),
      extra_right_context_final = (opts.extra_right_context_final < 0 ?
                                   opts.extra_right_context :
                                   opts.extra_right_context_final),
      num_tasks = tasks->size();
  for (int32 i = 0; i < num_tasks; i++) {
    NnetInferenceTask &task = (*tasks)[i];
    // begin_output_t and end_output_t are the subsampled frame indexes at
    // the output; you'd have to multiply them by f to get real frame indexes.
    int32 begin_output_t = task.first_used_output_frame_index -
        task.num_initial_unused_output_frames,
        end_output_t = begin_output_t + task.num_output_frames;
    // begin_input_t and end_input_t are the real 't' values corresponding to
    // begin_output_t and end_output_t; they are the beginning and end
    // (i.e. first and last-plus-one) frame indexes without any left or right
    // context.
    int32 begin_input_t = begin_output_t * f,
        end_input_t = end_output_t * f;
    // Detect whether the left and right edges touch (or pass over) the left
    // and right boundaries.  Note: we don't expect begin_output_t to ever be
    // negative.
    bool left_edge = (begin_output_t <= 0),
        right_edge = (end_output_t >= num_subsampled_frames);
    int32 tot_left_context = nnet_left_context +
        (left_edge ? extra_left_context_initial : opts.extra_left_context),
        tot_right_context = nnet_right_context +
        (right_edge ? extra_right_context_final : opts.extra_right_context);

    // 'is_edge' is only true if it's an edge minibatch *and* its being an
    // edge actually made a difference to the structure of the example.
    task.is_edge =
        (tot_left_context != nnet_left_context + opts.extra_left_context ||
         tot_right_context !=  nnet_right_context + opts.extra_right_context);

    int32 begin_input_t_padded = begin_input_t - tot_left_context,
        end_input_t_padded = end_input_t + tot_right_context;

    // 'task.first_input_t' is a representation of 'begin_input_t_padded' in a
    // shifted/normalized numbering where the output time indexes start from
    // zero.
    task.first_input_t = begin_input_t_padded - (begin_output_t * f);

    task.input.Resize(end_input_t_padded - begin_input_t_padded,
                      input.NumCols(), kUndefined);
    // the 't' value below is in the numbering of 'input'.
    for (int32 t = begin_input_t_padded; t < end_input_t_padded; t++) {
      int32 t_clipped = t;
      if (t_clipped < 0) t_clipped = 0;
      if (t_clipped >= num_input_frames) t_clipped = num_input_frames - 1;
      SubVector<BaseFloat> dest(task.input,
                                t - begin_input_t_padded),
          src(input, t_clipped);
      dest.CopyFromVec(src);
    }
  }
}

} // namespace utterance_splitting


void NnetBatchComputer::SplitUtteranceIntoTasks(
    bool output_to_cpu,
    const Matrix<BaseFloat> &input,
    const Vector<BaseFloat> *ivector,
    const Matrix<BaseFloat> *online_ivectors,
    int32 online_ivector_period,
    std::vector<NnetInferenceTask> *tasks) {
  using namespace utterance_splitting;


  { // This block does some checking.
    if (input.NumCols() != input_dim_) {
      KALDI_ERR << "Input features did not have expected dimension: expected "
          << input_dim_ << ", got " << input.NumCols();
    }
    int32 ivector_dim = (ivector != NULL ? ivector->Dim() :
                         (online_ivectors != NULL ?
                          online_ivectors->NumCols() : 0));
    if (ivector_dim_ != 0 && ivector_dim == 0)
      KALDI_ERR << "Model expects i-vectors but none were supplied";
    else if (ivector_dim_ == 0 && ivector_dim != 0)
      KALDI_ERR << "You supplied i-vectors but model does not expect them.";
    else if (ivector_dim != ivector_dim_)
      KALDI_ERR << "I-vector dimensions mismatch: model expects "
                << ivector_dim_ << ", you supplied " << ivector_dim;
  }


  int32 num_input_frames = input.NumRows(),
      f = opts_.frame_subsampling_factor,
      num_subsampled_frames = (num_input_frames + f - 1) / f,
      num_subsampled_frames_per_chunk = opts_.frames_per_chunk / f;

  GetOutputFrameInfoForTasks(opts_, num_subsampled_frames,
                             num_subsampled_frames_per_chunk,
                             tasks);

  SplitInputToTasks(opts_, nnet_left_context_, nnet_right_context_,
                    input, tasks);

  if (ivector != NULL) {
    KALDI_ASSERT(online_ivectors == NULL);
    for (size_t i = 0; i < tasks->size(); i++)
      (*tasks)[i].ivector = *ivector;
  } else if (online_ivectors != NULL) {
    AddOnlineIvectorsToTasks(opts_, *online_ivectors,
                             online_ivector_period, tasks);
  }

  for (size_t i = 0; i < tasks->size(); i++) {
    (*tasks)[i].output_to_cpu = output_to_cpu;
    // The priority will be set by the user; this just avoids undefined
    // behavior.
    (*tasks)[i].priority = 0.0;
  }
}


void MergeTaskOutput(
    const std::vector<NnetInferenceTask> &tasks,
    Matrix<BaseFloat> *output) {
  int32 num_tasks = tasks.size(),
      num_output_frames = 0,
      output_dim = -1;
  for (int32 i = 0; i < num_tasks; i++) {
    const NnetInferenceTask &task = tasks[i];
    num_output_frames += task.num_used_output_frames;
    if (i == 0) {
      output_dim = (task.output_to_cpu ?
                    task.output_cpu.NumCols() :
                    task.output.NumCols());
    }
  }
  KALDI_ASSERT(num_output_frames != 0 && output_dim != 0);
  int32 cur_output_frame = 0;
  output->Resize(num_output_frames, output_dim);
  for (int32 i = 0; i < num_tasks; i++) {
    const NnetInferenceTask &task = tasks[i];
    int32 skip = task.num_initial_unused_output_frames,
        num_used = task.num_used_output_frames;
    KALDI_ASSERT(cur_output_frame == task.first_used_output_frame_index);
    if (task.output_to_cpu) {
      output->RowRange(cur_output_frame, num_used).CopyFromMat(
          task.output_cpu.RowRange(skip, num_used));
    } else {
      output->RowRange(cur_output_frame, num_used).CopyFromMat(
          task.output.RowRange(skip, num_used));
    }
    cur_output_frame += num_used;
  }
  KALDI_ASSERT(cur_output_frame == num_output_frames);
}


NnetBatchInference::NnetBatchInference(
    const NnetBatchComputerOptions &opts,
    const Nnet &nnet,
    const VectorBase<BaseFloat> &priors):
    computer_(opts, nnet, priors),
    is_finished_(false),
    utterance_counter_(0) {
  // 'thread_' will run the Compute() function in the background.
  compute_thread_ = std::thread(ComputeFunc, this);
}


void NnetBatchInference::AcceptInput(
    const std::string &utterance_id,
    const Matrix<BaseFloat> &input,
    const Vector<BaseFloat> *ivector,
    const Matrix<BaseFloat> *online_ivectors,
    int32 online_ivector_period) {

  UtteranceInfo *info = new UtteranceInfo();
  info->utterance_id = utterance_id;
  info->num_tasks_finished = 0;
  bool output_to_cpu = true;  // This wrapper is for when you need the nnet
                              // output on CPU, e.g.  because you want it
                              // written to disk.  If this needs to be
                              // configurable in the future, we can make changes
                              // then.
  computer_.SplitUtteranceIntoTasks(
      output_to_cpu, input, ivector, online_ivectors,
      online_ivector_period, &(info->tasks));

  // Setting this to a nonzero value will cause the AcceptTask() call below to
  // hang until the computation thread has made some progress, if too much
  // data is already queued.
  int32 max_full_minibatches = 2;

  // Earlier utterances have higher priority, which is important to make sure
  // that their corresponding tasks are completed and they can be output to disk.
  double priority = -1.0 * (utterance_counter_++);
  for (size_t i = 0; i < info->tasks.size(); i++) {
    info->tasks[i].priority = priority;
    computer_.AcceptTask(&(info->tasks[i]), max_full_minibatches);
  }
  utts_.push_back(info);
  tasks_ready_semaphore_.Signal();
}

bool NnetBatchInference::GetOutput(std::string *utterance_id,
                                   Matrix<BaseFloat> *output) {
  if (utts_.empty())
    return false;

  UtteranceInfo *info = *utts_.begin();
  std::vector<NnetInferenceTask> &tasks = info->tasks;
  int32 num_tasks = tasks.size();
  for (; info->num_tasks_finished < num_tasks; ++info->num_tasks_finished) {
    Semaphore &semaphore = tasks[info->num_tasks_finished].semaphore;
    if (is_finished_) {
      semaphore.Wait();
    } else {
      if (!semaphore.TryWait()) {
        // If not all of the tasks of this utterance are ready yet,
        // just return false.
        return false;
      }
    }
  }
  MergeTaskOutput(tasks, output);
  *utterance_id = info->utterance_id;
  delete info;
  utts_.pop_front();
  return true;
}

NnetBatchInference::~NnetBatchInference() {
  if (!is_finished_)
    KALDI_ERR << "Object destroyed before Finished() was called.";
  if (!utts_.empty())
    KALDI_ERR << "You should get all output before destroying this object.";
  compute_thread_.join();
}

void NnetBatchInference::Finished() {
  is_finished_ = true;
  tasks_ready_semaphore_.Signal();
}

// This is run as the thread of class NnetBatchInference.
void NnetBatchInference::Compute() {
  bool allow_partial_minibatch = false;
  while (true) {
    // keep calling Compute() as long as it makes progress.
    while (computer_.Compute(allow_partial_minibatch));

    // ... then wait on tasks_ready_semaphore_.
    tasks_ready_semaphore_.Wait();
    if (is_finished_) {
      allow_partial_minibatch = true;
      while (computer_.Compute(allow_partial_minibatch));
      return;
    }
  }
}


NnetBatchDecoder::NnetBatchDecoder(
    const fst::Fst<fst::StdArc> &fst,
    const LatticeFasterDecoderConfig &decoder_opts,
    const TransitionModel &trans_model,
    const fst::SymbolTable *word_syms,
    bool allow_partial,
    int32 num_threads,
    NnetBatchComputer *computer):
  fst_(fst), decoder_opts_(decoder_opts),
  trans_model_(trans_model), word_syms_(word_syms),
  allow_partial_(allow_partial),  computer_(computer),
  is_finished_(false), tasks_finished_(false), priority_offset_(0.0),
  tot_like_(0.0), frame_count_(0), num_success_(0), num_fail_(0),
  num_partial_(0) {
  KALDI_ASSERT(num_threads > 0);
  for (int32 i = 0; i < num_threads; i++)
    decode_threads_.push_back(new std::thread(DecodeFunc, this));
  compute_thread_ = std::thread(ComputeFunc, this);
}

void NnetBatchDecoder::SetPriorities(std::vector<NnetInferenceTask> *tasks) {
  size_t num_tasks = tasks->size();
  double priority_offset = priority_offset_;
  for (size_t i = 0; i < num_tasks; i++)
    (*tasks)[i].priority = priority_offset - (double)i;
}

void NnetBatchDecoder::UpdatePriorityOffset(double priority) {
  size_t num_tasks = decode_threads_.size(),
      new_weight = 1.0 / num_tasks,
      old_weight = 1.0 - new_weight;
  // The next line is vulnerable to a race condition but if it happened it
  // wouldn't matter.
  priority_offset_ = priority_offset_ * old_weight + priority * new_weight;
}

void NnetBatchDecoder::AcceptInput(
    const std::string &utterance_id,
    const Matrix<BaseFloat> &input,
    const Vector<BaseFloat> *ivector,
    const Matrix<BaseFloat> *online_ivectors,
    int32 online_ivector_period){
  // This function basically does a handshake with one of the decoder threads.
  // It may have to wait till one of the decoder threads becomes ready.
  input_utterance_.utterance_id = utterance_id;
  input_utterance_.input = &input;
  input_utterance_.ivector = ivector;
  input_utterance_.online_ivectors = online_ivectors;
  input_utterance_.online_ivector_period = online_ivector_period;


  UtteranceOutput *this_output = new UtteranceOutput();
  this_output->utterance_id = utterance_id;
  pending_utts_.push_back(this_output);

  input_ready_semaphore_.Signal();
  input_consumed_semaphore_.Wait();
}

int32 NnetBatchDecoder::Finished() {
  is_finished_ = true;
  for (size_t i = 0; i < decode_threads_.size(); i++)
    input_ready_semaphore_.Signal();
  for (size_t i = 0; i < decode_threads_.size(); i++) {
    decode_threads_[i]->join();
    delete decode_threads_[i];
    decode_threads_[i] = NULL;
  }
  // don't clear decode_threads_, since its size is needed in the destructor to
  // compute timing.

  tasks_finished_ = true;
  tasks_ready_semaphore_.Signal();
  compute_thread_.join();
  return num_success_;
}


bool NnetBatchDecoder::GetOutput(
    std::string *utterance_id,
    CompactLattice *clat,
    std::string *sentence) {
  if (!decoder_opts_.determinize_lattice)
    KALDI_ERR << "Don't call this version of GetOutput if you are "
        "not determinizing.";
  while (true) {
    if (pending_utts_.empty())
      return false;
    if (!pending_utts_.front()->finished)
      return false;
    UtteranceOutput *this_output = pending_utts_.front();
    pending_utts_.pop_front();
    if (this_output->compact_lat.NumStates() == 0) {
      delete this_output;
      // ... and continue round the loop, without returning any output to the
      // user for this utterance.  Something went wrong in decoding: for
      // example, the user specified allow_partial == false and no final-states
      // were active on the last frame, or something more unexpected.  A warning
      // would have been printed in the decoder thread.
    } else {
      *clat = this_output->compact_lat;
      utterance_id->swap(this_output->utterance_id);
      sentence->swap(this_output->sentence);
      delete this_output;
      return true;
    }
  }
}


bool NnetBatchDecoder::GetOutput(
    std::string *utterance_id,
    Lattice *lat,
    std::string *sentence) {
  if (decoder_opts_.determinize_lattice)
    KALDI_ERR << "Don't call this version of GetOutput if you are "
        "determinizing.";
  while (true) {
    if (pending_utts_.empty())
      return false;
    if (!pending_utts_.front()->finished)
      return false;
    UtteranceOutput *this_output = pending_utts_.front();
    pending_utts_.pop_front();
    if (this_output->lat.NumStates() == 0) {
      delete this_output;
      // ... and continue round the loop, without returning any output to the
      // user for this utterance.  Something went wrong in decoding: for
      // example, the user specified allow_partial == false and no final-states
      // were active on the last frame, or something more unexpected.  A warning
      // would have been printed in the decoder thread.
    } else {
      *lat = this_output->lat;  // OpenFST has shallow copy so no need to swap.
      utterance_id->swap(this_output->utterance_id);
      sentence->swap(this_output->sentence);
      delete this_output;
      return true;
    }
  }
}

void NnetBatchDecoder::Compute() {
  while (!tasks_finished_) {
    tasks_ready_semaphore_.Wait();
    bool allow_partial_minibatch = true;
    while (computer_->Compute(allow_partial_minibatch));
  }
}

void NnetBatchDecoder::Decode() {
  while (true) {
    input_ready_semaphore_.Wait();
    if (is_finished_)
      return;

    std::vector<NnetInferenceTask> tasks;
    std::string utterance_id;
    // we can be confident that the last element of 'pending_utts_' is the one
    // for this utterance, as we know exactly at what point in the code the main
    // thread will be in AcceptInput().
    UtteranceOutput *output_utterance = pending_utts_.back();
    {
      UtteranceInput input_utterance(input_utterance_);
      utterance_id = input_utterance.utterance_id;
      bool output_to_cpu = true;
      computer_->SplitUtteranceIntoTasks(output_to_cpu,
                                         *(input_utterance.input),
                                         input_utterance.ivector,
                                         input_utterance.online_ivectors,
                                         input_utterance.online_ivector_period,
                                         &tasks);
      KALDI_ASSERT(output_utterance->utterance_id == utterance_id);
      input_consumed_semaphore_.Signal();
      // Now let input_utterance go out of scope; it's no longer valid as it may
      // be overwritten by something else.
    }

    SetPriorities(&tasks);
    for (size_t i = 0; i < tasks.size(); i++)
      computer_->AcceptTask(&(tasks[i]));
    tasks_ready_semaphore_.Signal();

    {
      int32 frame_offset = 0;
      LatticeFasterDecoder decoder(fst_, decoder_opts_);
      decoder.InitDecoding();


      for (size_t i = 0; i < tasks.size(); i++) {
        NnetInferenceTask &task = tasks[i];
        task.semaphore.Wait();
        UpdatePriorityOffset(task.priority);

        SubMatrix<BaseFloat> post(task.output_cpu,
                                  task.num_initial_unused_output_frames,
                                  task.num_used_output_frames,
                                  0, task.output_cpu.NumCols());
        DecodableMatrixMapped decodable(trans_model_, post, frame_offset);
        frame_offset += post.NumRows();
        decoder.AdvanceDecoding(&decodable);
        task.output.Resize(0, 0);  // Free some memory.
      }

      bool use_final_probs = true;
      if (!decoder.ReachedFinal()) {
        if (allow_partial_) {
          KALDI_WARN << "Outputting partial output for utterance "
                     << utterance_id << " since no final-state reached\n";
          use_final_probs = false;
          std::unique_lock<std::mutex> lock(stats_mutex_);
          num_partial_++;
        } else {
          KALDI_WARN << "Not producing output for utterance " << utterance_id
                     << " since no final-state reached and "
                     << "--allow-partial=false.\n";
          std::unique_lock<std::mutex> lock(stats_mutex_);
          num_fail_++;
          continue;
        }
      }
      // if we reached this point, we are getting a lattice.
      decoder.GetRawLattice(&output_utterance->lat, use_final_probs);
      // Let the decoder and the decodable object go out of scope, to save
      // memory.
    }
    ProcessOutputUtterance(output_utterance);
  }
}


void NnetBatchDecoder::UtteranceFailed() {
  std::unique_lock<std::mutex> lock(stats_mutex_);
  num_fail_++;
}

void NnetBatchDecoder::ProcessOutputUtterance(UtteranceOutput *output) {
  fst::Connect(&(output->lat));
  if (output->lat.NumStates() == 0) {
    KALDI_WARN << "Unexpected problem getting lattice for utterance "
               << output->utterance_id;
    std::unique_lock<std::mutex> lock(stats_mutex_);
    num_fail_++;
    return;
  }

  { // This block accumulates diagnostics, prints log messages, and sets
    // output->sentence.
    Lattice best_path;
    LatticeWeight weight;
    ShortestPath(output->lat, &best_path);
    std::vector<int32> alignment;
    std::vector<int32> words;
    GetLinearSymbolSequence(best_path, &alignment, &words, &weight);
    int32 num_frames = alignment.size();
    if (word_syms_ != NULL) {
      std::ostringstream os;
      for (size_t i = 0; i < words.size(); i++) {
        std::string s = word_syms_->Find(words[i]);
        if (s == "")
          KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
        os << s << ' ';
      }
      output->sentence = os.str();
    }
    double likelihood = -(weight.Value1() + weight.Value2());
    // Note: these logging messages will be out-of-order w.r.t. the transcripts
    // that are printed to cerr; we keep those transcripts in the same order
    // that the utterances were in, but these logging messages may be out of
    // order (due to multiple threads).
    KALDI_LOG << "Log-like per frame for utterance " << output->utterance_id
              << " is " << (likelihood / num_frames) << " over "
              << num_frames << " frames.";
    KALDI_VLOG(2) << "Cost for utterance " << output->utterance_id << " is "
                  << weight.Value1() << " + " << weight.Value2();

    std::unique_lock<std::mutex> lock(stats_mutex_);
    tot_like_ += likelihood;
    frame_count_ += num_frames;
    num_success_ += 1;
  }

  if (decoder_opts_.determinize_lattice) {
    if (!DeterminizeLatticePhonePrunedWrapper(
            trans_model_,
            &output->lat,
            decoder_opts_.lattice_beam,
            &(output->compact_lat),
            decoder_opts_.det_opts))
      KALDI_WARN << "Determinization finished earlier than the beam for "
                 << "utterance " << output->utterance_id;
    output->lat.DeleteStates();  // Save memory.
  }

  // We'll write the lattice without acoustic scaling, so we need to reverse
  // the scale that we applied when decoding.
  BaseFloat acoustic_scale = computer_->GetOptions().acoustic_scale;
  if (acoustic_scale != 0.0) {
    if (decoder_opts_.determinize_lattice)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale),
                        &(output->compact_lat));
    else
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale),
                        &(output->lat));
  }
  output->finished = true;
}



NnetBatchDecoder::~NnetBatchDecoder() {
  if (!is_finished_ || !pending_utts_.empty()) {
    // At this point the application is bound to fail so raising another
    // exception is not a big problem.
    KALDI_ERR << "Destroying NnetBatchDecoder object without calling "
        "Finished() and consuming the remaining output";
  }
  // Print diagnostics.

  kaldi::int64 input_frame_count =
      frame_count_ * computer_->GetOptions().frame_subsampling_factor;
  int32 num_threads = static_cast<int32>(decode_threads_.size());

  KALDI_LOG << "Overall likelihood per frame was "
            << tot_like_ / std::max<int64>(1, frame_count_)
            << " over " << frame_count_ << " frames.";

  double elapsed = timer_.Elapsed();
  // the +1 below is just to avoid division-by-zero errors.
  KALDI_LOG << "Time taken "<< elapsed
            << "s: real-time factor assuming 100 frames/sec is "
            << (num_threads * elapsed * 100.0 /
                std::max<int64>(input_frame_count, 1))
            << " (per thread; with " << num_threads << " threads).";
  KALDI_LOG << "Done " << num_success_ << " utterances ("
            << num_partial_ << " forced out); failed for "
            << num_fail_;
}


}  // namespace nnet3
}  // namespace kaldi
