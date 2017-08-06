// nnet3/nnet-compute.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#include <iterator>
#include <sstream>
#include "nnet3/nnet-compute.h"

namespace kaldi {
namespace nnet3 {


NnetComputer::NnetComputer(const NnetComputeOptions &options,
                           const NnetComputation &computation,
                           const Nnet &nnet,
                           Nnet *nnet_to_update):
    options_(options), computation_(computation), nnet_(nnet),
    program_counter_(0), nnet_to_update_(nnet_to_update) {
  KALDI_ASSERT(computation.indexes_cuda.size() == computation.indexes.size() &&
 computation.indexes_ranges_cuda.size() == computation.indexes_ranges.size() &&
               "You must call NnetComputation::ComputeCudaIndexes() before "
               "executing the computation.");
  matrices_.resize(computation.matrices.size());
  debug_ = (options_.debug || GetVerboseLevel() >= 5);
  if (debug_) {
    ComputationVariables variables;
    variables.Init(computation);
    ComputeCommandAttributes(nnet, computation, variables,
                             &command_attributes_);
    std::string preamble;
    computation.GetCommandStrings(nnet, &preamble, &command_strings_);
    KALDI_LOG << preamble;
    computation.GetSubmatrixStrings(nnet, &submatrix_strings_);
  }
}

//static
BaseFloat NnetComputer::MatrixStddev(const CuMatrixBase<BaseFloat> &m) {
  if (m.NumRows() == 0)
    return 0.0;
  return std::sqrt(TraceMatMat(m, m, kTrans) / (m.NumRows() * m.NumCols()));
}

//static
BaseFloat NnetComputer::ParameterStddev(const Component &c) {
  const UpdatableComponent *uc = dynamic_cast<const UpdatableComponent*>(&c);
  KALDI_ASSERT(uc != NULL &&
               "Attempting to get parameter stddev of non-updatable component");
  return std::sqrt(uc->DotProduct(*uc) / uc->NumParameters());
}

void NnetComputer::DebugBeforeExecute(int32 command,
                                      CommandDebugInfo *info) {
  {
    const std::vector<int32> &matrices_written =
        command_attributes_[command].matrices_written;
    size_t size = matrices_written.size();
    info->matrices_written_stddevs.resize(size);
    for (size_t i = 0; i < size; i++) {
      int32 m = matrices_written[i];
      info->matrices_written_stddevs[i] = MatrixStddev(matrices_[m]);
    }
  }
  {
    const std::vector<int32> &submatrices_written =
        command_attributes_[command].submatrices_written;
    size_t size = submatrices_written.size();
    info->submatrices_written_stddevs.resize(size);
    for (size_t i = 0; i < size; i++) {
      int32 s = submatrices_written[i];
      if (!computation_.IsWholeMatrix(s)) {
        const CuSubMatrix<BaseFloat> submat(GetSubMatrix(s));
        info->submatrices_written_stddevs[i] = MatrixStddev(submat);
      }
    }
  }
  const NnetComputation::Command &c = computation_.commands[command];
  if (c.command_type == kBackprop) {
    const Component *component = nnet_.GetComponent(c.arg1);
    if (component->Properties() & kUpdatableComponent)
      info->components_parameter_stddev = ParameterStddev(*component);
  }
}


void NnetComputer::DebugAfterExecute(int32 command,
                                     const CommandDebugInfo &info,
                                     double command_exec_time) {
  std::ostringstream os;
  os << command_strings_[command] << "\t|\t";
  {
    const std::vector<int32> &matrices_written =
        command_attributes_[command].matrices_written;
    size_t size = matrices_written.size();
    KALDI_ASSERT(info.matrices_written_stddevs.size() == size);
    for (size_t i = 0; i < size; i++) {
      int32 m = matrices_written[i];
      BaseFloat old_stddev = info.matrices_written_stddevs[i],
          stddev = MatrixStddev(matrices_[m]);
      os << 'm' << m << ": " << old_stddev << "->" << stddev << " ";
    }
  }
  {
    const std::vector<int32> &submatrices_written =
        command_attributes_[command].submatrices_written;
    size_t size = submatrices_written.size();
    KALDI_ASSERT(info.submatrices_written_stddevs.size() == size);
    for (size_t i = 0; i < size; i++) {
      int32 s = submatrices_written[i];
      if (!computation_.IsWholeMatrix(s)) {
        const CuSubMatrix<BaseFloat> submat(GetSubMatrix(s));
        BaseFloat old_stddev = info.submatrices_written_stddevs[i],
            stddev = MatrixStddev(submat);
        os << submatrix_strings_[s] << ": " << old_stddev << "->"
           << stddev << " ";
      }
    }
  }
  const NnetComputation::Command &c = computation_.commands[command];
  if (c.command_type == kBackprop) {
    const Component *component = nnet_.GetComponent(c.arg1);
    if (component->Properties() & kUpdatableComponent) {
      const std::string &component_name = nnet_.GetComponentName(c.arg1);
      os << component_name << ": " << info.components_parameter_stddev
         << "->" << ParameterStddev(*component) << " ";
    }
  }
  os << "\t|\t time: " << command_exec_time << " secs";
  KALDI_LOG << os.str();
}


void NnetComputer::SaveMemo(int32 memo_index,
                            const Component &c, void *memo) {
  if (memo_index <= 0) {
    if (memo != NULL) {  // memo was returned but is not needed.
      c.DeleteMemo(memo);
    }
  } else {
    if (memos_.size() <= static_cast<size_t>(memo_index))
      memos_.resize(memo_index + 1, NULL);
    memos_[memo_index] = memo;
  }
}

void* NnetComputer::GetMemo(int32 memo_index) {
  if (memo_index == 0) {
    return NULL;
  } else {
    if (static_cast<size_t>(memo_index) >= memos_.size())
      KALDI_ERR << "Memo requested that was not generated.";
    void *ans = memos_[memo_index];
    memos_[memo_index] = NULL;
    return ans;
  }
}




void NnetComputer::ExecuteCommand() {
  const NnetComputation::Command &c = computation_.commands[program_counter_];
  int32 m1, m2;
  try {
    switch (c.command_type) {
      case kAllocMatrixZeroed:
        m1 = computation_.submatrices[c.arg1].matrix_index;
        matrices_[m1].Resize(computation_.matrices[m1].num_rows,
                             computation_.matrices[m1].num_cols,
                             kSetZero,
                             computation_.matrices[m1].stride_type);
        break;
      case kAllocMatrixUndefined:
        m1 = computation_.submatrices[c.arg1].matrix_index;
        matrices_[m1].Resize(computation_.matrices[m1].num_rows,
                             computation_.matrices[m1].num_cols,
                             kUndefined,
                             computation_.matrices[m1].stride_type);
        break;
      case kDeallocMatrix:
        m1 = computation_.submatrices[c.arg1].matrix_index;
        matrices_[m1].Resize(0, 0);
        break;
      case kAllocMatrixFromOther:
        m1 = computation_.submatrices[c.arg1].matrix_index;
        m2 = computation_.submatrices[c.arg2].matrix_index;
        matrices_[m1].Swap(&(matrices_[m2]));
        break;
      case kAllocMatrixFromOtherZeroed:
        m1 = computation_.submatrices[c.arg1].matrix_index;
        m2 = computation_.submatrices[c.arg2].matrix_index;
        matrices_[m1].Swap(&(matrices_[m2]));
        matrices_[m1].SetZero();
        break;
      case kPropagate: {
        const Component *component = nnet_.GetComponent(c.arg1);
        ComponentPrecomputedIndexes *indexes =
            computation_.component_precomputed_indexes[c.arg2].data;
        const CuSubMatrix<BaseFloat> input(GetSubMatrix(c.arg3));
        CuSubMatrix<BaseFloat> output(GetSubMatrix(c.arg4));
        void *memo = component->Propagate(indexes, input, &output);
        if (c.arg6) {  // need to store stats.
          KALDI_ASSERT(nnet_to_update_ != NULL);
          Component *upd_component = nnet_to_update_->GetComponent(c.arg1);
          bool was_in_place = (c.arg3 == c.arg4);
          // if propagate was in-place, provide empty matrix and not 'input', as
          // input is no longer valid.
          const CuSubMatrix<BaseFloat> maybe_input(
              GetSubMatrix(was_in_place ? 0 : c.arg3));
          upd_component->StoreStats(maybe_input, output, memo);
        }
        SaveMemo(c.arg5, *component, memo);
        break;
      }
      case kBackprop:
      case kBackpropNoModelUpdate:  {
        std::ostringstream debug_str;
        KALDI_ASSERT(nnet_to_update_ != NULL);
        debug_str << nnet_.GetComponentName(c.arg1);
        const Component *component = nnet_.GetComponent(c.arg1);
        KALDI_ASSERT(!(computation_.need_model_derivative && !nnet_to_update_));
        Component *upd_component = (nnet_to_update_ &&
                                    c.command_type == kBackprop &&
                                    computation_.need_model_derivative ?
                                    nnet_to_update_->GetComponent(c.arg1) :
                                    NULL);
        ComponentPrecomputedIndexes *indexes =
            computation_.component_precomputed_indexes[c.arg2].data;
        const CuSubMatrix<BaseFloat> in_value(GetSubMatrix(c.arg3));
        const CuSubMatrix<BaseFloat> out_value(GetSubMatrix(c.arg4));
        const CuSubMatrix<BaseFloat> out_deriv(GetSubMatrix(c.arg5));
        CuSubMatrix<BaseFloat> in_deriv(GetSubMatrix(c.arg6));
        void *memo = GetMemo(c.arg7);
        component->Backprop(debug_str.str(), indexes,
                            in_value, out_value, out_deriv,
                            memo, upd_component,
                            c.arg6 == 0 ? NULL : &in_deriv);
        if (memo != NULL)
          component->DeleteMemo(memo);
        break;
      }
      case kMatrixCopy: {
        CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
        const CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg2));
        dest.CopyFromMat(src);
        break;
      }
      case kMatrixAdd: {
        CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
        const CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg2));
        dest.AddMat(1.0, src);
        break;
      }
      case kAddRows: {
        CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
        const CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg2));
        const CuArray<int32> &indexes = computation_.indexes_cuda[c.arg3];
        dest.AddRows(1.0, src, indexes);
        break;
      }
      case kCopyRows: {
        CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
        const CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg2));
        const CuArray<int32> &indexes = computation_.indexes_cuda[c.arg3];
        dest.CopyRows(src, indexes);
        break;
      }
      case kCopyRowsMulti: {
        CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
        CuArray<const BaseFloat*> pointers;
        GetPointers(c.arg2, dest.NumCols(), &pointers);
        dest.CopyRows(pointers);
        break;
      }
      case kCopyToRowsMulti: {
        CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg1));
        CuArray<BaseFloat*> pointers;
        GetPointers(c.arg2, src.NumCols(), &pointers);
        src.CopyToRows(pointers);
        break;
      }
      case kAddRowsMulti: {
        CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
        CuArray<const BaseFloat*> pointers;
        GetPointers(c.arg2, dest.NumCols(), &pointers);
        dest.AddRows(1.0, pointers);
        break;
      }
      case kAddToRowsMulti: {
        CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg1));
        CuArray<BaseFloat*> pointers;
        GetPointers(c.arg2, src.NumCols(), &pointers);
        src.AddToRows(1.0, pointers);
        break;
      }
      case kAddRowRanges: {
        CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
        const CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg2));
        const CuArray<Int32Pair> &pairs = computation_.indexes_ranges_cuda[c.arg3];
        dest.AddRowRanges(src, pairs);
        break;
      }
      case kNoOperation: case kNoOperationPermanent: case kNoOperationMarker:
      case kNoOperationLabel:
        break;
      case kGotoLabel:
        KALDI_ASSERT(computation_.commands[c.arg1].command_type == kNoOperationLabel);
        program_counter_ = c.arg1;
        break;
      default:
        KALDI_ERR << "Invalid command in computation";
    }
  } catch (...) {
    if (!debug_) {
      std::string preamble;
      computation_.GetCommandStrings(nnet_, &preamble, &command_strings_);
      KALDI_WARN << "Printing some background info since error was detected";
      KALDI_LOG << preamble;
      for (int32 prev_c = 0; prev_c < program_counter_; prev_c++)
        KALDI_LOG << command_strings_[prev_c];
    }
    // the following will re-throw the error, but now we've printed more info
    // about what went wrong.
    KALDI_ERR << "Error running command " << command_strings_[program_counter_];
  }
}

CuSubMatrix<BaseFloat> NnetComputer::GetSubMatrix(int32 submatrix_index) {
  KALDI_PARANOID_ASSERT(static_cast<size_t>(submatrix_index) <
                        computation_.submatrices.size());
  const NnetComputation::SubMatrixInfo &info =
      computation_.submatrices[submatrix_index];
  const CuMatrix<BaseFloat> &mat = matrices_[info.matrix_index];
  return CuSubMatrix<BaseFloat>(
      mat, info.row_offset, info.num_rows, info.col_offset, info.num_cols);
}

void NnetComputer::GetPointers(int32 indexes_multi_index,
                               int32 num_cols,
                               CuArray<BaseFloat*> *pointers) {
  KALDI_ASSERT(static_cast<size_t>(indexes_multi_index)
               < computation_.indexes_multi.size());
  const std::vector<std::pair<int32,int32> > &pairs =
      computation_.indexes_multi[indexes_multi_index];
  int32 size = pairs.size();
  std::vector<BaseFloat*> vec(size);

  // the map "lookup" maps from submatrix index to the Data()
  // pointer of that submatrix, and the corresponding Stride().
  unordered_map<int32, std::pair<BaseFloat*, int32> > lookup;

  for (int32 i = 0; i < size; i++) {
    int32 submatrix_index = pairs[i].first, row = pairs[i].second;
    if (submatrix_index != -1) {
      unordered_map<int32, std::pair<BaseFloat*, int32> >::iterator
          iter = lookup.find(submatrix_index);
      if (iter == lookup.end()) {
        CuSubMatrix<BaseFloat> m = GetSubMatrix(submatrix_index);
        lookup[submatrix_index] = std::pair<BaseFloat*, int32>(m.Data(),
                                                               m.Stride());
        iter = lookup.find(submatrix_index);
      }
      BaseFloat *data = iter->second.first;
      int32 stride = iter->second.second;
      vec[i] = data + (row * stride);
    } else {
      // -1 is a marker that will be translated to NULL.
      vec[i] = NULL;
    }
  }
#ifdef KALDI_PARANOID
  for (int32 i = 0; i < size; i += 30 + RandInt(0, 9)) {
    // Do a pseudo-random spot check that the row-indexes are not out of range.
    int32 submatrix_index = pairs[i].first, row = pairs[i].second;
    CuSubMatrix<BaseFloat> m = GetSubMatrix(submatrix_index);
    KALDI_ASSERT(row >= 0 && row < m.NumRows() && num_cols == m.NumCols());
  }
#endif
  pointers->CopyFromVec(vec);
}

void NnetComputer::GetPointers(int32 indexes_multi_index,
                               int32 num_cols,
                               CuArray<const BaseFloat*> *pointers) {
  GetPointers(indexes_multi_index, num_cols,
              reinterpret_cast<CuArray<BaseFloat*>*>(pointers));
}

void NnetComputer::Run() {
  const std::vector<NnetComputation::Command> &c = computation_.commands;
  int32 num_commands = c.size();

  if (program_counter_ >= num_commands) {
    computation_.Print(std::cerr, nnet_);
    KALDI_ERR << "Running computation that has finished: program-counter="
              << program_counter_;
  }
  CheckNoPendingIo();

  CommandDebugInfo info;
  Timer timer;
  double total_elapsed_previous = 0.0;

  for (; program_counter_ < num_commands; program_counter_++) {
    if (c[program_counter_].command_type == kAcceptInput ||
        c[program_counter_].command_type == kProvideOutput) {
      // We have hit a part of the computation that requires user
      // interaction, e.g. the end of the forward or backward phase.
      break;
    }
    if (debug_)
      DebugBeforeExecute(program_counter_, &info);
    ExecuteCommand();
    if (debug_) {
      double total_elapsed_now = timer.Elapsed();
      DebugAfterExecute(program_counter_, info,
                        total_elapsed_now - total_elapsed_previous);
      total_elapsed_previous = total_elapsed_now;
    }
  }
}

void NnetComputer::AcceptInput(const std::string &node_name,
                               CuMatrix<BaseFloat> *input) {
  bool is_output = false;
  int32 matrix_index = GetIoMatrixIndex(node_name, is_output);

  const NnetComputation::MatrixInfo &matrix_info =
      computation_.matrices[matrix_index];
  if (input->NumRows() != matrix_info.num_rows) {
    KALDI_ERR << "Num-rows mismatch for input '" << node_name
              << "': " << matrix_info.num_rows
              <<  " in computation-request, " << input->NumRows()
              << " provided.";
  }
  if (input->NumCols() != matrix_info.num_cols) {
    KALDI_ERR << "Num-cols mismatch for input '" << node_name
              << "': " << matrix_info.num_cols
              <<  " in computation-request, " << input->NumCols()
              << " provided.";
  }
  if (matrix_info.stride_type == kDefaultStride ||
      input->Stride() == input->NumCols()) {
    matrices_[matrix_index].Swap(input);
  } else {
    matrices_[matrix_index].Resize(matrix_info.num_rows,
                                   matrix_info.num_cols,
                                   kUndefined, kStrideEqualNumCols);
    matrices_[matrix_index].CopyFromMat(*input);
    input->Resize(0, 0);
  }
}

const CuMatrixBase<BaseFloat> &NnetComputer::GetOutput(
    const std::string &node_name) {
  bool is_output = true;
  int32 matrix_index = GetIoMatrixIndex(node_name, is_output);
  KALDI_ASSERT(matrices_[matrix_index].NumRows() != 0);
  return matrices_[matrix_index];
}


void NnetComputer::GetOutputDestructive(const std::string &node_name,
                                        CuMatrix<BaseFloat> *output) {
  bool is_output = true;
  int32 matrix_index = GetIoMatrixIndex(node_name, is_output);
  KALDI_ASSERT(matrices_[matrix_index].NumRows() != 0);
  matrices_[matrix_index].Swap(output);
  matrices_[matrix_index].Resize(0, 0);
}


void NnetComputer::CheckNoPendingIo() {
  const std::vector<NnetComputation::Command> &c = computation_.commands;
  while (program_counter_ < static_cast<int32>(c.size()) &&
         (c[program_counter_].command_type == kAcceptInput ||
          c[program_counter_].command_type == kProvideOutput)) {
    pending_commands_.push_back(program_counter_);
    program_counter_++;
  }
  for (size_t i = 0; i < pending_commands_.size(); i++) {
    // the order here doesn't really matter; we go from back to front
    // as it's more efficient, not that efficiency really matters here.
    int32 command = pending_commands_[i];
    if (c[command].command_type == kAcceptInput) {
      // we can't ignore if we needed input from the user that hasn't been
      // provided.
      int32 node = c[command].arg2;
      KALDI_ERR << "Cannot run computation-- we did not get input for node '"
                << nnet_.GetNodeName(node) << "'";
    }
  }
  pending_commands_.clear();
}

int32 NnetComputer::GetIoMatrixIndex(const std::string &node_name, bool is_output) {
  const std::vector<NnetComputation::Command> &c = computation_.commands;
  int32 node_index = nnet_.GetNodeIndex(node_name);
  if (node_index == -1)
    KALDI_ERR << "No node named '" << node_name << "'in network.";
  // first make sure all the I/O commands that we immediately expect, are listed
  // in 'pending_commands_'.
  while (program_counter_ < static_cast<int32>(computation_.commands.size()) &&
         ((c[program_counter_].command_type == kAcceptInput ||
           c[program_counter_].command_type == kProvideOutput ||
           c[program_counter_].command_type == kNoOperationMarker))) {
    if (c[program_counter_].command_type != kNoOperationMarker)
      pending_commands_.push_back(program_counter_);
    program_counter_++;
  }
  for (size_t i = 0; i < pending_commands_.size(); i++) {
    int32 command = pending_commands_[i];
    bool this_command_is_output =
        (c[command].command_type == kProvideOutput);
    int32 this_submatrix_index = c[command].arg1,
        this_node_index = c[command].arg2;
    if (this_command_is_output == is_output && node_index == this_node_index) {
      if (!is_output) {
        pending_commands_.erase(pending_commands_.begin() + i);
        // don't erase the command for outputs, as that would prevent things
        // from being output twice, which is an unnecessary restriction.
      }
      if (!(computation_.IsWholeMatrix(this_submatrix_index)))
        KALDI_ERR << "Getting input or output that is not a whole matrix "
                  << "(probably some optimization code needs to be changed)";
      return computation_.submatrices[this_submatrix_index].matrix_index;
    }
  }
  // if you get the following error it will likely be a bug in the calling code,
  // or possibly due to giving the wrong egs.
  KALDI_ERR << "Could not "
            << (is_output ? "provide output " : "accept input ")
            << "for network node " << node_name
            << " (it is not expected at this point in the computation)";
  return 0;  // Suppress compiler warnings; this line will never be reached.
}


void NnetComputer::AcceptInputs(const Nnet &nnet,
                                const std::vector<NnetIo> &io_vec) {
  for (size_t i = 0; i < io_vec.size(); i++) {
    const NnetIo &io = io_vec[i];
    int32 node_index = nnet.GetNodeIndex(io.name);
    if (node_index == -1)
      KALDI_ERR << "No node named '" << io.name << "' in nnet.";
    if (nnet.IsInputNode(node_index)) {
      CuMatrix<BaseFloat> cu_input(io.features.NumRows(),
                                   io.features.NumCols(),
                                   kUndefined);
      cu_input.CopyFromGeneralMat(io.features);
      this->AcceptInput(io.name, &cu_input);
    }
  }
}

} // namespace nnet3
} // namespace kaldi
