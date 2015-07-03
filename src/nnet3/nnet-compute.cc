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


NnetComputer::NnetComputer(const NnetComputation &computation,
                           const Nnet &nnet,
                           Nnet *nnet_to_update):
    computation_(computation), nnet_(nnet), nnet_to_update_(nnet_to_update) {
  KALDI_ASSERT(computation.indexes_cuda.empty() &&
               "You must call NnetComputation::ComputeCudaIndexes() before "
               "executing the computation.");
  matrices_.resize(computation.matrices.size());
}

void NnetComputer::ExecuteCommand(int32 command) {
  const NnetComputation::Command &c = computation_.commands[command];
  switch (c.command_type) {
    case NnetComputation::kResizeMatrixZeroed:
      matrices_[c.arg1].Resize(computation_.matrices[c.arg2].num_rows,
                               computation_.matrices[c.arg2].num_cols,
                               kSetZero);
      break;
    case NnetComputation::kResizeMatrixUndefined:
      matrices_[c.arg1].Resize(computation_.matrices[c.arg2].num_rows,
                               computation_.matrices[c.arg2].num_cols,
                               kUndefined);
      break;
    case NnetComputation::kResizeMatrixEmpty:
      matrices_[c.arg1].Resize(0, 0);
      break;
    case NnetComputation::kPropagate: {
      const Component *component = nnet_.GetComponent(c.arg1);
      ComponentPrecomputedIndexes *indexes =
          computation_.component_precomputed_indexes[c.arg2];
      const CuSubMatrix<BaseFloat> input(GetSubMatrix(c.arg3));
      CuSubMatrix<BaseFloat> output(GetSubMatrix(c.arg4));
      component->Propagate(indexes, input, &output);
      break;
    }
    case NnetComputation::kStoreStats: {
      KALDI_ASSERT(nnet_to_update_ != NULL);
      Component *upd_component = nnet_to_update_->GetComponent(c.arg1);
      CuSubMatrix<BaseFloat> output(GetSubMatrix(c.arg2));
      upd_component->StoreStats(output);
      break;
    }
    case NnetComputation::kBackprop: {
      int32 node_index = c.arg1;
      std::ostringstream debug_str;
      KALDI_ASSERT(nnet_to_update_ != NULL);      
      debug_str << "node " << node_index << '['
                << nnet_.GetNodeNames()[node_index] << ']';
      const Component *component = nnet_.GetComponent(c.arg2);
      Component *upd_component = nnet_to_update_->GetComponent(c.arg2);
      ComponentPrecomputedIndexes *indexes =
          computation_.component_precomputed_indexes[c.arg3];
      const CuSubMatrix<BaseFloat> in_value(GetSubMatrix(c.arg4));
      const CuSubMatrix<BaseFloat> out_value(GetSubMatrix(c.arg5));
      CuSubMatrix<BaseFloat> in_deriv(GetSubMatrix(c.arg6));
      const CuSubMatrix<BaseFloat> out_deriv(GetSubMatrix(c.arg7));
      component->Backprop(debug_str.str(), indexes,
                          in_value, out_value, out_deriv, upd_component,
                          &in_deriv);
      break;
    }
    case NnetComputation::kMatrixCopy: {
      const CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg1));
      CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg2));
      dest.CopyFromMat(src);
      break;
    }
    case NnetComputation::kMatrixAdd: {
      const CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg1));
      CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg2));
      dest.AddMat(1.0, src);
      break;
    }
    case NnetComputation::kAddRows: {
      CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
      const CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg2));
      const CuArray<int32> &indexes = computation_.indexes_cuda[c.arg3];
      dest.AddRows(1.0, src, indexes);
      break;
    }
    case NnetComputation::kCopyRows: {
      CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
      const CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg2));
      const CuArray<int32> &indexes = computation_.indexes_cuda[c.arg3];
      dest.CopyRows(src, indexes);
      break;
    }
    case NnetComputation::kCopyRowsMulti: {
      CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
      CuArray<const BaseFloat*> pointers;
      GetPointers(c.arg2, dest.NumCols(), &pointers);
      dest.CopyRows(pointers);
      break;
    }
    case NnetComputation::kCopyToRowsMulti: {
      CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg1));
      CuArray<BaseFloat*> pointers;
      GetPointers(c.arg2, src.NumCols(), &pointers);
      src.CopyToRows(pointers);
      break;
    }
    case NnetComputation::kAddRowsMulti: {
      CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
      CuArray<const BaseFloat*> pointers;
      GetPointers(c.arg2, dest.NumCols(), &pointers);
      dest.AddRows(1.0, pointers);
      break;
    }
    case NnetComputation::kAddToRowsMulti: {
      CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg1));
      CuArray<BaseFloat*> pointers;
      GetPointers(c.arg2, src.NumCols(), &pointers);
      src.AddToRows(1.0, pointers);
      break;
    }
    case NnetComputation::kAddRowRanges: {
      CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
      const CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg2));
      const CuArray<Int32Pair> &pairs = computation_.indexes_multi_cuda[c.arg3];
      dest.AddRowRanges(src, pairs);
      break;
    }
    case NnetComputation::kNoOperation: case NnetComputation::kNoOperationMarker:
      break;
    default:
      KALDI_ERR << "Invalid command in computation";
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

void NnetComputer::Forward() {
  KALDI_ASSERT(computation_.forward_computation_end <=
               computation_.commands.size());
  for (int32 i = 0; i < computation_.forward_computation_end; i++)
    ExecuteCommand(i);
}


void NnetComputer::Backward() {
  KALDI_ASSERT(computation_.forward_computation_end <
               computation_.commands.size());
  int32 size = computation_.commands.size();
  for (int32 i = computation_.forward_computation_end; i < size; i++)
    ExecuteCommand(i);
}


} // namespace nnet3
} // namespace kaldi
