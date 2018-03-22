// nnet3bin/nnet3-xvector-compute.cc

// Copyright 2017   Johns Hopkins University (author: Daniel Povey)
//           2017   Johns Hopkins University (author: Daniel Garcia-Romero)
//           2017   David Snyder

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


#include "nnet3/nnet-xvector-threaded.h"

namespace kaldi {
namespace nnet3 {

XVectorExtractorParallelClass::XVectorExtractorParallelClass(
    const NnetSimpleComputationOptions &opts,
    const Nnet &nnet,
    CachingOptimizingCompiler *compiler, 
    std::string utt,
    const int chunk_size,
    const int min_chunk_size,
    const Matrix<BaseFloat> &feats,
    BaseFloatVectorWriter *xvector_writer
  ):
  opts_(opts),  
  nnet_(&nnet),
  compiler_(*compiler),
  utt_(utt),
  chunk_size_(chunk_size),
  min_chunk_size_(min_chunk_size),
  feats_(feats),
  xvector_writer_(xvector_writer) {
    tot_weight_ = 0.0;
    xvector_avg_.Resize(nnet_->OutputDim("output"), kSetZero);
  }

void XVectorExtractorParallelClass::operator () () {
    // feature chunking
      int32 num_rows = feats_.NumRows(),
            feat_dim = feats_.NumCols(),
            this_chunk_size = chunk_size_;

      if (num_rows < min_chunk_size_) {
        KALDI_WARN << "Minimum chunk size of " << min_chunk_size_
                   << " is greater than the number of rows "
                   << "in utterance: " << utt_;
        // let's make sure client does this check 
        // TODO: exit gracefully 
      } else if (num_rows < chunk_size_) {
        // KALDI_LOG << "Chunk size of " << chunk_size_ << " is greater than "
        //          << "the number of rows in utterance: " << utt_
        //          << ", using chunk size  of " << num_rows;
        this_chunk_size = num_rows;
      } else if (chunk_size_ == -1) {
        this_chunk_size = num_rows;
      }

      int32 num_chunks = ceil(
        num_rows / static_cast<BaseFloat>(this_chunk_size));

      // Iterate over the feature chunks.
      for (int32 chunk_indx = 0; chunk_indx < num_chunks; chunk_indx++) {
        // If we're nearing the end of the input, we may need to shift the
        // offset back so that we can get this_chunk_size frames of input to
        // the nnet.
        int32 offset = std::min(
          this_chunk_size, num_rows - chunk_indx * this_chunk_size);
        if (offset < min_chunk_size_)
          continue;
        SubMatrix<BaseFloat> sub_features(
          feats_, chunk_indx * this_chunk_size, offset, 0, feat_dim);
        Vector<BaseFloat> xvector;
        tot_weight_ += offset;
        
        RunNnetComputation(sub_features, *nnet_, &compiler_, &xvector);

        xvector_avg_.AddVec(offset, xvector);
      }
  }


XVectorExtractorParallelClass::~XVectorExtractorParallelClass () {
    xvector_avg_.Scale(1.0 / tot_weight_);
    xvector_writer_->Write(utt_, xvector_avg_);
}


void XVectorExtractorParallelClass::RunNnetComputation(const MatrixBase<BaseFloat> &features,
    const Nnet &nnet, CachingOptimizingCompiler *compiler,
    Vector<BaseFloat> *xvector) {
      ComputationRequest request;
      request.need_model_derivative = false;
      request.store_component_stats = false;
      request.inputs.push_back(
        IoSpecification("input", 0, features.NumRows()));
      IoSpecification output_spec;
      output_spec.name = "output";
      output_spec.has_deriv = false;
      output_spec.indexes.resize(1);
      request.outputs.resize(1);
      request.outputs[0].Swap(&output_spec);
      // const NnetComputation *computation = compiler->Compile(request);
      std::shared_ptr<const NnetComputation> computation = compiler->Compile(request);
      Nnet *nnet_to_update = NULL;  // we're not doing any update.
      NnetComputer computer(NnetComputeOptions(), *computation,
                      nnet, nnet_to_update);
      CuMatrix<BaseFloat> input_feats_cu(features);
      computer.AcceptInput("input", &input_feats_cu);
      computer.Run();
      CuMatrix<BaseFloat> cu_output;
      computer.GetOutputDestructive("output", &cu_output);
      xvector->Resize(cu_output.NumCols());
      xvector->CopyFromVec(cu_output.Row(0));
    }
}
// nnet3
}
// 
