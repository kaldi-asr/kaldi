// nnet2/nnet-compute.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)
// Copyright 2015   David Snyder

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

#include "nnet2/nnet-compute.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet2 {

/*
  This class does the forward and possibly backward computation for (typically)
  a whole utterance of contiguous features.  You'll instantiate one of
  these classes each time you want to do this computation.
*/
class NnetComputer {
 public:
  /* Initializer.  If pad == true, pad input with nnet.LeftContext() frames on
     the left and nnet.RightContext() frames on the right (duplicate the first
     and last frames.) */
  NnetComputer(const Nnet &nnet,
               const CuMatrixBase<BaseFloat> &input_feats,
               bool pad, 
               Nnet *nnet_to_update = NULL);
  
  /// The forward-through-the-layers part of the computation.
  void Propagate();
  
  void Backprop(CuMatrix<BaseFloat> *tmp_deriv);
                
  
  /// Computes objf derivative at last layer, and returns objective
  /// function summed over labels and multiplied by utterance_weight.
  /// [Note: utterance_weight will normally be 1.0].
  BaseFloat ComputeLastLayerDeriv(const Posterior &pdf_post,
                                  CuMatrix<BaseFloat> *deriv) const;
  
  CuMatrixBase<BaseFloat> &GetOutput() { return forward_data_.back(); }
  
 private:  
  const Nnet &nnet_;
  std::vector<CuMatrix<BaseFloat> > forward_data_;
  Nnet *nnet_to_update_; // May be NULL, if just want objective function
  // but no gradient info or SGD.
  std::vector <ChunkInfo> chunk_info_;
};

NnetComputer::NnetComputer(const Nnet &nnet,
                           const CuMatrixBase<BaseFloat> &input_feats,
                           bool pad,
                           Nnet *nnet_to_update):
    nnet_(nnet), nnet_to_update_(nnet_to_update) {
  int32 dim = input_feats.NumCols();
  if (dim != nnet.InputDim()) {
    KALDI_ERR << "Feature dimension is " << dim << " but network expects "
              << nnet.InputDim();
  }
  forward_data_.resize(nnet.NumComponents() + 1);

  int32 left_context = (pad ? nnet_.LeftContext() : 0),
       right_context = (pad ? nnet_.RightContext() : 0);

  int32 num_rows = left_context + input_feats.NumRows() + right_context;
  nnet.ComputeChunkInfo(num_rows, 1, &chunk_info_);

  CuMatrix<BaseFloat> &input(forward_data_[0]);
  input.Resize(num_rows, dim);
  input.Range(left_context, input_feats.NumRows(),
              0, dim).CopyFromMat(input_feats);
  for (int32 i = 0; i < left_context; i++)
    input.Row(i).CopyFromVec(input_feats.Row(0));
  int32 last_row = input_feats.NumRows() - 1;
  for (int32 i = 0; i < right_context; i++)
    input.Row(num_rows - i - 1).CopyFromVec(input_feats.Row(last_row));
}


/// This is the forward part of the computation.
void NnetComputer::Propagate() {
  for (int32 c = 0; c < nnet_.NumComponents(); c++) {
    const Component &component = nnet_.GetComponent(c);
    CuMatrix<BaseFloat> &input = forward_data_[c],
                     &output = forward_data_[c+1];
    component.Propagate(chunk_info_[c], chunk_info_[c+1], input, &output);
    const Component *prev_component = (c == 0 ? NULL : &(nnet_.GetComponent(c-1)));
    bool will_do_backprop = (nnet_to_update_ != NULL),
         keep_last_output = will_do_backprop &&
                             ((c>0 && prev_component->BackpropNeedsOutput()) ||
                              component.BackpropNeedsInput());
    if (!keep_last_output)
      forward_data_[c].Resize(0, 0); // We won't need this data; save memory.
  }
}

BaseFloat NnetComputer::ComputeLastLayerDeriv(const Posterior &pdf_post,
                                              CuMatrix<BaseFloat> *deriv) const {
  // TODO: convert this to proper CUDA code, c.f. ComputeObjfAndDeriv
  // in nnet-update.cc (I'm not sure, though, that this code is ever reached.)
  int32 num_components = nnet_.NumComponents();
  double tot_objf = 0.0, tot_weight = 0.0;
  const CuMatrix<BaseFloat> &last_layer_output = forward_data_[num_components];
  int32 num_frames = last_layer_output.NumRows(),
          num_pdfs = last_layer_output.NumCols();
  KALDI_ASSERT(pdf_post.size() == static_cast<size_t>(num_frames));
  deriv->Resize(num_frames, num_pdfs); // will zero it.
  for (int32 i = 0; i < deriv->NumRows(); i++) {
    for (size_t j = 0; j < pdf_post[i].size(); j++) {
      int32 label = pdf_post[i][j].first;
      BaseFloat weight = pdf_post[i][j].second;
      KALDI_ASSERT(label >= 0 && label < num_pdfs);
      BaseFloat this_prob = last_layer_output(i, label);
      KALDI_ASSERT(this_prob > 0.99e-20); // We floored to 1.0e-20 in SoftmaxLayer.
      tot_objf += weight * Log(this_prob);
      tot_weight += weight;
      (*deriv)(i, label) += weight / this_prob; // could be "=", assuming the
      // labels are all distinct.
    }
  }
  KALDI_VLOG(4) << "Objective function is " << (tot_objf/tot_weight) <<
      " per frame over " << tot_weight << " samples.";
  return tot_objf;  
}


void NnetComputer::Backprop(CuMatrix<BaseFloat> *tmp_deriv) {
  KALDI_ASSERT(nnet_to_update_ != NULL); // Or why do backprop?
  // If later this reasoning changes, we can change this
  // statement and add logic to make component_to_update, below,
  // NULL if necessary.
  
  for (int32 c = nnet_.NumComponents() - 1; c >= 0; c--) {
    const Component &component = nnet_.GetComponent(c);
    Component *component_to_update = &(nnet_to_update_->GetComponent(c));
    const CuMatrix<BaseFloat>  &input = forward_data_[c],
                            &output = forward_data_[c+1],
                      &output_deriv = *tmp_deriv;
    CuMatrix<BaseFloat> input_deriv;
    component.Backprop(chunk_info_[c], chunk_info_[c+1], input, output, output_deriv, 
                       component_to_update, &input_deriv);
    *tmp_deriv = input_deriv;
  }
}

void NnetComputation(const Nnet &nnet,
                     const CuMatrixBase<BaseFloat> &input,  // features
                     bool pad_input,
                     CuMatrixBase<BaseFloat> *output) {
  NnetComputer nnet_computer(nnet, input, pad_input, NULL);
  nnet_computer.Propagate();
  output->CopyFromMat(nnet_computer.GetOutput());
}

void NnetComputationChunked(const Nnet &nnet,
                     const CuMatrixBase<BaseFloat> &input,  // features
                     int32 chunk_size,
                     CuMatrixBase<BaseFloat> *output) {
  int32 num_rows,
       num_chunks = ceil((BaseFloat)input.NumRows() / chunk_size),
       dim = input.NumCols(),
       left_context = nnet.LeftContext(),
       right_context = nnet.RightContext();
  CuMatrix<BaseFloat> full_input;
  num_rows = left_context + input.NumRows() + right_context;
  full_input.Resize(num_rows, dim);
  full_input.Range(left_context, input.NumRows(),
            0, dim).CopyFromMat(input);
  for (int32 i = 0; i < left_context; i++)
    full_input.Row(i).CopyFromVec(input.Row(0));
  int32 last_row = input.NumRows() - 1;
  for (int32 i = 0; i < right_context; i++)
    full_input.Row(num_rows - i - 1).CopyFromVec(input.Row(last_row));

  for (int32 i = 0; i < num_chunks; i++) {
    int32 index = i * chunk_size,
          offset = std::min(num_rows - chunk_size * i, 
                            left_context + chunk_size + right_context);
    CuSubMatrix<BaseFloat> chunk_input(full_input, index, offset, 0, dim);
    CuMatrix<BaseFloat> cu_chunk_input(chunk_input);

    // Note: we have already accounted for input padding, so we pass
    // pad_input==false to the NnetComputer.
    NnetComputer nnet_computer(nnet, cu_chunk_input, false, NULL);
    nnet_computer.Propagate();
    CuMatrix<BaseFloat> cu_chunk_output(nnet_computer.GetOutput());
    CuSubMatrix<BaseFloat> chunk_out(*output, i * chunk_size, 
                           cu_chunk_output.NumRows(), 0, 
                           cu_chunk_output.NumCols());
    chunk_out.CopyFromMat(cu_chunk_output);
  }
}

BaseFloat NnetGradientComputation(const Nnet &nnet,
                                  const CuMatrixBase<BaseFloat> &input,
                                  bool pad_input,
                                  const Posterior &pdf_post,
                                  Nnet *nnet_to_update) {
  NnetComputer nnet_computer(nnet, input, pad_input, nnet_to_update);
  nnet_computer.Propagate();
  CuMatrix<BaseFloat> deriv;
  BaseFloat ans;
  ans = nnet_computer.ComputeLastLayerDeriv(pdf_post, &deriv);  
  nnet_computer.Backprop(&deriv);
  return ans;
}


} // namespace nnet2
} // namespace kaldi
