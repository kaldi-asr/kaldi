// nnet/nnet-parallel-component.h

// Copyright 2014  Brno University of Technology (Author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_PARALLEL_COMPONENT_H_
#define KALDI_NNET_NNET_PARALLEL_COMPONENT_H_

#include <string>
#include <vector>
#include <sstream>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"


namespace kaldi {
namespace nnet1 {

class ParallelComponent : public MultistreamComponent {
 public:
  ParallelComponent(int32 dim_in, int32 dim_out):
    MultistreamComponent(dim_in, dim_out)
  { }

  ~ParallelComponent()
  { }

  Component* Copy() const { return new ParallelComponent(*this); }
  ComponentType GetType() const { return kParallelComponent; }

  const Nnet& GetNestedNnet(int32 id) const { return nnet_.at(id); }
  Nnet& GetNestedNnet(int32 id) { return nnet_.at(id); }

  void InitData(std::istream &is) {
    // define options
    std::vector<std::string> nested_nnet_proto;
    std::vector<std::string> nested_nnet_filename;
    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<NestedNnet>" || token == "<NestedNnetFilename>") {
        while (is >> std::ws, !is.eof()) {
          std::string file_or_end;
          ReadToken(is, false, &file_or_end);
          if (file_or_end == "</NestedNnet>" ||
              file_or_end == "</NestedNnetFilename>") break;
          nested_nnet_filename.push_back(file_or_end);
        }
      } else if (token == "<NestedNnetProto>") {
        while (is >> std::ws, !is.eof()) {
          std::string file_or_end;
          ReadToken(is, false, &file_or_end);
          if (file_or_end == "</NestedNnetProto>") break;
          nested_nnet_proto.push_back(file_or_end);
        }
      } else { KALDI_ERR << "Unknown token " << token << ", typo in config?"
                         << " (NestedNnet|NestedNnetFilename|NestedNnetProto)";
      }
    }
    // Initialize,
    // First, read nnets from files,
    if (nested_nnet_filename.size() > 0) {
      for (int32 i = 0; i < nested_nnet_filename.size(); i++) {
        Nnet nnet;
        nnet.Read(nested_nnet_filename[i]);
        nnet_.push_back(nnet);
        KALDI_LOG << "Loaded nested <Nnet> from file : "
                  << nested_nnet_filename[i];
      }
    }
    // Second, initialize nnets from prototypes,
    if (nested_nnet_proto.size() > 0) {
      for (int32 i = 0; i < nested_nnet_proto.size(); i++) {
        Nnet nnet;
        nnet.Init(nested_nnet_proto[i]);
        nnet_.push_back(nnet);
        KALDI_LOG << "Initialized nested <Nnet> from prototype : "
                  << nested_nnet_proto[i];
      }
    }
    // Check dim-sum of nested nnets,
    int32 nnet_input_sum = 0, nnet_output_sum = 0;
    for (int32 i = 0; i < nnet_.size(); i++) {
      nnet_input_sum += nnet_[i].InputDim();
      nnet_output_sum += nnet_[i].OutputDim();
    }
    KALDI_ASSERT(InputDim() == nnet_input_sum);
    KALDI_ASSERT(OutputDim() == nnet_output_sum);
  }

  void ReadData(std::istream &is, bool binary) {
    // read
    ExpectToken(is, binary, "<NestedNnetCount>");
    int32 nnet_count;
    ReadBasicType(is, binary, &nnet_count);
    for (int32 i = 0; i < nnet_count; i++) {
      ExpectToken(is, binary, "<NestedNnet>");
      int32 dummy;
      ReadBasicType(is, binary, &dummy);
      Nnet nnet;
      nnet.Read(is, binary);
      nnet_.push_back(nnet);
    }
    ExpectToken(is, binary, "</ParallelComponent>");

    // check dim-sum of nested nnets
    int32 nnet_input_sum = 0, nnet_output_sum = 0;
    for (int32 i = 0; i < nnet_.size(); i++) {
      nnet_input_sum += nnet_[i].InputDim();
      nnet_output_sum += nnet_[i].OutputDim();
    }
    KALDI_ASSERT(InputDim() == nnet_input_sum);
    KALDI_ASSERT(OutputDim() == nnet_output_sum);
  }

  void WriteData(std::ostream &os, bool binary) const {
    // useful dims
    int32 nnet_count = nnet_.size();
    //
    WriteToken(os, binary, "<NestedNnetCount>");
    WriteBasicType(os, binary, nnet_count);
    if (!binary) os << "\n";
    for (int32 i = 0; i < nnet_count; i++) {
      WriteToken(os, binary, "<NestedNnet>");
      WriteBasicType(os, binary, i+1);
      if (!binary) os << "\n";
      nnet_[i].Write(os, binary);
    }
    WriteToken(os, binary, "</ParallelComponent>");
  }

  int32 NumParams() const {
    int32 ans = 0;
    for (int32 i = 0; i < nnet_.size(); i++) {
      ans += nnet_[i].NumParams();
    }
    return ans;
  }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    int32 offset = 0;
    for (int32 i = 0; i < nnet_.size(); i++) {
      int32 n_params = nnet_[i].NumParams();
      Vector<BaseFloat> gradient_aux;  // we need 'Vector<>',
      nnet_[i].GetGradient(&gradient_aux);  // copy gradient from Nnet,
      gradient->Range(offset, n_params).CopyFromVec(gradient_aux);
      offset += n_params;
    }
    KALDI_ASSERT(offset == NumParams());
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    int32 offset = 0;
    for (int32 i = 0; i < nnet_.size(); i++) {
      int32 n_params = nnet_[i].NumParams();
      Vector<BaseFloat> params_aux;  // we need 'Vector<>',
      nnet_[i].GetParams(&params_aux);  // copy params from Nnet,
      params->Range(offset, n_params).CopyFromVec(params_aux);
      offset += n_params;
    }
    KALDI_ASSERT(offset == NumParams());
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    int32 offset = 0;
    for (int32 i = 0; i < nnet_.size(); i++) {
      int32 n_params = nnet_[i].NumParams();
      nnet_[i].SetParams(params.Range(offset, n_params));
      offset += n_params;
    }
    KALDI_ASSERT(offset == NumParams());
  }

  std::string Info() const {
    std::ostringstream os;
    os << "\n";
    for (int32 i = 0; i < nnet_.size(); i++) {
      os << "nested_network #" << i+1 << " {\n"
         << nnet_[i].Info()
         << "}\n";
    }
    std::string s(os.str());
    s.erase(s.end() -1);  // removing last '\n'
    return s;
  }

  std::string InfoGradient() const {
    std::ostringstream os;
    os << "\n";
    for (int32 i = 0; i < nnet_.size(); i++) {
      os << "nested_gradient #" << i+1 << " {\n"
         << nnet_[i].InfoGradient(false)
         << "}\n";
    }
    std::string s(os.str());
    s.erase(s.end() -1);  // removing last '\n'
    return s;
  }

  std::string InfoPropagate() const {
    std::ostringstream os;
    for (int32 i = 0; i < nnet_.size(); i++) {
      os << "nested_propagate #" << i+1 << " {\n"
         << nnet_[i].InfoPropagate(false)
         << "}\n";
    }
    return os.str();
  }

  std::string InfoBackPropagate() const {
    std::ostringstream os;
    for (int32 i = 0; i < nnet_.size(); i++) {
      os << "nested_backpropagate #" << i+1 << " {\n"
         << nnet_[i].InfoBackPropagate(false)
         << "}\n";
    }
    return os.str();
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // column-offsets for data buffers 'in,out',
    int32 input_offset = 0, output_offset = 0;
    // loop over nnets,
    for (int32 i = 0; i < nnet_.size(); i++) {
      // get the data 'windows',
      CuSubMatrix<BaseFloat> src(
        in.ColRange(input_offset, nnet_[i].InputDim())
      );
      CuSubMatrix<BaseFloat> tgt(
        out->ColRange(output_offset, nnet_[i].OutputDim())
      );
      // forward through auxiliary matrix, as 'Propagate' requires 'CuMatrix',
      CuMatrix<BaseFloat> tgt_aux;
      nnet_[i].Propagate(src, &tgt_aux);
      tgt.CopyFromMat(tgt_aux);
      // advance the offsets,
      input_offset += nnet_[i].InputDim();
      output_offset += nnet_[i].OutputDim();
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // column-offsets for data buffers 'in,out',
    int32 input_offset = 0, output_offset = 0;
    // loop over nnets,
    for (int32 i = 0; i < nnet_.size(); i++) {
      // get the data 'windows',
      CuSubMatrix<BaseFloat> src(
        out_diff.ColRange(output_offset, nnet_[i].OutputDim())
      );
      CuSubMatrix<BaseFloat> tgt(
        in_diff->ColRange(input_offset, nnet_[i].InputDim())
      );
      // ::Backpropagate through auxiliary matrix (CuMatrix in the interface),
      CuMatrix<BaseFloat> tgt_aux;
      nnet_[i].Backpropagate(src, &tgt_aux);
      tgt.CopyFromMat(tgt_aux);
      // advance the offsets,
      input_offset += nnet_[i].InputDim();
      output_offset += nnet_[i].OutputDim();
    }
  }

  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    { }  // do nothing
  }

  /**
   * Overriding the default,
   * which was UpdatableComponent::SetTrainOptions(...)
   */
  void SetTrainOptions(const NnetTrainOptions &opts) {
    for (int32 i = 0; i < nnet_.size(); i++) {
      nnet_[i].SetTrainOptions(opts);
    }
  }

  /**
   * Overriding the default,
   * which was UpdatableComponent::SetLearnRateCoef(...)
   */
  void SetLearnRateCoef(BaseFloat val) {
    // loop over nnets,
    for (int32 i = 0; i < nnet_.size(); i++) {
      // loop over components,
      for (int32 j = 0; j < nnet_[i].NumComponents(); j++) {
        if (nnet_[i].GetComponent(j).IsUpdatable()) {
          UpdatableComponent& comp =
            dynamic_cast<UpdatableComponent&>(nnet_[i].GetComponent(j));
          // set the value,
          comp.SetLearnRateCoef(val);
        }
      }
    }
  }

  /**
   * Overriding the default,
   * which was UpdatableComponent::SetBiasLearnRateCoef(...)
   */
  void SetBiasLearnRateCoef(BaseFloat val) {
    // loop over nnets,
    for (int32 i = 0; i < nnet_.size(); i++) {
      // loop over components,
      for (int32 j = 0; j < nnet_[i].NumComponents(); j++) {
        if (nnet_[i].GetComponent(j).IsUpdatable()) {
          UpdatableComponent& comp =
            dynamic_cast<UpdatableComponent&>(nnet_[i].GetComponent(j));
          // set the value,
          comp.SetBiasLearnRateCoef(val);
        }
      }
    }
  }

  /**
   * Overriding the default,
   * which was MultistreamComponent::SetSeqLengths(...)
   */
  void SetSeqLengths(const std::vector<int32> &sequence_lengths) {
    sequence_lengths_ = sequence_lengths;
    // loop over nnets,
    for (int32 i = 0; i < nnet_.size(); i++) {
      nnet_[i].SetSeqLengths(sequence_lengths);
    }
  }

 private:
  std::vector<Nnet> nnet_;
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_PARALLEL_COMPONENT_H_
