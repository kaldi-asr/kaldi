// nnet/nnet-various.h

// Copyright 2012 Karel Vesely

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


#ifndef KALDI_NNET_VARIOUS_H
#define KALDI_NNET_VARIOUS_H

#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {




/**
 * Expands the time context of the input features
 * in N, out k*N, FrameOffset o_1,o_2,...,o_k
 * FrameOffset example 11frames: -5 -4 -3 -2 -1 0 1 2 3 4 5
 */
class Expand : public Component {
 public:
  Expand(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
    : Component(dim_in, dim_out, nnet)
  { }
  ~Expand()
  { }

  ComponentType GetType() const { 
    return kExpand; 
  }

  void ReadData(std::istream &is, bool binary) {
    //read double vector
    Vector<double> vec_d;
    vec_d.Read(is, binary);
    //convert to int vector
    std::vector<int32> vec_i(vec_d.Dim());
    for(int32 i=0; i<vec_d.Dim(); i++) {
      vec_i[i] = round(vec_d(i));
    }
    //push to GPU
    frame_offsets_.CopyFromVec(vec_i); 
  }

  void WriteData(std::ostream &os, bool binary) {
    std::vector<int32> vec_i;
    frame_offsets_.CopyToVec(&vec_i);
    Vector<double> vec_d(vec_i.size());
    for(int32 i=0; i<vec_d.Dim(); i++) {
      vec_d(i) = vec_i[i];
    }
    vec_d.Write(os, binary);
  }
   
  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    cu::Expand(in, frame_offsets_, out); 
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in_err, CuMatrix<BaseFloat> *out_err) { 
    KALDI_ERR << __func__ << "Not implemented!";
  }

 protected:
  CuStlVector<int32> frame_offsets_;
};



/**
 * Rearrange the matrix columns according to the indices in copy_from_indices_
 */
class Copy : public Component {
 public:
  Copy(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
    : Component(dim_in, dim_out, nnet)
  { }
  ~Copy()
  { }

  ComponentType GetType() const { 
    return kCopy; 
  }

  void ReadData(std::istream &is, bool binary) { 
    //read double vector
    Vector<double> vec_d;
    vec_d.Read(is, binary);
    //sobtract 1
    vec_d.Add(-1.0);
    //convert to int vector
    std::vector<int32> vec_i(vec_d.Dim());
    for(int32 i=0; i<vec_d.Dim(); i++) {
      vec_i[i] = round(vec_d(i));
    }
    //push to GPU
    copy_from_indices_.CopyFromVec(vec_i); 
  }

  void WriteData(std::ostream &os, bool binary) { 
    std::vector<int32> vec_i;
    copy_from_indices_.CopyToVec(&vec_i);
    Vector<double> vec_d(vec_i.size());
    for(int32 i=0; i<vec_d.Dim(); i++) {
      vec_d(i) = vec_i[i];
    }
    vec_d.Add(1.0);
    vec_d.Write(os, binary);
  }
   
  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) { 
    cu::Copy(in,copy_from_indices_,out); 
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &err_in, CuMatrix<BaseFloat> *err_out) { 
    KALDI_ERR << __func__ << "Not implemented!";
  }

 protected:
  CuStlVector<int32> copy_from_indices_;
};


} // namespace kaldi

#endif // KALDI_NNET_VARIOUS_H
