// nnet/nnet-rnnlm.h

// Copyright 2011  Karel Vesely

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



#ifndef KALDI_NNET_RNNLM_H
#define KALDI_NNET_RNNLM_H


#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-nnet.h"

#include <iostream>
#include <sstream>
#include <vector>


namespace kaldi {


/**
 * Rnnlm
 * reuccrent neural network language model
 * the model has 1-of-N input coding,
 * for sake of simplicity and efficiency, 
 * one update per input sequence is performed,
 * no cotexts between sequences are preserved
 */
class Rnnlm {
  //////////////////////////////////////
  // Typedefs
  typedef std::vector<Component*> RnnlmType;
  
 //////////////////////////////////////////////////////////////
 // Disable copy construction and assignment
 private:
  Rnnlm(Rnnlm&); 
  Rnnlm& operator=(Rnnlm&);
   
 //////////////////////////////////////////////////////////////
 // Constructor & Destructor
 public:
  Rnnlm() 
   : learn_rate_(0.0), l2_penalty_(0.0), l1_penalty_(0.0), 
     bptt_(4), preserve_(false) 
  { }

  ~Rnnlm() {
  }

 //////////////////////////////////////////////////////////////
 // Public interface
 public:


  /** Perform forward pass through the network
   *
   * in is a sequence of input symbols w_i that are used for the conditioninig
   *    the symbols are coded as 1..N indices
   * out is a matrix with posteriors probabilities, of symbols, given the inputs
   *    p(y_i|w_i), ie. first row contains the posteriors of the next word 
   *    after the first observed word1
   */ 
  void Propagate(const std::vector<int32>& in, Matrix<BaseFloat>* out);

  /// Perform backward pass through the network
  void Backpropagate(const MatrixBase<BaseFloat>& in_err);

  /// Score a sequence
  BaseFloat Score(const std::vector<int32>& seq, const VectorBase<BaseFloat>* hid = NULL, int32 prev_wrd = 1);

  MatrixIndexT InputDim() const; ///< Dimensionality of the input features
  MatrixIndexT OutputDimCls() const; ///< Dimensionality of the desired vectors
  MatrixIndexT OutputDim() const; ///< Dimensionality of the desired vectors

  /// Get the last hidden layer state
  const VectorBase<BaseFloat>& HidVector() const;
  /// Set hidden layer state for next input sequence
  void HidVector(const VectorBase<BaseFloat>& v);
  /// Get hidden layer states of last sequence
  const Matrix<BaseFloat>& HidMatrix() const;
  
 
  /// Read the MLP from file (can add layers to exisiting instance of Rnnlm)
  void Read(const std::string& file);  
  /// Read the MLP from stream (can add layers to exisiting instance of Rnnlm)
  void Read(std::istream& in, bool binary);  
  /// Write MLP to file
  void Write(const std::string& file, bool binary); 
  /// Write MLP to stream 
  void Write(std::ostream& out, bool binary);    
  
  /// Set the learning rate values to trainable layers, 
  /// factors can disable training of individual layers
  void LearnRate(BaseFloat lrate) {
    learn_rate_ = lrate;
  }
  /// Get the global learning rate value
  BaseFloat LearnRate() { 
    return learn_rate_; 
  }

  void L2Penalty(BaseFloat l2) {
    l2_penalty_ = l2;
  }
  void L1Penalty(BaseFloat l1) {
    l1_penalty_ = l1;
  }

  /// Set the BPTT order (number of timesteps back in time)
  void Bptt(int32 bptt) {
    bptt_ = bptt;
  }
 
  /// Preserve state across sentences
  void PreserveState(bool preserve) {
    preserve_ = preserve;
  }

 private:
  ///////
  //network parameters

  //layer1,input 1 of N coding
  Matrix<BaseFloat> V1_; ///< x->y
  Matrix<BaseFloat> U1_; ///< y(t-1)->y(t)
  Vector<BaseFloat> b1_;

  ///factorization - clustering words to classes
  //:TODO:
  Matrix<BaseFloat> W2cls_;
  Vector<BaseFloat> b2cls_;
  std::vector<int32> last_element_index_;

  ///output weights
  Matrix<BaseFloat> W2_;
  Vector<BaseFloat> b2_;
  //std::vector<SubMatrix<BaseFloat> > W2div_;

  ///////
  //buffers
  std::vector<int32> in_seq_;
  Matrix<BaseFloat> h2_;
  Vector<BaseFloat> h2_last_;
  Matrix<BaseFloat> e2_;
  Matrix<BaseFloat> e2_bptt_[2];

  Vector<BaseFloat> b2_corr_;
  Matrix<BaseFloat> V1_corr_;
  Matrix<BaseFloat> U1_corr_;
  Vector<BaseFloat> b1_corr_;

  ///////
  //global parameters
  BaseFloat learn_rate_; ///< global learning rate
  BaseFloat l2_penalty_; ///< l2 penalty
  BaseFloat l1_penalty_; ///< l1 penalty
  int32 bptt_; ///< bptt order
  bool preserve_; ///< preserve state across sentences
};
  

//////////////////////////////////////////////////////////////////////////
// INLINE FUNCTIONS 
// Rnnlm::
   
inline MatrixIndexT Rnnlm::InputDim() const { 
  return V1_.NumRows();
}

inline MatrixIndexT Rnnlm::OutputDimCls() const {
  return W2cls_.NumCols();
}

inline MatrixIndexT Rnnlm::OutputDim() const {
  return W2_.NumCols();
}



inline const VectorBase<BaseFloat>& Rnnlm::HidVector() const {
  return h2_.Row(h2_.NumRows()-1);
}

inline void Rnnlm::HidVector(const VectorBase<BaseFloat>& v) {
  h2_last_.CopyFromVec(v);
}

inline const Matrix<BaseFloat>& Rnnlm::HidMatrix() const {
  return h2_;
}

 
  
inline void Rnnlm::Read(const std::string& file) {
  bool binary;
  Input in(file,&binary);
  Read(in.Stream(),binary);
  in.Close();
}

inline void Rnnlm::Read(std::istream& in, bool binary) {
  ExpectMarker(in,binary,"<rnnlm_v1.0>");
  ExpectMarker(in,binary,"<v1>");
  in >> V1_; 
  ExpectMarker(in,binary,"<u1>");
  in >> U1_; 
  ExpectMarker(in,binary,"<b1>");
  in >> b1_; 
  ExpectMarker(in,binary,"<w2>");
  in >> W2_; 
  ExpectMarker(in,binary,"<b2>");
  in >> b2_;
  //in >> V1 >> U1 >> b1 >> W2cls >> b2cls >> last_element_index >> W2 >> b2;
  
  //same vocabulary size
  KALDI_ASSERT(V1_.NumRows() == W2_.NumCols());
  //same dim of hidden layer
  KALDI_ASSERT(V1_.NumCols() == U1_.NumCols());
  KALDI_ASSERT(V1_.NumCols() == b1_.Dim());
  //input of layer2 same as output of layer1
  KALDI_ASSERT(V1_.NumCols() == W2_.NumRows());
  KALDI_ASSERT(b2_.Dim() == W2_.NumCols());

  h2_last_.Resize(b1_.Dim());

  b2_corr_.Resize(b2_.Dim());
  V1_corr_.Resize(V1_.NumRows(),V1_.NumCols());
  U1_corr_.Resize(U1_.NumRows(),U1_.NumCols());
  b1_corr_.Resize(b1_.Dim());




}

inline void Rnnlm::Write(const std::string& file, bool binary) {
  Output out(file, binary, true);
  Write(out.Stream(),binary);
  out.Close();
}

inline void Rnnlm::Write(std::ostream& out, bool binary) {
  WriteMarker(out,binary,"<rnnlm_v1.0>");
  WriteMarker(out,binary,"<v1>");
  out << V1_; 
  WriteMarker(out,binary,"<u1>");
  out << U1_; 
  WriteMarker(out,binary,"<b1>");
  out << b1_; 
  WriteMarker(out,binary,"<w2>");
  out << W2_; 
  WriteMarker(out,binary,"<b2>");
  out << b2_;
  //out << V1 << U1 << b1 << W2cls << b2cls << last_element_index << W2 << b2;
}

    



} //namespace kaldi

#endif


