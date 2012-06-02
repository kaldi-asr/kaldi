// nnet/nnet-component.cc

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

#include "nnet_cpu/nnet-component.h"

#include "nnet_cpu/nnet-nnet.h"
#include "nnet_cpu/nnet-activation.h"
#include "nnet_cpu/nnet-biasedlinearity.h"

namespace kaldi {


const struct Component::key_value Component::kTokenMap[] = {
  { Component::kBiasedLinearity,"<biasedlinearity>" },
  { Component::kSigmoid,"<sigmoid>" },
  { Component::kSoftmax,"<softmax>" }
};


const char* Component::TypeToToken(ComponentType t) {
  int32 N=sizeof(kTokenMap)/sizeof(kTokenMap[0]);
  for(int i=0; i<N; i++) {
    if (kTokenMap[i].key == t) 
      return kTokenMap[i].value;
  }
  KALDI_ERR << "Unknown type" << t;
  return NULL;
}

Component::ComponentType Component::TokenToType(const std::string &s) {
  int32 N=sizeof(kTokenMap)/sizeof(kTokenMap[0]);
  for(int i=0; i<N; i++) {
    if (0 == strcmp(kTokenMap[i].value, s.c_str())) 
      return kTokenMap[i].key;
  }
  KALDI_ERR << "Unknown token" << s;
  return kUnknown;
}


Component* Component::Read(std::istream &is, bool binary, Nnet *nnet) {
  int32 dim_out, dim_in;
  std::string token;

  int first_char = Peek(is, binary);
  if (first_char == EOF) return NULL;

  ReadToken(is, binary, &token); 
  Component::ComponentType comp_type = Component::TokenToType(token);

  ReadBasicType(is, binary, &dim_out); 
  ReadBasicType(is, binary, &dim_in);

  Component *p_comp;
  switch (comp_type) {
    case Component::kBiasedLinearity :
      p_comp = new BiasedLinearity(dim_in, dim_out, nnet); 
      break;
    case Component::kSigmoid :
      p_comp = new Sigmoid(dim_in, dim_out, nnet);
      break;
    case Component::kSoftmax :
      p_comp = new Softmax(dim_in, dim_out, nnet);
      break;
    case Component::kUnknown :
    default :
      KALDI_ERR << "Missing type: " << token;
  }

  p_comp->ReadData(is, binary);
  return p_comp;
}


void Component::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, Component::TypeToToken(GetType()));
  WriteBasicType(os, binary, OutputDim());
  WriteBasicType(os, binary, InputDim());
  if(!binary) os << "\n";
  this->WriteData(os, binary);
}


} // namespace
