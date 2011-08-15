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

#include "cudannet/nnet-component.h"

#include "cudannet/nnet-nnet.h"
#include "cudannet/nnet-activation.h"
#include "cudannet/nnet-biasedlinearity.h"

namespace kaldi {


const struct Component::key_value Component::kMarkerMap[] = {
  { Component::kBiasedLinearity,"<biasedlinearity>" },
  { Component::kSigmoid,"<sigmoid>" },
  { Component::kSoftmax,"<softmax>" }
};


const char* Component::TypeToMarker(ComponentType t) {
  int32 N=sizeof(kMarkerMap)/sizeof(kMarkerMap[0]);
  for(int i=0; i<N; i++) {
    if(kMarkerMap[i].key == t) 
      return kMarkerMap[i].value;
  }
  KALDI_ERR << "Unknown type" << t;
  return NULL;
}

Component::ComponentType Component::MarkerToType(const std::string& s) {
  int32 N=sizeof(kMarkerMap)/sizeof(kMarkerMap[0]);
  for(int i=0; i<N; i++) {
    if(0 == strcmp(kMarkerMap[i].value,s.c_str())) 
      return kMarkerMap[i].key;
  }
  KALDI_ERR << "Unknown marker" << s;
  return kUnknown;
}


Component* Component::Read(std::istream& is, bool binary, Nnet* nnet) {
  int32 dim_out, dim_in;
  std::string token;

  int first_char = Peek(is,binary);
  if(first_char == EOF) return NULL;

  ReadMarker(is,binary,&token); 
  Component::ComponentType comp_type = Component::MarkerToType(token);

  ReadBasicType(is,binary,&dim_out); 
  ReadBasicType(is,binary,&dim_in);

  Component* p_comp;
  switch(comp_type) {
    case Component::kBiasedLinearity :
      p_comp = new BiasedLinearity(dim_in,dim_out,nnet); 
      break;
    case Component::kSigmoid :
      p_comp = new Sigmoid(dim_in,dim_out,nnet);
      break;
    case Component::kSoftmax :
      p_comp = new Softmax(dim_in,dim_out,nnet);
      break;
    case Component::kUnknown :
    default :
      KALDI_ERR << "Missing type: " << token;
  }

  p_comp->ReadData(is,binary);
  return p_comp;
}


void Component::Write(std::ostream& os, bool binary) const {
  WriteMarker(os,binary,Component::TypeToMarker(GetType()));
  WriteBasicType(os,binary,OutputDim());
  WriteBasicType(os,binary,InputDim());
  if(!binary) os << "\n";
  this->WriteData(os,binary);
}


} // namespace
