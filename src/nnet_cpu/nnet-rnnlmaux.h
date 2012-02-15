// nnet/nnet-rnnlmaux.h

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

#ifndef KALDI_NNET_RNNLMAUX_H
#define KALDI_NNET_RNNLMAUX_H

#include "base/io-funcs.h"
#include "util/kaldi-io.h"

#include <map>
#include <fstream>
#include <cstring>

namespace kaldi {


/**
 * Auxiliary class to group Rnnlm functions
 */
class RnnlmAux {
 public:
  static void ReadDict(const std::string& file, std::map<std::string,int32>* dict) {
    bool binary;
    Input in(file,&binary);
    std::string word;
    int32 index;
    while (Peek(in.Stream(),binary) != EOF) {
      ReadBasicType(in.Stream(),binary,&index);
      ReadToken(in.Stream(),binary,&word);
      (*dict)[word] = index;
    }
    in.Close();
  }

  static bool AddLine(std::istream& is, const std::map<std::string,int32>& dict, std::vector<int32>* seq) {

    char line[4096];
    is.getline(line,4096);
    
    const char* delim = " \t";
  
    std::vector<const char*> words;

    //parse the line, check OOVs'
    const char* w = NULL;
    while(NULL != (w = strtok((w==NULL?line:NULL),delim))) {
      if(dict.find(w) == dict.end()) return false; //OOV
      words.push_back(w);
    }
    
    //add line to seq
    for(int32 i=0; i<words.size(); i++) {
      std::string key(words[i]);
      seq->push_back(dict.find(key)->second);
    }
    //add end of sentence token
    seq->push_back(1);

    return true;
  }

};



}

#endif
