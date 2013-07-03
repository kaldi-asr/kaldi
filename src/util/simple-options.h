// util/simple-options.hh

// Copyright 2013  Tanel Alumae, Tallinn University of Technology

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

#ifndef KALDI_UTIL_SIMPLE_OPTIONS_H_
#define KALDI_UTIL_SIMPLE_OPTIONS_H_

#include <map>
#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "itf/options-itf.h"

namespace kaldi {

class SimpleOptions : public OptionsItf {
 public:
   SimpleOptions() {}
   
   ~SimpleOptions() {}
   
  // Methods from the interface
  void Register(const std::string &name,
                bool *ptr, const std::string &doc); 
  void Register(const std::string &name,
                int32 *ptr, const std::string &doc); 
  void Register(const std::string &name,
                uint32 *ptr, const std::string &doc); 
  void Register(const std::string &name,
                float *ptr, const std::string &doc); 
  void Register(const std::string &name,
                double *ptr, const std::string &doc); 
  void Register(const std::string &name,
                std::string *ptr, const std::string &doc); 
   
  // set option with the specified key, return true if successful
  bool SetOption(const std::string &key, const bool &value);
  bool SetOption(const std::string &key, const int32 &value);
  bool SetOption(const std::string &key, const uint32 &value);
  bool SetOption(const std::string &key, const float &value);
  bool SetOption(const std::string &key, const double &value);
  bool SetOption(const std::string &key, const std::string &value);
  bool SetOption(const std::string &key, const char* value);
  
  // get option with the specified key and put to 'value', return true if successful
  bool GetOption(const std::string &key, bool &value);
  bool GetOption(const std::string &key, int32 &value);
  bool GetOption(const std::string &key, uint32 &value);
  bool GetOption(const std::string &key, float &value);
  bool GetOption(const std::string &key, double &value);
  bool GetOption(const std::string &key, std::string &value);
  
  
  enum OptionType {
    BOOL,
    INT32,
    UINT32,
    FLOAT,
    DOUBLE,
    STRING
  };
    
  struct OptionInfo {
    OptionInfo(const std::string &doc, OptionType type, void * pointer)
      : doc(doc), type(type) {}
    std::string doc;
    OptionType type;
    void *pointer;
  };  
  
  std::vector<std::pair<std::string, OptionInfo> > GetOptionInfos();
  
  bool GetOptionType(const std::string &key, OptionType &type);
  
 private:
 
  std::vector<std::pair<std::string, OptionInfo> > option_infos_;
   
  // maps for option variables
  std::map<std::string, bool*> bool_map_;
  std::map<std::string, int32*> int_map_;
  std::map<std::string, uint32*> uint_map_;
  std::map<std::string, float*> float_map_;
  std::map<std::string, double*> double_map_;
  std::map<std::string, std::string*> string_map_;
};

}   // namespace kaldi

#endif  // KALDI_UTIL_SIMPLE_OPTIONS_H_
