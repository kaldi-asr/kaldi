// tensor/config.h

// Copyright      2019  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_TENSOR_CONFIG_H_
#define KALDI_TENSOR_CONFIG_H_ 1

#include <string>
#include "util/text-utils.h"

namespace kaldi {
namespace tensor {


/**
   This Config class is used when we want to store configuration information
   inside Variables (e.g., to set per-parameter learning rates).
   We'll eventually need mechanisms to read and write this.
 */
class Config {
 public:

  /**
    This template will be defined only for types
    `T = {std::string, bool, int32, float }`.

      @param [in] key   The name of the config parameter we are setting,
                      e.g. "learning-rate".   Must satisfy IsValidName(key),
                      i.e. starts with `[a-zA-Z]`, and contains only characters
                      `[a-zA-Z0-9_-]`.
      @param [in] value  The value to be set, of type string, bool, int32 or
                      float.  Any previous value (of whatever type) set for this
                      key will be overwritten.
  */
  template<typename T>  void SetValue(const std::string &key,
                                      const T &value);


  /**
    This template will be defined only for types
    `T = {std::string, bool, int32, float }`.

      @param [in] key   The name of the config parameter we are querying,
                      e.g. "learning-rate".   Must satisfy IsValidName(key),
                      i.e. starts with `[a-zA-Z]`, and contains only characters
                      `[a-zA-Z0-9_-]`.
      @param [out] value  The value to be set, of type string, bool, int32 or
                      float.  If the key was not present in the map, we return
                      false and don't set `value`.  If the key was present and
                      the value was of a compatible type, we set `value`.
                      If they key was present but the value was not of a
                      compatible type, we die with an error.
                      As for type compatibility: all types are compatible
                      with themselves, and the only automatic conversion we
                      do (so far) is from int to float.  We may add more
                      conversions later if needed.
      @return         Returns true if the key was in the map and of a compatible
                      type, false if the key was not in the map.  (Dies
                      if the key was present but with an incompatible type).
  */
  template<typename T> bool GetValue(const std::string &key,
                                     T *value);


 private:

  enum ValueType { kStringValue, kBoolValue, kIntValue, kFloatValue };

  struct ConfigElement {
    ValueType value_type;
    std::string str;
    union  {
      int32 i;
      float f;
      bool b;
    } u;
  };

  // If we later end up storing many configuration value, we could change this
  // to unordered_map, but in most cases it will only be one or two so that
  // would be overkill.
  std::map<std::string, ConfigElement> map_;
};


}  // namespace tensor
}  // namespace kaldi


// Include implementation of inline functions.
#include "tensor/config-inl.h"


#endif  // KALDI_TENSOR_CONFIG_H_
