// idlaktxp/txppbreak.h

// Copyright 2012 CereProc Ltd.  (Author: Matthew Aylett)

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
//

#ifndef KALDI_IDLAKTXP_TXPPBREAK_H_
#define KALDI_IDLAKTXP_TXPPBREAK_H_

// This file hold information that relates punctuation to pause insertion

#include <map>
#include <utility>
#include <vector>
#include <string>
#include "base/kaldi-common.h"
#include "idlaktxp/idlak-common.h"
#include "idlaktxp/txpxmldata.h"
#include "idlaktxp/txputf8.h"

namespace kaldi {

struct TxpPbreakInfo;

/// Position of break is when punctuation is before (PRE) or after (PST) token
enum TXPPBREAK_POS {TXPPBREAK_POS_PRE = 0,
                    TXPPBREAK_POS_PST = 1};

/// Lookup of punctuation to break information
typedef std::map<std::string, TxpPbreakInfo> PbreakMap;
/// punctuation/ break information pair
typedef std::pair<std::string, TxpPbreakInfo> PbreakItem;

/// Holds mapping between punctuation characters and what strength
/// (in terms of break index) and time (seconds). Also holds whether this
/// is valid when the punctuation is before or after a token
class TxpPbreak: public TxpXmlData {
 public:
  explicit TxpPbreak() : default_type_(4), default_time_(0.2f) {}
  ~TxpPbreak() {}
  void Init(const TxpParseOptions &opts, const std::string &name) {
    TxpXmlData::Init(opts, "pbreak", name);
  }
  bool Parse(const std::string &tpdb);
  /// Lookup a punctuation symbol and depending whether before or
  /// after token fill info with time and strength if it is stronger or
  /// longer than current values
  bool GetPbreak(const char* punc,
                 enum TXPPBREAK_POS pos,
                 TxpPbreakInfo &info);
  /// Get info for punctuation symbol before token
  const TxpPbreakInfo* GetPbreakPre(const char* punc);
  /// Get info for punctuation symbol after token
  const TxpPbreakInfo* GetPbreakPst(const char* punc);
  /// Given whitespace, column number and hzone info set
  /// flags for repeated or single linebreak
  void GetWhitespaceBreaks(const char* ws, int32 col,
                           bool hzone,
                           int32 hzone_start, int32 hzone_end,
                           bool* newline, bool* newline2);
  /// Return default break strength
  int32 get_default_type() {return default_type_;}
  /// Return default break time
  float32 get_default_time() {return default_time_;}


 private:
  void StartElement(const char* name, const char** atts);
  /// Map of punctuation characters that fire breaks before tokens
  PbreakMap pre_pbreak_;
  /// Map of punctuation characters that fire breaks after tokens
  PbreakMap pst_pbreak_;
  /// Default break strength
  int32 default_type_;
  /// Default break time
  float32 default_time_;
};

/// Structure which holds break type (strength) and break time
struct TxpPbreakInfo {
 public:
  TxpPbreakInfo() : type(0), time(0.0f) {}
  void Clear() {
    type = 0;
    time = 0.0f;
  }
  int32 type;
  float32 time;
};

}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_TXPPBREAK_H_
