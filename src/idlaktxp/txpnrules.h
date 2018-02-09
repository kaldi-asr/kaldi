// idlaktxp/txpnrules.h

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

#ifndef KALDI_IDLAKTXP_TXPNRULES_H_
#define KALDI_IDLAKTXP_TXPNRULES_H_

// This file defines the normaliser rules class which holds linguistic
// data used to convert tokens to normalised tokens

#include <map>
#include <utility>
#include <string>
#include "base/kaldi-common.h"
#include "idlaktxp/idlak-common.h"
#include "idlaktxp/txpxmldata.h"
#include "idlaktxp/txppcre.h"
#include "idlaktxp/txputf8.h"

namespace kaldi {

/// Maps a regular expression name to a pcre regular expression item
typedef std::map<std::string, pcre*> RgxMap;
/// Regular expression name/ pcre regular expression pair
typedef std::pair<std::string, pcre*> RgxItem;

struct TxpCaseInfo;

/** The nrules object contains regular expressions, lookup tables and
    normalisation rules (not implemented yet)
    Required regular expresisons for tokenisation etc. are also available
    as hard coded defaults
    
    Example of a regex which finds items like 39°

    \verbatim
    <regex name="int_degree">
        <exp>
            <![CDATA[^([0-9]+)(°)$]]>
        </exp>
    </regex>
    \endverbatim

    Example of a lookup table which converts telphone prefixes into the word
    telephone
    \verbatim
  <lookup name="telephone_prefix_table">
    <exp>
      <![CDATA[{"tel":"telephone", "telephone":"telephone", "ph":"phone", "t":"telephone", "phone":"phone"}]]>
    </exp>
  </lookup>
    \endverbatim

    For rule information see \ref idlaktxp_norm
*/  
class TxpNRules: public TxpXmlData {
 public:
  /// Construct without loading voice data
  explicit TxpNRules();
  ~TxpNRules();
  /// Set up hard coded regular expressions and lookups
  void Init(const TxpParseOptions &opts, const std::string &name);
  /// Checks to see what regexs have been found to determine whether to use
  /// hard coded defaults
  bool Parse(const std::string &tpdb);
  /// Given a lookup table name and a key return the value as a
  /// std::string pointer
  const std::string* Lkp(const std::string & name, const std::string & key);
  /// Given a regular expresison name return the pcre regex
  const pcre* GetRgx(const std::string & name);
  /// Copies a token into token, following whitespace into wspace and returns
  /// nput pointer incremented to next token
  const char* ConsumeToken(const char* input,
                            std::string* token,
                            std::string* wspace);
  /// Takes a token as input and breaks off a token delimited by punctuation
  /// This is used in tokenise to split tokens into sub tokens based on
  /// punctuation symbols
  const char* ConsumePunc(const char* input,
                           std::string* prepunc,
                           std::string* token,
                           std::string* pstpunc);
  /// Replace non-ascii punctuation with an ascii equivilent
  void ReplaceUtf8Punc(const std::string & tkin, std::string* tkout);
  /// Replaces characters which are not in the languages lexicon into
  /// equivilents. Downcases upper case and sets case information structure
  void NormCaseCharacter(std::string* norm, TxpCaseInfo & caseinfo);
  /// True if token is only standard lexicon characters (i.e. English a-z)
  bool IsAlpha(const std::string & token);

 private:
  void StartElement(const char* name, const char ** atts);
  void EndElement(const char *);
  void StartCData();
  void EndCData();
  void CharHandler(const char* data, int32 len);
  /// Parses the lookup format in the cdata into a map and adds it to lkps
  int32 MakeLkp(LookupMap *lkps,
                const std::string &name,
                const std::string &cdata);
  /// Parser status currently in cdata element
  bool incdata_;
  /// Buffer for cdata data
  std::string cdata_buffer_;
  /// Regex to find a lookup item
  const pcre* lkp_item_;
  /// Regex to find start of lookup
  const pcre* lkp_open_;
  // Basic tokenisation regexs. Typuically all are overidden in the
  // tpdb data file.
  /// Regex for finding whitespace
  const pcre* rgxwspace_;
  /// Regex for finding individual symbols that separate tokens
  const pcre* rgxsep_;
  /// Regex for finding punctuation symbols
  const pcre* rgxpunc_;
  /// Regex for finding lexicon valid character (e.g. English a-z)
  const pcre* rgxalpha_;
  // Hard coded lookups overridden by normalisation rules if present
  /// Regex for finding whitespace
  const pcre* rgxwspace_default_;
  /// Regex for finding individual symbols that separate tokens
  const pcre* rgxsep_default_;
  /// Regex for finding punctuation symbols
  const pcre* rgxpunc_default_;
  /// Regex for finding lexicon valid character (e.g. English a-z)
  const pcre* rgxalpha_default_;
  /// Map of hard coded lookup tables
  LookupMap locallkps_;
  /// Map of lookup tables in tpdb file
  LookupMap lkps_;
  /// Map of Regexes in tpdb file
  RgxMap rgxs_;
  /// Records type of ellement during expat parse
  std::string elementtype_;
  /// Records element name during expat parse
  std::string elementname_;
};

/// Stores all the character/case information found for a token
struct TxpCaseInfo {
 public:
  TxpCaseInfo()
      : uppercase(false), lowercase(false), symbols(false),
        foreign(false), capitalised(false)
  {}
  /// e.g. A-Z
  bool uppercase;
  /// e.g. a-z
  bool lowercase;
  /// e.g. []\|$
  bool symbols;
  /// e.g. óé
  bool foreign;
  /// e.g. Hello, Matthew
  bool capitalised;
};

}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_TXPNRULES_H_
