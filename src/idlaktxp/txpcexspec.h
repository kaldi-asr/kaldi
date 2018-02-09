// idlaktxp/txpcexspec.h

// Copyright 2013 CereProc Ltd.  (Author: Matthew Aylett)

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

#ifndef KALDI_IDLAKTXP_TXPCEXSPEC_H_
#define KALDI_IDLAKTXP_TXPCEXSPEC_H_

// This file defines the feature extraction system

#include <pugixml.hpp>
#include <deque>
#include <utility>
#include <string>
#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "idlaktxp/idlak-common.h"
#include "idlaktxp/txpxmldata.h"

namespace kaldi {

// whether to allow cross pause context across breaks < 4
// SPT (default) no, UTT yes.
enum CEXSPECPAU_HANDLER {CEXSPECPAU_HANDLER_SPURT = 0,
                         CEXSPECPAU_HANDLER_UTTERANCE = 1};

// Is the feature result a string or an integer
enum CEXSPEC_TYPE {CEXSPEC_TYPE_STR = 0,
                   CEXSPEC_TYPE_INT = 1};


// Default maximum size of a feature in bytes
// (can be set in cex-<architecture>.xml)
#define CEXSPEC_MAXFIELDLEN 5
// Default error code
#define CEXSPEC_ERROR "ERROR"
// Default pause handling - SPT means have two sil models between
// every phrase - HTS menas use a single sil model within utterances
#define CEXSPEC_PAUSEHANDLING "SPT"
// Default null value for features
#define CEXSPEC_NULL "xx"

// TODO(MPA): add padding control so that all features value strings can be
// set to the same length so that it is easier to read and compare them visually


class TxpCexspec;
struct TxpCexspecFeat;
class TxpCexspecModels;
class TxpCexspecContext;

// moving vector which keeps context for each context level
typedef std::deque<pugi::xml_node> XmlNodeVector;

// a feature function
typedef bool (* cexfunction)
(const TxpCexspec*, const TxpCexspecFeat*,
 const TxpCexspecContext*, std::string*);

// array of feature functiion names
extern const char* const CEXFUNCLBL[];
// array of feature functiion pointers
extern const cexfunction CEXFUNC[];
// array of feature function types
extern const enum CEXSPEC_TYPE CEXFUNCTYPE[];

/// lookup valid values from set name
typedef std::map<std::string, StringSet> LookupMapSet;
/// valid values/ set name pair
typedef std::pair<std::string, StringSet> LookupMapSetItem;
/// vector feature structures in architecture
typedef std::vector<TxpCexspecFeat> TxpCexspecFeatVector;

class TxpCexspec: public TxpXmlData {
 public:
  explicit TxpCexspec() :
      TxpXmlData(),
      cexspec_maxfieldlen_(CEXSPEC_MAXFIELDLEN),
      pauhand_(CEXSPECPAU_HANDLER_SPURT) {}
  ~TxpCexspec() {}
  void Init(const TxpParseOptions &opts, const std::string &name) {
    TxpXmlData::Init(opts, "cex", name);
  }
  // calculate biggest buffer required for feature output
  int32 MaxFeatureSize();
  // return pause handling strategy
  enum CEXSPECPAU_HANDLER GetPauseHandling() {return pauhand_;}
  // add pause structure to an XML document
  int32 AddPauseNodes(pugi::xml_document* doc);
  // call feature function and deal with pause behaviour
  bool ExtractFeatures(const TxpCexspecContext &context, std::string* buf);
  // check and append value - function string
  bool AppendValue(const TxpCexspecFeat &feat, bool error,
                   const char* s, std::string* buf) const;
  // check and append value - function integer
  bool AppendValue(const TxpCexspecFeat &feat, bool error,
                   int32 i, std::string* buf) const;
  // append a null value
  bool AppendNull(const TxpCexspecFeat &feat, std::string* buf) const;
  // append an error value
  bool AppendError(const TxpCexspecFeat &feat, std::string* buf) const;
  // return feature specific mapping between cex value and desired value
  const std::string Mapping(const TxpCexspecFeat &feat,
                            const std::string &instr) const;
  // set cex extraction function info
  void GetFunctionSpec(pugi::xml_node * header);

 private:
  // Parser for tpdb xml cex setup
  void StartElement(const char* name, const char** atts);
  // return index of a feature function by name
  int32 GetFeatureIndex(const std::string &name);
  // stores valid values for string based features
  LookupMapSet sets_;
  // stores null values for string based features
  LookupMap setnull_;
  // stores information on current feature architecture
  TxpCexspecFeatVector cexspecfeats_;
  // lookup for feature name to index of cexspecfeats_
  LookupInt cexspecfeatlkp_;
  // maximum feature field length
  int32 cexspec_maxfieldlen_;
  // pause handling
  enum CEXSPECPAU_HANDLER pauhand_;
  // used while parsing input XML to keep track of current set
  std::string curset_;
  // used while parsing input XML to keep track of current cexspec function
  std::string curfunc_;
};

// hold information on a feature function defined in cex-<architecture>.xml
struct TxpCexspecFeat {
  // name of feature
  std::string name;
  // htsname of feature (for information only)
  std::string htsname;
  // description of function (for information only)
  std::string desc;
  // delimiter used before feature in model name
  std::string delim;
  // value when no feature value is meaningful
  std::string nullvalue;
  // pointer to the extraction function
  cexfunction func;
  // whether to allow cross silence context pause
  bool pause_context;
  // the type of fuction (string or integer)
  enum CEXSPEC_TYPE type;
  // name of the valid set of values if a string type fucntion
  std::string set;
  // maximum value if an integer type value
  int32 max;
  // minimum value if an integer type value
  int32 min;
  // mapping from specific feature extraction values
  // to architecture specific values
  LookupMap mapping;
};

// container for a feature output full context HMM modelnames
class TxpCexspecModels {
 public:
  explicit TxpCexspecModels() {}
  ~TxpCexspecModels();
  // initialise model name output
  void Init(TxpCexspec *cexspec);
  // clear container for reuse
  void Clear();
  // append an empty model
  char* Append();
  // return total number of phone models produces by XML input
  int GetNoModels() {return models_.size();}
  // return a model name
  const char* GetModel(int idx) {return models_[idx];}
 private:
  // vector or locally allocated buffers each for a model
  CharPtrVector models_;
  // maximum buffer length required based on feature achitecture
  int32 buflen_;
};

// iterator for accessing linguistic structure in the XML document
class TxpCexspecContext {
 public:
  explicit TxpCexspecContext(const pugi::xml_document &doc,
                      enum CEXSPECPAU_HANDLER pauhand);
  ~TxpCexspecContext() {}
  // iterate to next item
  bool Next();
  // are we in a silence
  bool isBreak() {return isbreak_;}
  // is the silence doucument internal
  bool isBreakInternal() {return internalbreak_;}
  // is the break at the end or begining of a spt
  bool isEndBreak() {return endbreak_;}
  // is the break between sentences
  bool isUttBreak() {return uttbreak_;}
  // return phon back or forwards from current phone
  pugi::xml_node GetPhone(const int32 idx, const bool pause_context) const;
  // return parent syllable back or forwards from current phone
  pugi::xml_node GetSyllable(const int32 idx, const bool pause_context) const;
  // return parent token back or forwards from current phone
  pugi::xml_node GetWord(const int32 idx, const bool pause_context) const;
  // return parent spurt back or forwards from current phone
  pugi::xml_node GetSpurt(const int32 idx, const bool pause_context) const;
  // look up from the node until we find the correct current context node
  pugi::xml_node GetContextUp(const pugi::xml_node &node,
                              const char* name) const;
  
 private:
  bool isbreak_;
  bool endbreak_;
  bool internalbreak_;
  bool uttbreak_;
  enum CEXSPECPAU_HANDLER pauhand_;
  pugi::xpath_node_set phones_;
  pugi::xpath_node_set syllables_;
  pugi::xpath_node_set words_;
  pugi::xpath_node_set spurts_;
  pugi::xpath_node_set utterances_;
  pugi::xpath_node_set::const_iterator current_phone_;
  pugi::xpath_node_set::const_iterator current_syllable_;
  pugi::xpath_node_set::const_iterator current_word_;
  pugi::xpath_node_set::const_iterator current_spurt_;
  pugi::xpath_node_set::const_iterator current_utterance_;
  // phone contexts
  XmlNodeVector phone_contexts_;
  // syl contexts
  XmlNodeVector syllable_contexts_;
  // wrd contexts
  XmlNodeVector word_contexts_;
  // spt contexts
  XmlNodeVector spurt_contexts_;
  // utt contexts
  XmlNodeVector utterance_contexts_;
};

}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_TXPCEXSPEC_H_
