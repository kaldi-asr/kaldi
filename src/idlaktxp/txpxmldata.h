// idlaktxp/txpxmldata.h

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

#ifndef KALDI_IDLAKTXP_TXPXMLDATA_H_
#define KALDI_IDLAKTXP_TXPXMLDATA_H_

// This file wraps expat into a class
#include <expat.h>
#include <string>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "idlaktxp/idlak-common.h"
#include "idlaktxp/txpparse-options.h"

namespace kaldi {

/// Base class for all objects which require voice data
///
/// The class wraps expat and has virtual fuctions to handle the
/// XML parsing. In the objects derived from this, a file will typically
/// be built into a bespoke structure for rapid processing.
/// The callback functions need to be static to be set in expat, these then
/// call the virtual functions.
class TxpXmlData {
 public:
  /// Callback functions set in expat for start element
  static void StartElementCB(void* userData, const char* name,
                             const char** atts);
  /// Callback functions set in expat for end element
  static void EndElementCB(void* userData, const char* name);
  /// Callback functions set in expat for handling characters
  static void CharHandlerCB(void* userData, const char* data, int32 len);
  /// Callback functions set in expat for start of Cdata
  /// (used for regular expressions)
  static void StartCDataCB(void* userData);
  /// Callback functions set in expat for end of Cdata
  static void EndCDataCB(void* userData);

  explicit TxpXmlData() : parser_(NULL) {}
  virtual ~TxpXmlData();

  /// Initialise the data structure by processing a tpdb file
  virtual void Init(const TxpParseOptions &opts, const std::string &type,
                    const std::string &name);
  /// Inherited class for bespoke start element handling
  virtual void StartElement(const char* name, const char ** atts) {}
  /// Inherited class for bespoke end element handling
  virtual void EndElement(const char* name) {}
  /// Inherited class for bespoke character handling
  virtual void CharHandler(const char* data, int32 len) {}
  /// Inherited class for bespoke start cdata handling
  virtual void StartCData() {}
  /// Inherited class for bespoke start cdata handling
  virtual void EndCData() {}

  /// Load and parse a file belonging to the object in the tpdb directory
  /// file name is based on the object type and name set on construction
  /// the system first searches spk then acc then region then lang directories
  /// if present. i.e en/ga/bdl en/ga en/region_us en as set in the general
  /// section of the configuration file. If not present they load from the
  /// directory name given (tpdb).
  bool Parse(const std::string &tpdb);
  /// Utility to set a named attribute from an expat array of attribute
  /// key value pairs
  int32 SetAttribute(const char* name, const char ** atts, std::string *val);
  /// Return a general option value
  const char* GetOptValue(const char* key);

 protected:
  /// Options structure as all xmldats classes use the general section 
  const TxpParseOptions * opts_;
  /// Expat parser structure
  XML_Parser parser_;
  /// Type of object e.g. lexicon
  std::string type_;
  /// Name of object e.g. default or user etc.
  std::string name_;
  /// Directory of the tpdb (text processing database) files
  std::string tpdb_;
  /// Final full pathname of input file
  std::string fname_;
  /// Input file is read line by line into this buffer for processing
  std::string buffer_;
};

}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_TXPXMLDATA_H_
