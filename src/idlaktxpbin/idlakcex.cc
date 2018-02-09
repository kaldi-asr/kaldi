// idlaktxp/idlakcex.cc

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

#include <pugixml.hpp>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "idlaktxp/idlaktxp.h"

/// Takes output from idalktxp adds structure for pauses
/// and creates full context model names for each phone
///
/// You need a text processing database (tpdb) to run this. An example is in
/// ../../idlak-data/en/ga
int main(int argc, char *argv[]) {
  const char *usage =
      "Tokenise utf8 input xml\n"
      "Usage:  idlakcex [options] xml_input xml_output\n"
      "e.g.: ./idlakcex --pretty --tpdb=../../idlak-data/en/ga ../idlaktxp/test_data/mod-syllabify-out002.xml output.xml\n" //NOLINT
      "e.g.: cat  ../idlaktxp/test_data/mod-syllabify-out002.xml output.xml | idlakcex --pretty --tpdb=../../idlak-data/en/ga - - > output.xml\n"; //NOLINT
  // input output variables
  std::string filein;
  std::string fileout;
  std::string tpdb;
  std::string configf;
  std::string input;
  std::ofstream fout;
  // defaults to non-pretty XML output
  bool pretty = false;

  try {
    kaldi::TxpParseOptions po(usage);
    po.SetTpdb(tpdb);
    po.Register("pretty", &pretty,
                "Output XML with tabbing and line breaks to make it readable");
    po.Read(argc, argv);
    // Must have input and output filenames for XML
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    filein = po.GetArg(1);
    fileout = po.GetArg(2);
    // Set up input/output streams
    bool binary;
    kaldi::Input ki(filein, &binary);
    kaldi::Output kio(fileout, binary);
    // Set up each module
    kaldi::TxpCex cx;
    // Use pujiXMl to read input file
    pugi::xml_document doc;
    pugi::xml_parse_result r = doc.load(ki.Stream(), pugi::encoding_utf8);
    if (!r) {
      KALDI_ERR << "PugiXML Parse Error in Input Stream" << r.description()
                << "Error offset: " << r.offset;
    }
    // Initialise each module
    cx.Init(po);
    // Run each module on the input XML
    cx.Process(&doc);
    // Output result
    if (!pretty)
      doc.save(kio.Stream(), "", pugi::format_raw);
    else
      doc.save(kio.Stream(), "\t");
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
