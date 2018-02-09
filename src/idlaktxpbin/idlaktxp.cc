// idlaktxp/idlaktxp.cc

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

/// Example program that runs all modules in idlaktxp and produces
/// XML output
/// You need a text processing database (tpdb) to run this. An example is in
/// ../../idlak-data/arctic-bdl/tpdb
int main(int argc, char *argv[]) {
  const char *usage =
      "Tokenise utf8 input xml\n"
      "Usage:  idlaktxp [options] xml_input xml_output\n"
      "e.g.: ./idlaktxp --pretty --tpdb=../../idlak-data/en/ga ../idlaktxp/test_data/mod-test001.xml output.xml\n" //NOLINT
      "e.g.: cat  ../idlaktxp/test_data/mod-test001.xml output.xml | idlaktxp --pretty --tpdb=../../idlak-data/en/ga - - > output.xml\n"; //NOLINT
  // input output variables
  std::string filein;
  std::string fileout;
  std::string tpdb;
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
    kaldi::TxpTokenise t;
    kaldi::TxpPosTag p;
    kaldi::TxpPauses pz;
    kaldi::TxpPhrasing ph;
    kaldi::TxpPronounce pr;
    kaldi::TxpSyllabify sy;
    // Use pujiXMl to read input file
    pugi::xml_document doc;
    pugi::xml_parse_result r = doc.load(ki.Stream(), pugi::encoding_utf8);
    if (!r) {
      KALDI_ERR << "PugiXML Parse Error in Input Stream" << r.description()
                << "Error offset: " << r.offset;
    }
    // Initialise each module
    t.Init(po);
    p.Init(po);
    pz.Init(po);
    ph.Init(po);
    pr.Init(po);
    sy.Init(po);    
    // Run each module on the input XML
    t.Process(&doc);
    p.Process(&doc);
    pz.Process(&doc);
    ph.Process(&doc);
    pr.Process(&doc);
    sy.Process(&doc);
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
