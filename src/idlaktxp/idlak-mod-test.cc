// idlaktxp/idlak-mod-test.cc

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

// Performs regression test on a fixed set of idlak modules

// test files are of the form
// mod-testNNN.xml
// i.e. mod-test000.xml
// and in ascending order files are processed until it can't find
// one to avoid operating system dependent readdir functions
// regression data if present are of the form
// mod-<MODULENAME>-regNNN.xml
// ie mod-tokenise-reg000.xml
// output files of the form
// mod-<MODULENAME>-outNNN.xml
// ie mod-tokenise-out000.xml
// and with indentation for readability
// mod-<MODULENAME>-outNNN-verbose.xml
// ie mod-tokenise-out000-verbose.xml


#include <pugixml.hpp>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "idlaktxp/idlaktxp.h"

static void getTestFileName(int fno, std::string* fname,
                            const std::string &basename,
                            const std::string &ext);

static bool canOpenIn(const std::string & fname);

static bool testModule(kaldi::TxpModule* mod, const std::string &dirin,
                       const std::string &input, kaldi::int32 fno,
                       std::string* new_output_file);

static bool regressionTest(const std::string &output,
                           const std::string &regression);

int main(int argc, char *argv[]) {
  const char *usage =
      "Test IDLAK Module\n"
      "Usage:  idlak-mod-test xml_input_dir xml_output_dir\n";

  std::string dirin = "test_data";
  std::string dirout = "test_data";
  std::string filein;
  std::string fileout;
  std::string filereg;
  std::string tpdb = "../../idlak-data/en/ga";
  std::string configf;
  kaldi::int32 i, fno = 0;
  bool error = false, anyerror = false;
  std::vector<kaldi::TxpModule*> modules;

  try {
    kaldi::TxpParseOptions po(usage);
    po.SetTpdb(tpdb);
    po.Read(argc, argv);
    if (po.NumArgs() == 2) {
      dirin = po.GetArg(1);
      dirout = po.GetArg(2);
    }

    kaldi::TxpTokenise t;
    modules.push_back(&t);
    kaldi::TxpPauses pz;
    modules.push_back(&pz);
    kaldi::TxpPosTag p;
    modules.push_back(&p);
    kaldi::TxpPhrasing ph;
    modules.push_back(&ph);
    kaldi::TxpPronounce pr;
    modules.push_back(&pr);
    kaldi::TxpSyllabify sy;
    modules.push_back(&sy);
    kaldi::TxpCex cx;
    modules.push_back(&cx);

    for (i = 0; i < modules.size(); i++) {
      modules[i]->Init(po);
    }
    // increment through test files until we can't open one
    while (1) {
      getTestFileName(fno, &filein, dirin + "/mod-test", std::string(".xml"));
      if (!canOpenIn(filein)) break;
      for (i = 0; i < modules.size(); i++) {
        error = testModule(modules[i], dirin, filein, fno, &filein);
        if (error) anyerror = true;
      }
      fno++;
    }
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
  return anyerror;
}

static void getTestFileName(int fno, std::string* fname,
                        const std::string &basename,
                        const std::string &ext) {
  std::ostringstream stream;
  std::string zeros = "000";
  std::string fnostr;
  stream.clear();
  stream << zeros << fno;
  fnostr = stream.str();
  fnostr = fnostr.substr(fnostr.size() - 3);
  *fname = basename + fnostr + ext;
}

static bool canOpenIn(const std::string & fname) {
    std::fstream filetest;
    filetest.open(fname.c_str(), std::fstream::in);
    if (filetest.fail()) return false;
    filetest.close();
    return true;
}

static bool testModule(kaldi::TxpModule* mod, const std::string &dirin,
                       const std::string &input, kaldi::int32 fno,
                       std::string* new_output_file) {
  bool error = false;
  const std::string* name;
  std::string filereg;
  std::string file_output_verbose;
  bool binary;

  name = &mod->GetName();
  kaldi::Input ki(input, &binary);
  std::cout << "MODULE:" << mod->GetName() << " INPUT: " << input << std::endl;
  pugi::xml_document doc;
  pugi::xml_parse_result r = doc.load(ki.Stream(),
                                      pugi::encoding_utf8 |
                                      pugi::parse_ws_pcdata |
                                      pugi::parse_escapes);
  if (!r) {
    KALDI_ERR << "PugiXML Parse Error in Input Stream" << r.description()
              << "Error offset: " << r.offset;
    return true;
  }
  mod->Process(&doc);
  getTestFileName(fno, new_output_file, dirin + "/mod-" + *name + "-out",
                  std::string(".xml"));
  kaldi::Output kio(*new_output_file, binary);
  getTestFileName(fno, &file_output_verbose, dirin + "/mod-" + *name + "-outv",
                  std::string(".xml"));
  kaldi::Output kiov(file_output_verbose, binary);
  doc.save(kiov.Stream(), "\t");
  kiov.Close();
  doc.save(kio.Stream(), "", pugi::format_raw);
  kio.Close();
  getTestFileName(fno, &filereg, dirin + "/mod-" + *name + "-reg",
                  std::string(".xml"));
  if (canOpenIn(filereg)) {
    std::cout << "REGRESSION TEST: " << file_output_verbose << " " << filereg << std::endl << std::flush;
    if (!regressionTest(file_output_verbose, filereg)) error = true;
  }
  return error;
}

static bool regressionTest(const std::string &output,
                           const std::string &regression) {
  std::fstream outstr;
  std::fstream regstr;
  std::string outln;
  std::string regln;
  kaldi::int32 lno = 1;
  bool passed = true;
  outstr.open(output.c_str(), std::fstream::in);
  regstr.open(regression.c_str(), std::fstream::in);
  // Simple line by line comparison
  while (!outstr.eof() && !regstr.eof()) {
    std::getline(outstr, outln);
    std::getline(regstr, regln);
    if (outln != regln) {
      std::cerr << "MISMATCH LINE NO:" << lno << " [" << outln
                << "] REF:[" << regln << std::endl;
      passed = false;
    }
    lno++;
  }
  if (outstr.eof() != regstr.eof()) passed = false;
  outstr.close();
  regstr.close();
  return passed;
}
