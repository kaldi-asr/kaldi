// pyIdlak/python-api.cc

// Copyright 2018 CereProc Ltd.  (Authors: David A. Braude
//                                         Matthew Aylett)

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

#include "idlaktxp/idlaktxp.h"

#include "python-api.h"

struct PyTxpParseOptions {
  kaldi::TxpParseOptions * po_;
};


PyTxpParseOptions * PyTxpParseOptions_new(const char *usage)
{
  PyTxpParseOptions * pypo = new PyTxpParseOptions;
  pypo->po_ = new kaldi::TxpParseOptions(usage);
  return pypo;
}


void PyTxpParseOptions_delete(PyTxpParseOptions * pypo)
{
  if (pypo) {
    delete pypo->po_;
    delete pypo;
  }
}

void PyTxpParseOptions_PrintUsage(PyTxpParseOptions * pypo, bool print_command_line)
{
  if (pypo) {
    pypo->po_->PrintUsage(print_command_line);
  }
}
