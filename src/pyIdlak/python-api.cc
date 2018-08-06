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

struct PyPugiXMLDocument {
  pugi::xml_document * doc_;
};

struct PyIdlakBuffer {
  char * data_;
  int len_;
};

struct PyIdlakModule {
  enum IDLAKMOD modtype_;
  void * modptr_;
};


PyIdlakBuffer * PyIdlakBuffer_newfromstr(const char * data) {
  PyIdlakBuffer * pybuf;
  if (data) {
    pybuf = new PyIdlakBuffer;
    pybuf->len_ = strlen(data);
    pybuf->data_ = new char[pybuf->len_];
    strcpy(pybuf->data_, data);
  }
  return pybuf;
}

void PyIdlakBuffer_delete(PyIdlakBuffer * pybuf) {
  if (pybuf) {
    delete pybuf->data_;
    delete pybuf;
  }
}

const char * PyIdlakBuffer_get(PyIdlakBuffer * pybuf) {
  if (pybuf) return pybuf->data_;
  return NULL;
}
    
PyTxpParseOptions * PyTxpParseOptions_new(const char *usage) {
  PyTxpParseOptions * pypo = new PyTxpParseOptions;
  pypo->po_ = new kaldi::TxpParseOptions(usage);
  return pypo;
}


void PyTxpParseOptions_delete(PyTxpParseOptions * pypo) {
  if (pypo) {
    delete pypo->po_;
    delete pypo;
  }
}

void PyTxpParseOptions_PrintUsage(PyTxpParseOptions * pypo, bool print_command_line) {
  if (pypo) {
    pypo->po_->PrintUsage(print_command_line);
  }
}

const char * PyTxpParseOptions_GetArg(PyTxpParseOptions * pypo, int n) {
  if (pypo) {
    return pypo->po_->GetArg(n).c_str();
  }
  return NULL;
}

int PyTxpParseOptions_Read(PyTxpParseOptions * pypo, int argc, char *argv[]) {
  return pypo->po_->Read(argc, argv);
}

int PyTxpParseOptions_NumArgs(PyTxpParseOptions * pypo) {
  return  pypo->po_->NumArgs();
}

PyPugiXMLDocument * PyPugiXMLDocument_new() {
  PyPugiXMLDocument * pypugidoc = new PyPugiXMLDocument;
  pypugidoc->doc_ =  new pugi::xml_document;
  return pypugidoc;
}

void PyPugiXMLDocument_delete(PyPugiXMLDocument * pypugidoc) {
  if (pypugidoc) {
    delete pypugidoc->doc_;
    delete pypugidoc;
  }
}

void PyPugiXMLDocument_LoadString(PyPugiXMLDocument * pypugidoc, const char * data) {
  if (pypugidoc) {
    pypugidoc->doc_->load(data, pugi::encoding_utf8);
  }
}

PyIdlakBuffer * PyPugiXMLDocument_SavePretty(PyPugiXMLDocument * pypugidoc) {
  PyIdlakBuffer * pybuf;
  std::ostringstream stream;
  std::string output;
  const char * s;
  if (pypugidoc) {
    pypugidoc->doc_->save(stream, "\t");
    output = stream.str();
    s = output.c_str();
    pybuf = new PyIdlakBuffer;
    pybuf->len_ = strlen(s);
    pybuf->data_ = new char[pybuf->len_ + 1];
    strcpy(pybuf->data_, s);
  }
  return pybuf;
}

/* Modules all have Init, Process and Delete */

PyIdlakModule * PyIdlakModule_new(enum IDLAKMOD modtype, PyTxpParseOptions * pypo) {
  kaldi::TxpTokenise * t;
  kaldi::TxpPosTag * p;
  kaldi::TxpPauses * pz;
  kaldi::TxpPhrasing * ph;
  kaldi::TxpPronounce * pr;
  kaldi::TxpSyllabify * sy;
  kaldi::TxpCex * cx;
  
  if (!pypo) return NULL;
  PyIdlakModule * pymod = new PyIdlakModule;
  pymod->modtype_ = modtype;
  switch(modtype) {
    case Tokenise:
      t = new kaldi::TxpTokenise;
      t->Init(*(pypo->po_));
      pymod->modptr_ = (void *) t;
      break;
    case PosTag:
      p = new kaldi::TxpPosTag;
      p->Init(*(pypo->po_));
      pymod->modptr_ = (void *) p;
      break;
    case Pauses:
      pz = new kaldi::TxpPauses;
      pz->Init(*(pypo->po_));
      pymod->modptr_ = (void *) pz;
      break;
    case Phrasing:
      ph = new kaldi::TxpPhrasing;
      ph->Init(*(pypo->po_));
      pymod->modptr_ = (void *) ph;
      break;
    case Pronounce:
      pr = new kaldi::TxpPronounce;
      pr->Init(*(pypo->po_));
      pymod->modptr_ = (void *) pr;
      break;
    case Syllabify:
      sy = new kaldi::TxpSyllabify;
      sy->Init(*(pypo->po_));
      pymod->modptr_ = (void *) sy;
      break;
    case ContextExtraction:
      cx = new kaldi::TxpCex;
      cx->Init(*(pypo->po_));
      pymod->modptr_ = (void *) cx;
      break;
    case Empty:
    default:
      break;
  }
  return pymod;
}

void PyIdlakModule_delete(PyIdlakModule * pymod) {
  if (!pymod) return;
  switch(pymod->modtype_) {
    case Tokenise:
      delete static_cast<kaldi::TxpTokenise *>(pymod->modptr_);
      break;
    case PosTag:
      delete static_cast<kaldi::TxpPosTag *>(pymod->modptr_);
      break;
    case Pauses:
      delete static_cast<kaldi::TxpPauses *>(pymod->modptr_);
      break;
    case Phrasing:
      delete static_cast<kaldi::TxpPhrasing *>(pymod->modptr_);
      break;
    case Pronounce:
      delete static_cast<kaldi::TxpPronounce *>(pymod->modptr_);
      break;
    case Syllabify:
      delete static_cast<kaldi::TxpSyllabify *>(pymod->modptr_);
      break;
    case ContextExtraction:
      delete static_cast<kaldi::TxpCex *>(pymod->modptr_);
      break;
    case Empty:
    default:
      break;
  }
  delete pymod;
}

void PyIdlakModule_process(PyIdlakModule * pymod, PyPugiXMLDocument * pypugidoc) {
  if (!pymod || !pypugidoc) return;
  switch(pymod->modtype_) {
    case Tokenise:
      static_cast<kaldi::TxpTokenise *>(pymod->modptr_)->Process(pypugidoc->doc_);
      break;
    case PosTag:
      static_cast<kaldi::TxpPosTag *>(pymod->modptr_)->Process(pypugidoc->doc_);
      break;
    case Pauses:
      static_cast<kaldi::TxpPauses *>(pymod->modptr_)->Process(pypugidoc->doc_);
      break;
    case Phrasing:
      static_cast<kaldi::TxpPhrasing *>(pymod->modptr_)->Process(pypugidoc->doc_);
      break;
    case Pronounce:
      static_cast<kaldi::TxpPronounce *>(pymod->modptr_)->Process(pypugidoc->doc_);
      break;
    case Syllabify:
      static_cast<kaldi::TxpSyllabify *>(pymod->modptr_)->Process(pypugidoc->doc_);
      break;
    case ContextExtraction:
      static_cast<kaldi::TxpCex *>(pymod->modptr_)->Process(pypugidoc->doc_);
      break;
    case Empty:
    default:
      break;
  }
}
 
