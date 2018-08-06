// pyIdlak/python-api.h

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

#ifndef KALDI_PYIDLAK_PYTHON_API_H_
#define KALDI_PYIDLAK_PYTHON_API_H_

// TxpParseOptions wrappers
typedef struct PyTxpParseOptions PyTxpParseOptions;
typedef struct PyPugiXMLDocument PyPugiXMLDocument;
typedef struct PyIdlakBuffer PyIdlakBuffer;
typedef struct PyIdlakModule PyIdlakModule;

enum IDLAKMOD {Empty = 0,
               Tokenise = 1,
               PosTag = 2,
               Pauses = 3,
               Phrasing = 4,
               Pronounce = 5,
               Syllabify = 6,
               ContextExtraction = 7,
               NumMods = 7};

PyIdlakBuffer * PyIdlakBuffer_newfromstr(const char * data);
void PyIdlakBuffer_delete(PyIdlakBuffer * pybuf);
const char * PyIdlakBuffer_get(PyIdlakBuffer * pybuf);

PyTxpParseOptions * PyTxpParseOptions_new(const char *usage);
void PyTxpParseOptions_delete(PyTxpParseOptions * pypo);

void PyTxpParseOptions_PrintUsage(PyTxpParseOptions * pypo, bool print_command_line = false);
int PyTxpParseOptions_Read(PyTxpParseOptions * pypo, int argc, char * argv[]);
int PyTxpParseOptions_NumArgs(PyTxpParseOptions * pypo);
const char * PyTxpParseOptions_GetArg(PyTxpParseOptions * pypo, int n);

PyPugiXMLDocument * PyPugiXMLDocument_new();
void PyPugiXMLDocument_delete(PyPugiXMLDocument * pypugidoc);
void PyPugiXMLDocument_LoadString(PyPugiXMLDocument * pypugidoc, const char * data);
PyIdlakBuffer * PyPugiXMLDocument_SavePretty(PyPugiXMLDocument * pypugidoc);

PyIdlakModule * PyIdlakModule_new(enum IDLAKMOD modtype, PyTxpParseOptions * pypo);
void PyIdlakModule_delete(PyIdlakModule * pymod);
void PyIdlakModule_process(PyIdlakModule * pymod, PyPugiXMLDocument * pypugidoc);

#endif // KALDI_PYIDLAK_PYTHON_API_H_
