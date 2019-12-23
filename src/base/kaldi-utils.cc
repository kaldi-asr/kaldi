// base/kaldi-utils.cc
// Copyright 2009-2011   Karel Vesely;  Yanmin Qian;  Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifdef _WIN32_WINNT_WIN8
#include <Synchapi.h>
#elif defined(_WIN32) || defined(_MSC_VER) || defined(MINGW)
#include <Windows.h>
#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf _snprintf
#endif /* _MSC_VER < 1900 */
#else
#include <unistd.h>
#endif

#include <string>
#include "base/kaldi-common.h"


namespace kaldi {

std::string CharToString(const char &c) {
  char buf[20];
  if (std::isprint(c))
    snprintf(buf, sizeof(buf), "\'%c\'", c);
  else
    snprintf(buf, sizeof(buf), "[character %d]", static_cast<int>(c));
  return (std::string) buf;
}

void Sleep(float seconds) {
#if defined(_MSC_VER) || defined(MINGW)
  ::Sleep(static_cast<int>(seconds * 1000.0));
#elif defined(__CYGWIN__)
  sleep(static_cast<int>(seconds));
#else
  usleep(static_cast<int>(seconds * 1000000.0));
#endif
}

#if HAVE_CUDA == 1
#ifdef USE_NVTX
NvtxTracer::NvtxTracer(const char* name) {
  const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
  const int num_colors = sizeof(colors)/sizeof(uint32_t);
  int color_id = ((int)name[0])%num_colors;
	nvtxEventAttributes_t eventAttrib = {0};
	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.colorType = NVTX_COLOR_ARGB;
	eventAttrib.color = colors[color_id];
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
	eventAttrib.message.ascii = name;
	nvtxRangePushEx(&eventAttrib);
  // nvtxRangePushA(name);
}
NvtxTracer::~NvtxTracer() {
  nvtxRangePop();
}
#endif
#endif

}  // end namespace kaldi
