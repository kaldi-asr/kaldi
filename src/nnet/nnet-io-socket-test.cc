// nnet/nnet-io-socket-test.cc

// Copyright 2015   Brno University of Technology (Author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
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

#include "matrix/kaldi-vector.h"
#include "nnet/nnet-io-socket.h"

using kaldi::int32;
using kaldi::BaseFloat;

namespace kaldi {

using namespace nnet1;

static void UnitTestSendRecvRandomVector() {
  // connect,
  Socket socket("localhost", 12345);
  // create random vector,
  Vector<BaseFloat> v(1e7); // 10m elements, ~40MB,
  for (int32 i = 0; i < v.Dim(); i++) {
    v(i) = RandGauss(); // Random,
  }
  // send,
  v.Write(socket.SendStream(), true); // binary,
  socket.Send();
  // recieve,
  socket.Recv();
  // read to vector,
  Vector<BaseFloat> v2;
  v2.Read(socket.RecvStream(), true); // binary,
  // must be the same!
  AssertEqual(v, v2, 0.0);
}

} // namespace

int main() {
  kaldi::g_kaldi_verbose_level = 1;
  kaldi::UnitTestSendRecvRandomVector();
  std::cout << "Test OK.\n";
  return 0;
}

