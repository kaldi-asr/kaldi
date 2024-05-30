// feat/wave-reader-test.cc

// Copyright 2017  Smart Action LLC (kkm)

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

#include <iostream>

#include "base/kaldi-math.h"
#include "feat/wave-reader.h"
#include "matrix/kaldi-matrix.h"

using namespace kaldi;

// Ugly macros to package bytes in wave file order (low-endian).
#define BY(n,k) ((char)((uint32)(n) >> (8 * (k)) & 0xFF))
#define WRD(n) BY(n,0), BY(n,1)
#define DWRD(n) BY(n,0), BY(n,1), BY(n,2), BY(n,3)

static void UnitTestStereo8K() {
  /* Reference file written with Adobe Audition (random data):
00000000  52 49 46 46 32 00 00 00  57 41 56 45 66 6d 74 20  |RIFF2...WAVEfmt |
00000010  12 00 00 00 01 00 02 00  40 1f 00 00 00 7d 00 00  |........@....}..|
00000020  04 00 10 00 00 00 64 61  74 61 0c 00 00 00 00 00  |......data......|
00000030  31 51 ff 21 f4 63 38 4c  26 60                    |1Q.!.c8L&`|
  */

  const int hz = 8000;
  const int byps = hz * 2 /* channels */ * 2 /* bytes/sample */;
  const char file_data[] = {
    'R', 'I', 'F', 'F',
    DWRD(50),   // File length after this point.
    'W', 'A', 'V', 'E',
    'f', 'm', 't', ' ',
    DWRD(18),   // sizeof(struct WAVEFORMATEX)
    WRD(1),     // WORD  wFormatTag;
    WRD(2),     // WORD  nChannels;
    DWRD(hz),   // DWORD nSamplesPerSec; 40 1f 00 00
    DWRD(byps), // DWORD nAvgBytesPerSec; 00 7d 00 00
    WRD(4),     // WORD  nBlockAlign;
    WRD(16),    // WORD  wBitsPerSample;
    WRD(0),     // WORD  cbSize;
    'd', 'a', 't', 'a',
    DWRD(12),   // 'data' chunk length.
    WRD(0), WRD(-1),
    WRD(-32768), WRD(0),
    WRD(32767), WRD(1)
  };

  const char expect_mat[] = "[ 0 -32768 32767 \n -1 0 1 ]";

  // Read binary file data.
  std::istringstream iws(std::string(file_data, sizeof file_data),
                         std::ios::in | std::ios::binary);
  WaveData wave;
  wave.Read(iws);

  // Read expected wave data.
  std::istringstream ies(expect_mat, std::ios::in);
  Matrix<BaseFloat> expected;
  expected.Read(ies, false /* text */);

  AssertEqual(wave.SampFreq(), hz, 0);
  AssertEqual(wave.Duration(), 3.0 /* samples */ / hz /* Hz */, 1E-6);
  AssertEqual(wave.Data(), expected);
}

static void UnitTestMono22K() {
  /* Reference file written with Adobe Audition (random data):
00000000  52 49 46 46 30 00 00 00  57 41 56 45 66 6d 74 20  |RIFF0...WAVEfmt |
00000010  12 00 00 00 01 00 01 00  22 56 00 00 44 ac 00 00  |........"V..D...|
00000020  02 00 10 00 00 00 64 61  74 61 0a 00 00 00 25 36  |......data....%6|
00000030  cb 41 1b 4d 04 4e 62 3d                           |.A.M.Nb=|
  */

  const int hz = 22050;
  const int byps = hz * 1 /* channels */ * 2 /* bytes/sample */;
  const char file_data[] = {
    'R', 'I', 'F', 'F',
    DWRD(48),   // File length after this point.
    'W', 'A', 'V', 'E',
    'f', 'm', 't', ' ',
    DWRD(18),   // sizeof(struct WAVEFORMATEX)
    WRD(1),     // WORD  wFormatTag;
    WRD(1),     // WORD  nChannels;
    DWRD(hz),   // DWORD nSamplesPerSec;
    DWRD(byps), // DWORD nAvgBytesPerSec;
    WRD(2),     // WORD  nBlockAlign;
    WRD(16),    // WORD  wBitsPerSample;
    WRD(0),     // WORD  cbSize;
    'd', 'a', 't', 'a',
    DWRD(10),   // 'data' chunk length.
    WRD(0), WRD(-1), WRD(-32768), WRD(32767), WRD(1)
  };

  const char expect_mat[] = "[ 0 -1 -32768 32767 1 ]";

  // Read binary file data.
  std::istringstream iws(std::string(file_data, sizeof file_data),
                         std::ios::in | std::ios::binary);
  WaveData wave;
  wave.Read(iws);

  // Read expected matrix.
  std::istringstream ies(expect_mat, std::ios::in);
  Matrix<BaseFloat> expected;
  expected.Read(ies, false /* text */);

  AssertEqual(wave.SampFreq(), hz, 0);
  AssertEqual(wave.Duration(), 5.0 /* samples */ / hz /* Hz */, 1E-6);
  AssertEqual(wave.Data(), expected);
}

static void UnitTestEndless1() {
  const int hz = 8000;
  const int byps = hz * 1 /* channels */ * 2 /* bytes/sample */;
  const char file_data[] = {
    'R', 'I', 'F', 'F',
    DWRD(0),    // File length unknown
    'W', 'A', 'V', 'E',
    'f', 'm', 't', ' ',
    DWRD(18),   // sizeof(struct WAVEFORMATEX)
    WRD(1),     // WORD  wFormatTag;
    WRD(1),     // WORD  nChannels;
    DWRD(hz),   // DWORD nSamplesPerSec;
    DWRD(byps), // DWORD nAvgBytesPerSec;
    WRD(2),     // WORD  nBlockAlign;
    WRD(16),    // WORD  wBitsPerSample;
    WRD(0),     // WORD  cbSize;
    'd', 'a', 't', 'a',
    DWRD(0),    // 'data' chunk length unknown.
    WRD(1), WRD(2), WRD(3)
  };

  const char expect_mat[] = "[ 1 2 3 ]";

  // Read binary file data.
  std::istringstream iws(std::string(file_data, sizeof file_data),
                         std::ios::in | std::ios::binary);
  WaveData wave;
  wave.Read(iws);

  // Read expected matrix.
  std::istringstream ies(expect_mat, std::ios::in);
  Matrix<BaseFloat> expected;
  expected.Read(ies, false /* text */);

  AssertEqual(wave.Data(), expected);
}

static void UnitTestEndless2() {
  const int hz = 8000;
  const int byps = hz * 1 /* channels */ * 2 /* bytes/sample */;
  const char file_data[] = {
    'R', 'I', 'F', 'F',
    DWRD(-1),   // File length unknown
    'W', 'A', 'V', 'E',
    'f', 'm', 't', ' ',
    DWRD(18),   // sizeof(struct WAVEFORMATEX)
    WRD(1),     // WORD  wFormatTag;
    WRD(1),     // WORD  nChannels;
    DWRD(hz),   // DWORD nSamplesPerSec;
    DWRD(byps), // DWORD nAvgBytesPerSec;
    WRD(2),     // WORD  nBlockAlign;
    WRD(16),    // WORD  wBitsPerSample;
    WRD(0),     // WORD  cbSize;
    'd', 'a', 't', 'a',
    DWRD(-1),   // 'data' chunk length unknown.
    WRD(1), WRD(2), WRD(3)
  };

  const char expect_mat[] = "[ 1 2 3 ]";

  // Read binary file data.
  std::istringstream iws(std::string(file_data, sizeof file_data),
                         std::ios::in | std::ios::binary);
  WaveData wave;
  wave.Read(iws);

  // Read expected matrix.
  std::istringstream ies(expect_mat, std::ios::in);
  Matrix<BaseFloat> expected;
  expected.Read(ies, false /* text */);

  AssertEqual(wave.Data(), expected);
}

static void UnitTest() {
  UnitTestStereo8K();
  UnitTestMono22K();
  UnitTestEndless1();
  UnitTestEndless2();
}

int main() {
  try {
    UnitTest();
    std::cout << "LGTM\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return 1;
  }
}
