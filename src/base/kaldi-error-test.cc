// base/kaldi-error-test.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "base/kaldi-common.h"

namespace kaldi {
namespace {

using namespace ::testing;

TEST(DefaultLoggingTest, Exceptions) {
  EXPECT_NO_THROW(KALDI_LOG << "Test");
  EXPECT_NO_THROW(KALDI_WARN << "Test");
  EXPECT_ANY_THROW(KALDI_ERR << "Test");
  EXPECT_ANY_THROW(KALDI_ASSERT(4 + 2 == 42));
}

TEST(DefaultLoggingTest, WarninigStderrPrefix) {
  testing::internal::CaptureStderr();
  KALDI_WARN << "Twas brillig";
  std::string stderr_log = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_log, StartsWith("WARNING "));
  EXPECT_THAT(stderr_log, HasSubstr(" Twas brillig"));
}

TEST(DefaultLoggingTest, ErrorStderrPrefix) {
  testing::internal::CaptureStderr();
  try {
    KALDI_ERR << "Twas brillig";
  } catch (...) {
  }
  std::string stderr_log = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_log, StartsWith("ERROR "));
  EXPECT_THAT(stderr_log, HasSubstr(" Twas brillig"));
}

TEST(DefaultLoggingTest, AssertStderrMessage) {
  testing::internal::CaptureStderr();
  try {
    KALDI_ASSERT(4 + 2 == 42);
  } catch (...) {
  }
  std::string stderr_log = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_log, HasSubstr("4 + 2 == 42"));
}

// Keep this test near the end as it messes up with the compiler line number.
TEST(DefaultLoggingTest, LogStderrPrefixFileLine) {
  testing::internal::CaptureStderr();
#line 1234 "foo.cc"
  KALDI_LOG << "Twas brillig";
#line 10000 "kaldi-error-test.cc"
  std::string stderr_log = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_log, StartsWith("LOG "));
  EXPECT_THAT(stderr_log, HasSubstr(":foo.cc:1234)"));
  EXPECT_THAT(stderr_log, HasSubstr(" Twas brillig"));
}

}  // namespace
}  // namespace kaldi
