// online2bin/online2-wav-nnet3-latgen-faster-i2x.cc

// Copyright 2017 i2x GmbH (author: Christoph Feinauer)

#include <algorithm>
#include <random>
#include <feat/wave-reader.h>
#include "util/common-utils.h"
#include "online2/online2-nnet3-latgen-i2x-wrapper.h"

using namespace kaldi;

int main(int argc, char *argv[]) {
  const char *usage =
      "Reads in wav file(s) and simulates online decoding with neural nets\n"
          "(nnet3 setup, i2x wrapper).\n"
          "\n"
          "Usage: online2-wav-nnet3-latgen-faster-i2x [options] <path-to-resource-dir-in> <wavefile-scp-in>\n";

  ParseOptions po(usage);
  po.Read(argc, argv);

  if (po.NumArgs() != 2) {
    po.PrintUsage();
    return 1;
  }

  std::string resource_dir = po.GetArg(1),
      wav_rspecifier = po.GetArg(2);

  DecoderFactory decoder_factory(resource_dir);

  const size_t kBlockSize_MAX = 200;

  KALDI_ASSERT(wav_rspecifier.size() > 4);
  wav_rspecifier.erase(0, 4);
  std::vector<std::pair<std::string, std::string> > file_list;
  ReadScriptFile(wav_rspecifier, true, &file_list);

  for (auto pair: file_list) {

    const std::string &utt = pair.first;

    std::ifstream is(pair.second, std::ifstream::binary);

    WaveInfo header;
    header.Read(is);

    Decoder *decoder = decoder_factory.StartDecodingSession();
    KALDI_ASSERT(decoder != nullptr);

    while (is) {
      const size_t kBlockSize = rand()%(kBlockSize_MAX-1)+1;
      std::string buffer;
      buffer.resize(kBlockSize);
      is.read(&buffer[0], kBlockSize);
      if (is.gcount() < kBlockSize) {
        buffer.resize(is.gcount());
      }
      if (buffer.empty()) {
        break;
      }
      int32 return_code = decoder->FeedBytestring(buffer);
      KALDI_ASSERT(return_code == 0);
    }

    int32_t return_code = decoder->FeedBytestring(std::string(""));
    KALDI_ASSERT(return_code == 0);
    RecognitionResult result;
    return_code = decoder->GetResultAndFinalize(&result);
    KALDI_LOG << utt << ": " << result.transcript;
    KALDI_ASSERT(return_code == 0);
    delete decoder;
  }
}