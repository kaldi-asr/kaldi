// online2bin/online2-wav-nnet3-latgen-faster-i2x.cc

// Copyright 2017 i2x GmbH (author: Christoph Feinauer)

#include <algorithm>
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

  SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
  static constexpr int32 chunk_length = 200;
  int16 buf[chunk_length];

  while (!wav_reader.Done()) {
    Decoder *decoder = decoder_factory.StartDecodingSession();
    KALDI_ASSERT(decoder != nullptr);

    const std::string &utt = wav_reader.Key();
    const WaveData &wave_data = wav_reader.Value();
    SubVector<BaseFloat> data(wave_data.Data(), 0);

    int32 return_code = 0;
    int32 i = 0;
    while (i < data.Dim()) {
      int32_t length = std::min(
          static_cast<int32>(data.Dim()) - i,
          chunk_length);
      for (size_t t = 0; t < length; t++) {
        buf[t] = static_cast<int16>(data(i + t));
      }
      return_code = decoder->FeedChunk(buf, length);
      KALDI_ASSERT(return_code == 0);
      i += chunk_length;
    }
    return_code = decoder->FeedChunk(nullptr, 0);
    KALDI_ASSERT(return_code == 0);
    RecognitionResult result;
    return_code = decoder->GetResultAndFinalize(&result);
    KALDI_LOG << utt << ": " << result.transcript;
    KALDI_ASSERT(return_code == 0);
    delete decoder;
    wav_reader.Next();
  }
}