#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"

namespace kaldi {
//Randomly select two groups (uttid, startpoint) from noise list, respectively.
//The length of selected utterance is bigger than chunk_size, which guarantees
//we can get a complete noise chunk. At the same time, the startpoint is randomly
//selected from [0, len(utt)-chunk_size].
void RandomSelectTwoNoiseUtt(const std::vector<std::pair<std::string, float>>& utt2dur_list,
                             const int32& utt2dur_len,
                             const int32& chunk_size,
                             std::vector<std::pair<std::string, float>>* output) {
  for(int32 index = 0; index < 2; ++index) {
    int32 r_index = -1;
    do {
      // r_index indicate the random index of utt2dur_list
      r_index = RandInt(0, utt2dur_len-1);
    } while (utt2dur_list[r_index].second > chunk_size);
    // random number in [0, utt2dur]
    float start_point = RandInt(0, (int)(utt2dur_list[r_index].second)*100) * 1.0 / 100;
    output->push_back(std::make_pair(utt2dur_list[r_index].first, start_point));
  }
  KALDI_ASSERT(output->size() == 2);
}


} //The end of namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Get the data chunks. We sequentially read the wav files. And cut them\n"
        "into 'chunk_size' length fragment. And we randomly select two 'chunk_size'\n"
        "length fragments from noise-list. Then we the store the 'source' chunk and\n"
        "'noise' chunks into the corresponding matrix separately."
        "Usage:  fvector-chunk [options...] <wav-rspecifier> <noise-rspecifier>"
        "<utt2dur-rxfilename> <feats-wspecifier> <noise1-wspecifier> <noise2-wspecifier>\n";

    // construct all the global objects
    ParseOptions po(usage);
    int32 chunk_size = 120;
    int32 channel = -1;
    int32 shift_time = 60;
    BaseFloat min_duration = 0.0;
    int32 srand_seed = 1;
    int32 block_size = 32;
    BaseFloat samp_freq = 8000;

    po.Register("channel", &channel, "Channel to extract (-1 -> expect mono, "
                "0 -> left, 1 -> right)");
    po.Register("chunk-size", &chunk_size, "The expected length of the chunk.");
    po.Register("shift-time", &shift_time, "Time shift, which decide the overlap "
                "of two adjacent chunks in the same utterance.");
    po.Register("min-duration", &min_duration, "Minimum duration of segments "
                "to process (in seconds).");
    po.Register("srand", &srand_seed, "Seed for random number generator.");
    po.Register("block-size",&block_size, "Specify the number of lines of feature "
                "block; the number of lines of noise block will be twice.");
    po.Register("sample-frequency", &samp_freq, "Specify the sample frequency. "
                "(default=8000)");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    srand(srand_seed);

    std::string wav_rspecifier = po.GetArg(1);
    std::string noise_rspecifier = po.GetArg(2);
    std::string utt2dur_rxfilename = po.GetArg(3);
    std::string output_feature_wspecifier = po.GetArg(4);
    std::string output_noise1_wspecifier = po.GetArg(5);
    std::string output_noise2_wspecifier = po.GetArg(6);


    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    RandomAccessTableReader<WaveHolder> noise_reader(noise_rspecifier);
    Input ki(utt2dur_rxfilename);
    BaseFloatMatrixWriter feature_writer;  // typedef to TableWriter<something>.
    BaseFloatMatrixWriter noise1_writer;
    BaseFloatMatrixWriter noise2_writer;

    //Read the utt2dur file
    //the vector--utt2dur is used to randomly select the noise chunk.
    std::vector<std::pair<std::string, float>> utt2dur;
    std::string line;
    while (std::getline(ki.Stream(), line)) {
      std::vector<std::string> split_line;
      // Split the line by space or tab and check the number of fields in each
      // line. There must be 2 fields--segment utt_id and duration
      SplitStringToVector(line, " \t\r", true, &split_line);
      if (split_line.size() != 2) {
        KALDI_WARN << "Invalid line in segments file: " << line;
        continue;
      }
      std::string utt = split_line[0],
        duration_str = split_line[1];

      double duration;
      if (!ConvertStringToReal(duration_str, &duration)) {
        KALDI_WARN << "Invalid line in utt2dur file: " << line;
        continue;
      }
      utt2dur.push_back(std::make_pair(utt, duration));
    }
    //random number in [0, utt2dur_len), so we get variable "utt2dur_len"
    int32 utt2dur_len = utt2dur.size();

    // Start to chunk the data, each source chunk and 2 corresponding noise 
    // chunks were store into corresping block matrix. When counter == block_size,
    // write one source block and two noise blocks.
    int32 num_utts = 0, num_success = 0;
    int32 counter = 0;
    int32 dim = static_cast<int32>(samp_freq * chunk_size / 1000);
    Matrix<BaseFloat> feature_block(block_size, dim),
                      noise_block1(block_size, dim),
                      noise_block2(block_size, dim);

    for (; !reader.Done(); reader.Next()) {
      num_utts++;
      std::string utt = reader.Key();
      const WaveData &wave_data = reader.Value();
      if (wave_data.Duration() < min_duration) {
        KALDI_WARN << "File: " << utt << " is too short ("
                   << wave_data.Duration() << " sec): producing no output.";
        continue;
      }
      int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
      {  // This block works out the channel (0=left, 1=right...)
        KALDI_ASSERT(num_chan > 0);  // should have been caught in
        // reading code if no channels.
        if (channel == -1) {
          this_chan = 0;
          if (num_chan != 1)
            KALDI_WARN << "Channel not specified but you have data with "
                       << num_chan  << " channels; defaulting to zero";
        } else {
          if (this_chan >= num_chan) {
            KALDI_WARN << "File with id " << utt << " has "
                       << num_chan << " channels but you specified channel "
                       << channel << ", producing no output.";
            continue;
          }
        }
      }

      KALDI_ASSERT(wave_data.SampFreq() == samp_freq);
      SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
      //e.g. A "waveform" is 285ms, chunk_size is 120ms, shift_time is 70ms. At last, the chunks
      //will be 0-120ms, 70-190ms, 140-260ms. So num_chunk = 3
      int32 num_chunk = (int)((waveform.Dim() / wave_data.SampFreq() - chunk_size ) / shift_time) + 1;
      try {
        for (int32 index = 0; index < num_chunk; ++index) {
          int32 source_start = wave_data.SampFreq() * (index * shift_time);
          feature_block.CopyRowFromVec(SubVector<BaseFloat>(waveform, source_start, dim), counter);
          //1. Generate 2 random number form [0, utt2dur_len)
          //2. From vector utt2dur, get the 2 pairs
          //3. Generate 2 random "start point" number from [0, utt2dur[x][1])
          //The three steps is implemented by function--"RandomSelectTwoNoiseUtt"
          //The output vector, "two_random_uttid", contains two pairs. For each
          //pair, its content is <uttid, start_point>
          std::vector<std::pair<std::string, float>> two_random_uttid;
          RandomSelectTwoNoiseUtt(utt2dur, utt2dur_len, chunk_size/1000, 
                                  &two_random_uttid);
          //4. According to the utt2dur[x][0]--utt_id and startpoint form RandomAccessTable
          //   read noise chunk.
          //5. The features matrix has 3 lines: source, nosie1, noise2.
          const WaveData &noise_wav1 = noise_reader.Value(two_random_uttid[0].first);
          KALDI_ASSERT(wave_data.SampFreq() == noise_wav1.SampFreq());
          SubVector<BaseFloat> noise1(noise_wav1.Data(), 0);
          noise_block1.CopyRowFromVec(SubVector<BaseFloat>(noise1, two_random_uttid[0].second, dim), counter);
          
          const WaveData &noise_wav2 = noise_reader.Value(two_random_uttid[1].first);
          KALDI_ASSERT(wave_data.SampFreq() == noise_wav2.SampFreq());
          SubVector<BaseFloat> noise2(noise_wav2.Data(), 0);
          noise_block2.CopyRowFromVec(SubVector<BaseFloat>(noise2, two_random_uttid[1].second, dim), counter);
          counter++;
          
          // when "counter == block_size", store the matrices.
          if (counter == block_size) {
            std::ostringstream utt_id_new;
            utt_id_new << utt << '_' << index;
            feature_writer.Write(utt_id_new.str(), feature_block);
            noise1_writer.Write(utt_id_new.str(), noise_block1);
            noise2_writer.Write(utt_id_new.str(), noise_block2);
            counter = 0;
          }
        }
      } catch (...) {
        KALDI_WARN << "Failed to compute features for utterance "
                   << utt;
        continue;
      }
      
      if (num_utts % 10 == 0)
        KALDI_LOG << "Processed " << num_utts << " utterances";
      KALDI_VLOG(2) << "Processed features for key " << utt;
      num_success++;
    }
    KALDI_LOG << " Done " << num_success << " out of " << num_utts
              << " utterances.";
    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
