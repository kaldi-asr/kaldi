// onlinebin/online-audio-client.cc

// Copyright 2012 Cisco Systems (author: Matthias Paulik)
// Copyright 2013 Polish-Japanese Institute of Information Technology (author: Danijel Korzinek)

//   Modifications to the original contribution by Cisco Systems made by:
//   Vassil Panayotov

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
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

#include "util/parse-options.h"
#include "util/kaldi-table.h"
#include "feat/wave-reader.h"
#include "online/online-audio-source.h"

namespace kaldi {

bool WriteFull(int32 desc, char* data, int32 size);
bool ReadLine(int32 desc, std::string* str);
std::string TimeToTimecode(float time);

struct RecognizedWord {
  std::string word;
  float start, end;
};

}  //namespace kaldi

int main(int argc, char** argv) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {

    const char *usage =
        "Sends an audio file to the KALDI audio server (onlinebin/online-audio-server-decode-faster)\n"
            "and prints the result optionally saving it to an HTK label file or WebVTT subtitles file\n\n"
            "e.g.: ./online-audio-client 192.168.50.12 9012 'scp:wav_files.scp'\n\n";
    ParseOptions po(usage);

    bool htk = false, vtt = false;
    int32 channel = -1;
    int32 packet_size = 1024;

    po.Register("htk", &htk, "Save the result to an HTK label file");
    po.Register("vtt", &vtt, "Save the result to a WebVTT subtitle file");
    po.Register(
        "channel", &channel,
        "Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right)");
    po.Register("packet-size", &packet_size, "Send this many bytes per packet");

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      return 1;
    }

    std::string server_addr_str = po.GetArg(1);
    std::string server_port_str = po.GetArg(2);
    int32 server_port = strtol(server_port_str.c_str(), 0, 10);
    std::string wav_rspecifier = po.GetArg(3);

    int32 client_desc = socket(AF_INET, SOCK_STREAM, 0);
    if (client_desc == -1) {
      std::cerr << "ERROR: couldn't create socket!" << std::endl;
      return -1;
    }

    struct hostent* hp;
    unsigned long addr;

    addr = inet_addr(server_addr_str.c_str());
    if (addr == INADDR_NONE) {
      hp = gethostbyname(server_addr_str.c_str());
      if (hp == NULL) {
        std::cerr << "ERROR: couldn't resolve host string: " << server_addr_str
                  << std::endl;
        close(client_desc);
        return -1;
      }

      addr = *((unsigned long*) hp->h_addr);
    }

    sockaddr_in server;
    server.sin_addr.s_addr = addr;
    server.sin_family = AF_INET;
    server.sin_port = htons(server_port);
    if (::connect(client_desc, (struct sockaddr*) &server, sizeof(server))) {
      std::cerr << "ERROR: couldn't connect to server!" << std::endl;
      close(client_desc);
      return -1;
    }

    KALDI_VLOG(2) << "Connected to KALDI server at host " << server_addr_str
        << " port " << server_port << std::endl;

    char* pack_buffer = new char[packet_size];

    SequentialTableReader < WaveHolder > reader(wav_rspecifier);
    for (; !reader.Done(); reader.Next()) {
      std::string wav_key = reader.Key();

      KALDI_VLOG(2) << "File: " << wav_key << std::endl;

      const WaveData &wav_data = reader.Value();

      if (wav_data.SampFreq() != 16000)
        KALDI_ERR << "Sampling rates other than 16kHz are not supported!";

      int32 num_chan = wav_data.Data().NumRows(), this_chan = channel;
      {   // This block works out the channel (0=left, 1=right...)
        KALDI_ASSERT(num_chan > 0);  // should have been caught in
        // reading code if no channels.
        if (channel == -1) {
          this_chan = 0;
          if (num_chan != 1)
            KALDI_WARN << "Channel not specified but you have data with "
                << num_chan << " channels; defaulting to zero";
        } else {
          if (this_chan >= num_chan) {
            KALDI_WARN << "File with id " << wav_key << " has " << num_chan
                << " channels but you specified channel " << channel
                << ", producing no output.";
            continue;
          }
        }
      }

      OnlineVectorSource au_src(wav_data.Data().Row(this_chan));
      Vector < BaseFloat > data(packet_size / 2);
      while (au_src.Read(&data)) {
        for (int32 i = 0; i < data.Dim(); i++) {
          short sample = (short) data(i);
          memcpy(&pack_buffer[i * 2], (char*) &sample, 2);
        }

        int32 size = data.Dim() * 2;
        WriteFull(client_desc, (char*) &size, 4);

        WriteFull(client_desc, pack_buffer, size);
      }

      //send last packet
      int32 size = 0;
      WriteFull(client_desc, (char*) &size, 4);

      std::string reco_output;
      std::vector<RecognizedWord> results;
      float total_input_dur = 0.0f, total_reco_dur = 0.0f;

      while (true) {
        std::string line;
        if (!ReadLine(client_desc, &line))
          KALDI_ERR << "Server disconnected!";

        if (line.substr(0, 7) != "RESULT:") {
          if (line.substr(0, 8) == "PARTIAL:") {
            std::cout << line.substr(8) << " " << std::flush;
            continue;
          }
          KALDI_ERR << "Header parse error: " << line;
        }

        std::cout << std::endl;

        if (line == "RESULT:DONE")
          break;

        int32 res_num = 0;
        float input_dur = 0;
        float reco_dur = 0;

        std::string tok, key, val;
        size_t beg = 7, end, eq;

        do {
          end = line.find_first_of(',', beg);
          tok = line.substr(beg, end - beg);
          beg = end + 1;
          eq = tok.find_first_of('=');
          if (eq == std::string::npos || eq >= tok.size() - 1) {
            KALDI_WARN << "Error parsing header token " << tok;
            continue;
          }

          key = tok.substr(0, eq);
          val = tok.substr(eq + 1);

          if (key == "NUM") {
            res_num = strtol(val.c_str(), 0, 10);
          } else if (key == "FORMAT") {
            if (val != "WSE") {
              KALDI_ERR << "Only WSE format supported by this program!";
            }
          } else if (key == "RECO-DUR") {
            reco_dur = strtof(val.c_str(), 0);
          } else if (key == "INPUT-DUR") {
            input_dur = strtof(val.c_str(), 0);
          } else {
            KALDI_WARN << "Unknown header key: " << key;
          }
        } while (end != std::string::npos);

        total_input_dur += input_dur;
        total_reco_dur += reco_dur;

        for (int32 i = 0; i < res_num; i++) {
          std::string line;
          if (!ReadLine(client_desc, &line))
            KALDI_ERR << "Server disconnected!";

          std::string word_str, start_str, end_str;

          end = line.find_first_of(',');
          word_str = line.substr(0, end);
          beg = end + 1;
          end = line.find_first_of(',', beg);
          start_str = line.substr(beg, end - beg);
          beg = end + 1;
          end = line.find_first_of(',', beg);
          end_str = line.substr(beg, end - beg);

          RecognizedWord word;
          word.word = word_str;
          word.start = strtof(start_str.c_str(), 0);
          word.end = strtof(end_str.c_str(), 0);

          results.push_back(word);

          reco_output += word_str + " ";
        }
      }

      {
        float speed = total_input_dur / total_reco_dur;
        KALDI_VLOG(2) << "Recognized (" << speed << "xRT): " << reco_output
            << std::endl;
      }

      if (htk) {
        std::string name = wav_key + ".lab";
        std::ofstream htk_file(name.c_str());
        for (size_t i = 0; i < results.size(); i++)
          htk_file << (int) (results[i].start * 10000000) << " "
              << (int) (results[i].end * 10000000) << " " << results[i].word << std::endl;
        htk_file.close();
      }

      if (vtt && !results.empty()) {
        std::vector<RecognizedWord> subtitles;
        RecognizedWord subtitle_cue;

        subtitle_cue.start = -1;
        subtitle_cue.end = -1;
        subtitle_cue.word = "";

        for (size_t i = 0; i < results.size(); i++) {
          if (subtitle_cue.end >= 0) {
            if (results[i].start - subtitle_cue.end > 3.0f
                || results[i].word.size() + subtitle_cue.word.size() > 64) {

              if (results[i].start - subtitle_cue.end < 0.1f)
                subtitle_cue.end = results[i].start - 0.1f;

              subtitles.push_back(subtitle_cue);
              subtitle_cue.start = -1;
              subtitle_cue.end = -1;
              subtitle_cue.word = "";

            }
          }

          if (subtitle_cue.start < 0)
            subtitle_cue.start = results[i].start;
          else
            subtitle_cue.word += " ";

          subtitle_cue.end = results[i].end + 1.0f;

          subtitle_cue.word += results[i].word;
        }

        subtitles.push_back(subtitle_cue);

        std::string name = wav_key + ".vtt";
        std::ofstream vtt_file(name.c_str());

        vtt_file << "WEBVTT FILE" << std::endl << std::endl;

        for (size_t i = 0; i < subtitles.size(); i++)
          vtt_file << (i + 1) << std::endl << TimeToTimecode(subtitles[i].start)
              << " --> " << TimeToTimecode(subtitles[i].end) << std::endl
              << subtitles[i].word << std::endl << std::endl;

        vtt_file.close();
      }
    }

    close(client_desc);
    delete[] pack_buffer;
  }

  catch (const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }

  return 0;
}

namespace kaldi {

bool WriteFull(int32 desc, char* data, int32 size) {
  int32 to_write = size;
  int32 wrote = 0;
  while (to_write > 0) {
    int32 ret = write(desc, data + wrote, to_write);
    if (ret <= 0)
      return false;

    to_write -= ret;
    wrote += ret;
  }

  return true;
}

int32 buffer_offset = 0;
int32 buffer_fill = 0;
char read_buffer[1025];

bool ReadLine(int32 desc, std::string* str) {
  *str = "";

  while (true) {
    if (buffer_offset >= buffer_fill) {
      buffer_fill = read(desc, read_buffer, 1024);

      if (buffer_fill <= 0)
        break;

      buffer_offset = 0;
    }

    for (int32 i = buffer_offset; i < buffer_fill; i++) {
      if (read_buffer[i] == '\r' || read_buffer[i] == '\n') {
        read_buffer[i] = 0;
        *str += (read_buffer + buffer_offset);

        buffer_offset = i + 1;

        if (i < buffer_fill) {
          if (read_buffer[i] == '\n' && read_buffer[i + 1] == '\r') {
            read_buffer[i + 1] = 0;
            buffer_offset = i + 2;
          }
          if (read_buffer[i] == '\r' && read_buffer[i + 1] == '\n') {
            read_buffer[i + 1] = 0;
            buffer_offset = i + 2;
          }
        }

        return true;
      }
    }

    read_buffer[buffer_fill] = 0;
    *str += (read_buffer + buffer_offset);
    buffer_offset = buffer_fill;
  }

  return false;
}

std::string TimeToTimecode(float time) {

  char buf[64];

  int32 h, m, s, ms;
  s = (int32) time;
  ms = (int32)((time - (float) s) * 1000.0f);
  m = s / 60;
  s %= 60;
  h = m / 60;
  m %= 60;

  snprintf(buf, 64, "%02d:%02d:%02d.%03d", h, m, s, ms);

  return buf;
}

}  //namespace kaldi
