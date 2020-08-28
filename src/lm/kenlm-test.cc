#include "util/text-utils.h"
#include "util/kaldi-io.h"
#include "lm/kenlm.h"

namespace kaldi {

void UnitTestKenLm() {
  KenLm lm;
  lm.Load("test_data/lm.kenlm", "test_data/words.txt");
  lm.WriteReindex("test_data/reindex.txt");

  std::ifstream is("test_data/sentences.txt");
  std::string sentence;
  while(std::getline(is, sentence)) {
    std::vector<std::string> words;
    SplitStringToVector(sentence, " ", true, &words);

    KenLm::State state[2];
    KenLm::State* istate = &state[0];
    KenLm::State* ostate = &state[1];

    lm.SetStateToBos(istate);
    
    std::string sentence_log;
    for (int i = 0; i < words.size(); i++) {
      std::string word = words[i];
      BaseFloat word_score = lm.Score(istate, lm.GetWordIndex(word), ostate);
      sentence_log += " " + word + ":" + std::to_string(lm.GetWordIndex(word)) + ":" + std::to_string(word_score);
      std::swap(istate, ostate);
    }
    KALDI_LOG << sentence_log;
  }
}

} // namespace kaldi

int main(int argc, char *argv[]) {
  using namespace kaldi;
  UnitTestKenLm();
  return 0;
}
