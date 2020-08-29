#include "util/text-utils.h"
#include "util/kaldi-io.h"
#include "lm/kenlm.h"

namespace kaldi {

void UnitTestKenLm() {
  KenLm lm;
  lm.Load("test_data/lm.kenlm", "test_data/words.txt"); // words.txt is Kaldi's output symbol table
  std::ifstream is("test_data/sentences.txt");
  std::string sentence;
  while(std::getline(is, sentence)) {
    std::vector<std::string> words;
    SplitStringToVector(sentence, " ", true, &words);

    KenLm::State state[2];
    KenLm::State* istate = &state[0];
    KenLm::State* ostate = &state[1];

    lm.SetStateToBeginOfSentence(istate);
    
    std::string sentence_log;
    for (int i = 0; i < words.size(); i++) {
      std::string w = words[i];
      // note here KenLM gives log10 score, not natural log
      BaseFloat score = lm.Score(istate, lm.GetWordIndex(w), ostate);
      sentence_log += " " + w +
        ":" + std::to_string(lm.GetWordIndex(w)) +
        ":" + std::to_string(score);
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
