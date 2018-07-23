There are two top-level scripts in this directory which demonstrate
end-to-end (specifically single-stage flat-start) LF-MMI (i.e. chain)
training. "run_end2end_phone.sh" is basically like "../run.sh"
except it does not train any GMM or SGMM models and after doing data/dict
preparation and feature extraction goes straight to flat-start chain training.
It uses a phoneme-based lexicon just like "../run.sh" does.
"run_end2end_char.sh" is exactly like "run_end2end_phone.sh" excpet it
uses a trivial grapheme-based (i.e. character-based) lexicon.
For details please see the comments at top of local/chain/e2e/run_flatstart_*.sh
and also src/chain/chain-generic-numerator.h.
