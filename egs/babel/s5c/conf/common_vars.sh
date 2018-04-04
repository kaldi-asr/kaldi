#keyword search default
glmFile=conf/glm
duptime=0.5
case_insensitive=false
use_pitch=true
# Lexicon and Language Model parameters
oovSymbol="<unk>"
lexiconFlags="-oov <unk>"
boost_sil=1.5 #  note from Dan: I expect 1.0 might be better (equivalent to not
              # having the option)... should test.
cer=0

#Declaring here to make the definition inside the language conf files more
# transparent and nice
declare -A dev10h_more_kwlists
declare -A dev2h_more_kwlists
declare -A eval_more_kwlists
declare -A shadow_more_kwlists

[ -f ./path.sh ] && . ./path.sh; # source the path.
[ -f ./cmd.sh ] && . ./cmd.sh; # source train and decode cmds.
