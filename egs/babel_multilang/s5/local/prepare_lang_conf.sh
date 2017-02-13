#!/bin/bash
# This script maps lang-name to its config w.r.t fullLP or limitedLP condition.

fullLP=true
. ./utils/parse_options.sh

if [ $# -ne 1 ]; then
  echo "Usage: $(basename $0)  <lang>"
  echo " e.g.: $(basename $0)  101-cantonese"
  exit 1
fi

lang=$1
echo lang = $lang and fullLP = $fullLP
if $fullLP; then
  lang_type=-fullLP
  lang_type2=.FLP
else
  lang_type=-limitedLP
  lang_type2=.LLP
fi

case "$lang" in
  101-cantonese)
    langconf=conf/lang/101-cantonese${lang_type}.official.conf
    ;;
  102-assamese)
    langconf=conf/lang/102-assamese${lang_type}.official.conf
    ;;
  103-bengali)
    langconf=conf/lang/103-bengali${lang_type}.official.conf
    ;;
  104-pashto)
    langconf=conf/lang/104-pashto${lang_type}.official.conf
    ;;
  105-turkish)
    langconf=conf/lang/105-turkish${lang_type}.official.conf
    ;;
  106-tagalog)
    langconf=conf/lang/106-tagalog${lang_type}.official.conf
    ;;
  107-vietnamese)
    langconf=conf/lang/107-vietnamese${lang_type}.official.conf
    ;;
  201-haitian)
    langconf=conf/lang/201-haitian${lang_type}.official.conf
    ;;
  202-swahili)
    langconf=conf/lang/202-swahili${lang_type}.official.conf
    ;;
  203-lao)
    langconf=conf/lang/203-lao${lang_type}.official.conf
    ;;
  204-tamil)
    langconf=conf/lang/204-tamil${lang_type}.official.conf
    ;;
  205-kurmanji)
    langconf=conf/lang/205-kurmanji${lang_type2}.official.conf
    ;;
  206-zulu)
    langconf=conf/lang/206-zulu-${lang_type}.official.conf
    ;;
  207-tokpisin)
    langconf=conf/lang/207-tokpisin${lang_type2}.official.conf
    ;;
  301-cebuano)
    langconf=conf/lang/301-cebuano${lang_type2}.official.conf
    ;;
  302-kazakh)
    langconf=conf/lang/302-kazakh${lang_type2}.official.conf
    ;;
  303-telugu)
    langconf=conf/lang/303-telugu${lang_type2}.official.conf
    ;;
  304-lithuanian)
    langconf=conf/lang/304-lithuanian${lang_type2}.official.conf
    ;;
  305-guarani)
    langconf=conf/lang/305-guarani${lang_type2}.official.conf
    ;;
  306-igbo)
    langconf=conf/lang/306-igbo${lang_type2}.official.conf
    ;;
  307-amharic)
    langconf=conf/lang/307-amharic${lang_type2}.official.conf
    ;;
  401-mongolian)
    langconf=conf/lang/401-mongolian${lang_type2}.official.conf
    ;;
  402-javanese)
    langconf=conf/lang/402-javanese${lang_type2}.official.conf
    ;;
  403-dholuo)
    langconf=conf/lang/403-dholuo${lang_type2}.official.conf
    ;;
  404-georgian)
    langconf=conf/lang/404-georgian.FLP.official.conf
    ;;
  *)
    echo "Unknown language code $lang." && exit 1
esac

mkdir -p langconf/$lang
rm -rf langconf/$lang/*
cp $langconf langconf/$lang/lang.conf

