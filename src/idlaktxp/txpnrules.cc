// idlaktxp/txpnrules.cc

// Copyright 2012 CereProc Ltd.  (Author: Matthew Aylett)

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
//

#include "idlaktxp/txpnrules.h"
#include "idlaktxp/txputf8.h"

namespace kaldi {

TxpNRules::TxpNRules()
    : incdata_(false),
    cdata_buffer_(""),
    lkp_item_(NULL),
    lkp_open_(NULL),
    rgxwspace_(NULL),
    rgxsep_(NULL),
    rgxpunc_(NULL),
    rgxalpha_(NULL),
    rgxwspace_default_(NULL),
    rgxsep_default_(NULL),
    rgxpunc_default_(NULL),
    rgxalpha_default_(NULL) {
}

TxpNRules::~TxpNRules() {
  RgxMap::iterator it;
  for (it = rgxs_.begin(); it != rgxs_.end(); it++) {
    pcre_free(it->second);
  }
  pcre_free(const_cast<pcre *>(lkp_item_));
  pcre_free(const_cast<pcre *>(lkp_open_));
  pcre_free(const_cast<pcre *>(rgxwspace_default_));
  pcre_free(const_cast<pcre *>(rgxsep_default_));
  pcre_free(const_cast<pcre *>(rgxpunc_default_));
  pcre_free(const_cast<pcre *>(rgxalpha_default_));
}

void TxpNRules::Init(const TxpParseOptions &opts, const std::string &name) {
  TxpPcre pcre;
  TxpXmlData::Init(opts, "nrules", name);
  lkp_item_ = pcre.Compile("[\n\\s]*(u?[\\'\\\"](.*?)[\\'\\\"]\\s*:\\s*u?[\\'\\\"](.*?)[\\'\\\"])[\n\\s]*[,}][\n\\s]*");// NOLINT
  lkp_open_ = pcre.Compile("[\n\\s]*{");

  rgxwspace_ = rgxwspace_default_ = pcre.Compile("^([ \n\t\r]+)");
  rgxsep_ = rgxsep_default_ = pcre.Compile("^([\\-\\+\\=\\/@]$)");
  rgxpunc_ = rgxpunc_default_ = pcre.Compile("^([\\(\\)\\[\\]\\{\\}\\\"'!\\?\\.,;:\\|]*)([^\\(\\)\\[\\]\\{\\}\\\"!\\?\\.,;:\\|]*)([\\(\\)\\[\\]\\{\\}\\\"'!\\?\\.,;:\\|]*)");// NOLINT
  rgxalpha_ = rgxalpha_default_ = pcre.Compile("^[a-zA-Z_']+$");
  MakeLkp(&locallkps_, "downcase", "{\"A\":\"a\", \"B\":\"b\", \"C\":\"c\", \"D\":\"d\", \"E\":\"e\", \"F\":\"f\", \"G\":\"g\", \"H\":\"h\", \"I\":\"i\", \"J\":\"j\", \"K\":\"k\", \"L\":\"l\", \"M\":\"m\", \"N\":\"n\", \"O\":\"o\", \"P\":\"p\", \"Q\":\"q\", \"R\":\"r\", \"S\":\"s\", \"T\":\"t\", \"U\":\"u\", \"V\":\"v\", \"W\":\"w\", \"X\":\"x\", \"Y\":\"y\", \"Z\":\"z\"}");// NOLINT
  MakeLkp(&locallkps_, "convertillegal", "{\"À\":\"A\", \"Á\":\"A\", \"Â\":\"A\", \"Ã\":\"A\", \"Å\":\"A\", \"Æ\":\"AE\", \"à\":\"a\", \"á\":\"a\", \"â\":\"a\", \"ã\":\"a\", \"å\":\"a\", \"æ\":\"ae\", \"Ç\":\"C\", \"ç\":\"c\", \"È\":\"E\", \"É\":\"E\", \"Ê\":\"E\", \"Ë\":\"E\", \"è\":\"e\", \"é\":\"e\", \"ê\":\"e\", \"ë\":\"e\", \"Ì\":\"I\", \"Í\":\"I\", \"Î\":\"I\", \"Ï\":\"I\", \"ì\":\"i\", \"í\":\"i\", \"î\":\"i\", \"ï\":\"i\", \"Ñ\":\"N\", \"ñ\":\"n\", \"Ò\":\"O\", \"Ó\":\"O\", \"Ô\":\"O\", \"Õ\":\"O\", \"Ø\":\"O\", \"ò\":\"o\", \"ó\":\"o\", \"ô\":\"o\", \"õ\":\"o\", \"ø\":\"o\", \"Ù\":\"U\", \"Ú\":\"U\", \"Û\":\"U\", \"Ű\":u\"Ü\", \"ù\":\"u\", \"ú\":\"u\", \"û\":\"u\", \"ű\":u\"ü\", \"Ý\":\"Y\", \"ý\":\"y\"}");// NOLINT
  MakeLkp(&locallkps_, "utfpunc2ascii", "{\"‘\":\"'\", \"’\":\"'\", \"‛\":\"'\", '“':'\"', '”':'\"',\"΄\":\"'\", \"´\":\"'\", \"`\":\"'\", \"…\":\".\", \"„\":'\"', '–':'-', '–':'-', '–':'-', '—':'-', \"＇\":\"'\"}");// NOLINT
  MakeLkp(&locallkps_, "symbols", "{\"\\\":\"backslash\", \"_\": \"underscore\", \"*\":\"asterisk\", \"#\":\"hash\", \"@\":\"at\", \".\":\"dot\", \"~\":\"tilde\", \"%\":\"percent\", \"<\":\"less than\", \">\":\"more than\", \"=\":\"equals\", \"&\":\"ampersand\", \"+\":\"plus\", \"÷\": \"divided by\", \"¢\": \"cents\", \":\":\"colon\", \"/\":\"slash\", \",\":\"comma\", \"°\":\"degrees\", \"?\":\"question mark\", \"!\":\"exclamation mark\", \"½\":\"half\", \"¼\":\"quarter\", \"¾\":\"three quarters\", \"©\":\"copyright\", \"²\":\"squared\", \"³\":\"cubed\", \"×\":\"times\", \"·\":\"point\", \"√\":\"square root\"}");// NOLINT
  MakeLkp(&locallkps_, "asdigits", "\'0\':\'oh\', \'1\':\'one\', \'2\':\'two\', \'3\':\'three\', \'4\':\'four\', \'5\':\'five\', \'6\':\'six\', \'7\':\'seven\', \'8\':\'eight\', \'9\':\'nine\'}");// NOLINT
}


bool TxpNRules::Parse(const std::string &tpdb) {
  const pcre* rgx;
  bool r;
  r = TxpXmlData::Parse(tpdb);
  if (r) {
    rgx = GetRgx("whitespace");
    if (rgx) rgxwspace_ = rgx;
    rgx = GetRgx("separators");
    if (rgx) rgxsep_ = rgx;
    // Current punc_strip_full in tpdb not appropriate use hard coded version
    // rgx = GetRgx("punc_strip_full");
    // if (rgx) rgxpunc_ = rgx;
    rgx = GetRgx("alpha");
    if (rgx) rgxalpha_ = rgx;
  } else {
    KALDI_WARN << "Error reading normaliser rule file: " << tpdb;
  }
  return r;
}

const std::string* TxpNRules::Lkp(const std::string & name,
                                   const std::string & key) {
  std::string namekey;
  LookupMap::iterator it;
  namekey = name + ":" + key;
  it = lkps_.find(namekey);
  if (it != lkps_.end()) {
    return &(it->second);
  } else {
    it = locallkps_.find(namekey);
    if (it != locallkps_.end()) {
      return &(it->second);
    } else {
      return NULL;
    }
  }
}

const pcre* TxpNRules::GetRgx(const std::string & name) {
  RgxMap::iterator it;
  it = rgxs_.find(name);
  if (it != rgxs_.end())
    return it->second;
  else
    return NULL;
}

// get intial token and following whitespace from input and return the rest
const char* TxpNRules::ConsumeToken(const char* input,
                                     std::string* token,
                                     std::string* wspace) {
  TxpPcre pcre;
  TxpUtf8 utf8;
  int32 clen;
  const char *p, *s, *w, *pre_s;
  // Check for initial whitespace
  *wspace = "";
  *token = "";
  w = pcre.Consume(rgxwspace_, input);
  if (w) {
    pcre.SetMatch(0, wspace);
    return w;
  }
  // Check for initial separator + whitespace
  clen = utf8.Clen(input);
  s =  pcre.Consume(rgxsep_, input, clen);
  if (s) {
    pcre.SetMatch(0, token);
    w = pcre.Consume(rgxwspace_, s);
    if (w) {
      pcre.SetMatch(0, wspace);
      return w;
    } else {
      return s;
    }
  }
  // Go through utf8 chars until we find whitespace or a separator
  p = input;
  while (*p) {
    clen = utf8.Clen(p);
    token->append(p, clen);
    // increment to next char
    p += clen;
    w = pcre.Consume(rgxwspace_, p);
    if (w) {
      pcre.SetMatch(0, wspace);
      return w;
    }
    pre_s = p;
    clen = utf8.Clen(p);
    s =  pcre.Consume(rgxsep_, p, clen);
    if (s) return pre_s;  // Next token
  }
  return p;
}

void TxpNRules::ReplaceUtf8Punc(const std::string & tkin,
                                std::string* tkout) {
  const char* p;
  const std::string* r;
  TxpUtf8 utf8;
  int32 clen;
  p = tkin.c_str();
  *tkout = "";
  while (*p) {
    clen = utf8.Clen(p);
    r = Lkp(std::string("utfpunc2ascii"), std::string(p, clen));
    if (r)
      tkout->append(*r);
    else
      tkout->append(p, clen);
    p += clen;
  }
}

const char* TxpNRules::ConsumePunc(const char* input,
                                    std::string* prepunc,
                                    std::string* token,
                                    std::string* pstpunc) {
  TxpPcre pcre;
  const char* p;
  p = pcre.Consume(rgxpunc_, input);
  if (p) {
    pcre.SetMatch(0, prepunc);
    pcre.SetMatch(1, token);
    pcre.SetMatch(2, pstpunc);
    return p;
  } else {
    *prepunc = input;
    *token = "";
    *pstpunc = "";
    return input += strlen(input);
  }
}

void TxpNRules::NormCaseCharacter(std::string* norm, TxpCaseInfo & caseinfo) {
  const char* p;
  const std::string* r;
  std::string c;
  std::string result;
  int32 clen, j = 0;
  bool alpha;
  TxpUtf8 utf8;
  TxpPcre pcre;
  p = norm->c_str();
  while (*p) {
    alpha = false;
    clen = utf8.Clen(p);
    c = std::string(p, clen);
    if (pcre.Execute(GetRgx(std::string("alpha")), c)) {
      alpha = true;
      caseinfo.lowercase = true;
    }
    r = Lkp(std::string("convertillegal"), c);
    if (r) {
      c = *r;
      caseinfo.foreign = true;
      alpha = true;
    }
    r = Lkp(std::string("downcase"), c);
    if (r) {
      caseinfo.uppercase = true;
      alpha = true;
      c = *r;
    }
    if (r && j > 0) caseinfo.capitalised = false;
    if (!r && j == 0) caseinfo.capitalised = false;
    if (!alpha) {
      caseinfo.symbols = true;
      caseinfo.capitalised = false;
    }
    result.append(c);
    p += clen;
    j += 1;
  }
  *norm = result;
}

bool TxpNRules::IsAlpha(const std::string &token) {
  TxpPcre pcre;
  if (pcre.Execute(rgxalpha_, token.c_str())) return true;
  return false;
}

void TxpNRules::StartElement(const char* name, const char** atts) {
  // save name and type for procesing cdata
  if (!strcmp(name, "lookup") || !strcmp(name, "regex")) {
    elementtype_ = name;
    SetAttribute("name", atts, &elementname_);
  }
}

void TxpNRules::EndElement(const char* name) {
}

void TxpNRules::StartCData() {
  cdata_buffer_.clear();
  incdata_ = true;
}

// TODO(MPA): Add checking (i.e. max match on regex, duplications, empty keys)
void TxpNRules::EndCData() {
  const char* error;
  int erroffset;
  incdata_ = false;
  if (elementtype_ == "lookup") {
    MakeLkp(&lkps_, elementname_, cdata_buffer_);
  } else if (elementtype_ == "regex") {
    rgxs_.insert(RgxItem(elementname_, pcre_compile(cdata_buffer_.c_str(),
                                                    PCRE_UTF8, &error,
                                                    &erroffset, NULL)));
  }
}

void TxpNRules::CharHandler(const char* data, int32 len) {
  if (incdata_) cdata_buffer_.append(data, len);
}

// private member to format lookup table from xml cdata
int32 TxpNRules::MakeLkp(LookupMap *lkps,
                         const std::string &name,
                         const std::string &cdata) {
  TxpPcre pcre;
  int32 pplen;
  std::string key;
  std::string val;
  std::string tmp;
  std::string namekey;
  const char* pp, *npp;
  pp = cdata.c_str();
  pplen = cdata.length();
  // consume opening {
  pp = pcre.Consume(lkp_open_, pp, pplen);
  // no bracket! ignore
  if (!pp) pp = cdata.c_str();
  while (*pp) {
    npp = pcre.Consume(lkp_item_, pp, pplen);
    if (npp) {
      pcre.SetMatch(1, &key);
      pcre.SetMatch(2, &val);
      namekey = name + ":" + key;
      lkps->insert(LookupItem(namekey, val));
      pplen -= npp - pp;
      pp = npp;
    } else {
      return false;
    }
  }
  return true;
}



}  // namespace kaldi
