#ifndef KALDI_FSTEXT_OPENFST_COMPAT_H
#define KALDI_FSTEXT_OPENFST_COMPAT_H


#if OPENFST_VER < 10800
#define FST_FLAGS_fst_weight_separator FLAGS_fst_weight_separator
#define FST_FLAGS_fst_field_separator FLAGS_fst_field_separator
#define FST_FLAGS_v FLAGS_v

#endif

namespace fst {
#if OPENFST_VER >= 10800


template <typename... Args>
auto Map(Args&&... args) -> decltype(ArcMap(std::forward<Args>(args)...)) {
  return ArcMap(std::forward<Args>(args)...);
}

using MapFstOptions=ArcMapFstOptions;

template <class A, class B, class C>
using MapFst = ArcMapFst<A, B, C>;

template<typename Printer, typename Stream>
void printer_print(Stream &os, Printer &printer, const std::string &s) {
  printer.Print(os, s);
}

#else

template<typename Printer, typename Stream>
void printer_print(Stream &os, Printer &printer, const std::string &s) {
  printer.Print(&os, s);
}

#endif

}  // namespace fst

#endif  //KALDI_FSTEXT_OPENFST_COMPAT_H
