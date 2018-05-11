#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fvector/fvector-perturb.h"
#include "feat/wave-reader.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-simple-component.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
      "This binary is used to check the filter bank which is modeled by affine\n"
      "component. It computes the band-with of each learned filter."  
      "Usage:  fvector-debug-check-filter-bank [options...] <nnet-in> <component-name> <stats-out>\n";

    ParseOptions po(usage);

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
                component_name = po.GetArg(2),
                stats_wxfilename = po.GetArg(3);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);
    int32 component_index = nnet.GetComponentIndex(component_name);
    Matrix<BaseFloat> filter_bank(
        dynamic_cast<AffineComponent*>(nnet.GetComponent(component_index))->LinearParams());
    std::ofstream out;
    out.open(stats_wxfilename, std::ios::out);
    if (!out.is_open()) { 
      std::cout << "File open error." << std::endl;
      return -1;
    }
    int32 num_rows = filter_bank.NumRows();
    int32 num_columns = filter_bank.NumCols();
    out << "Number of rows: " << num_rows << std::endl;
    out << "Number of columns: " << num_columns << std::endl;
    // Each row can be regard as a filter.
    for (MatrixIndexT i = 0; i < num_rows; i++) {
      const SubVector<BaseFloat> current_row = filter_bank.Row(i);
      BaseFloat current_sum_2 = VecVec(current_row, current_row);
      BaseFloat current_max_2 = current_row.Max() * current_row.Max();
      BaseFloat band_with = current_sum_2 / current_max_2;
      out << "Filter " << i+1 << ": Quadratic Sum is " << current_sum_2 
          << " ;The square of max value is " << current_max_2
          << " ;Band with is " << band_with << std::endl;
    } 

    out.close();
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
